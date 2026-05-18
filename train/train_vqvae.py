import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hydra
import torch
import torch.distributed as dist
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.image.fid import FrechetInceptionDistance
import tqdm
import wandb

from train.utils import (
    aggregate_metrics,
    all_reduce_mean,
    build_scheduler,
    check_stop_file,
    checkpoint_state,
    cleanup_distributed,
    get_device,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
    move_to_device,
    prepare_metrics_for_log,
    save_checkpoint,
    set_seed,
)

STOP_FILE = ROOT / 'STOP'
from data import AtariEpisodeDataset
from tokenizer import (
    NLayerDiscriminator,
    VQVAE,
    adaptive_disc_weight,
    discriminator_hinge_loss,
    discriminator_metrics,
    generator_hinge_loss,
    generator_metrics,
    r1_gradient_penalty,
    vqvae_metrics,
)


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


VAL_RECONSTRUCTION_EVERY_STEPS = 500


def batch_frames(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch


def make_loaders(cfg):
    dataset = AtariEpisodeDataset(
        h5_path=to_absolute_path(cfg.data.h5_path),
        seq_len=cfg.data.seq_len,
    )

    n_train = int(len(dataset) * cfg.data.train_frac)
    n_val = int(len(dataset) * cfg.data.val_frac)
    n_test = len(dataset) - n_train - n_val
    if n_train <= 0 or n_val < 0 or n_test < 0:
        raise ValueError('Invalid dataset split sizes from train_frac/val_frac')

    generator = torch.Generator().manual_seed(cfg.train.seed)
    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator,
    )

    world_size = get_world_size()
    rank = get_rank()

    def sampler(subset, shuffle):
        if world_size > 1:
            return DistributedSampler(subset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        return None

    train_sampler = sampler(train_set, shuffle=True)
    val_sampler = sampler(val_set, shuffle=False)
    test_sampler = sampler(test_set, shuffle=False)

    loader_kwargs = {
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.train.num_workers,
        'pin_memory': torch.cuda.is_available(),
    }
    if cfg.train.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 4
    train_loader = DataLoader(
        train_set,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_set, shuffle=False, sampler=val_sampler, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, sampler=test_sampler, **loader_kwargs)
    return train_loader, val_loader, test_loader


def fid_images(images):
    images = images.detach().clamp(0, 1)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    return images


@torch.no_grad()
def collect_encoder_vectors(model, loader, device, num_batches):
    was_training = model.training
    model.eval()
    vectors = []
    total = min(len(loader), num_batches)

    for batch_idx, batch in enumerate(tqdm.tqdm(loader, total=total, desc='k-means init batches', disable=not is_main_process())):
        if batch_idx >= num_batches:
            break

        frames = move_to_device(batch_frames(batch), device)
        frames = model.flatten_frames(frames)
        z = model.encoder(frames)
        z = F.normalize(z, p=2, dim=1)
        z = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
        vectors.append(z)

    model.train(was_training)
    if not vectors:
        raise ValueError('Need model.codebook.init.num_batches > 0 for k-means init')
    return torch.cat(vectors, dim=0)


def nearest_centroids(vectors, centroids, chunk_size):
    assignments = []
    for start in range(0, vectors.shape[0], chunk_size):
        chunk = vectors[start:start + chunk_size]
        distances = torch.cdist(chunk, centroids)
        assignments.append(distances.argmin(dim=1))
    return torch.cat(assignments, dim=0)


@torch.no_grad()
def init_codebook_kmeans(model, loader, device, cfg):
    num_embeddings = model.quantizer.K
    dim = model.quantizer.D

    if is_main_process():
        vectors = collect_encoder_vectors(
            model,
            loader,
            device,
            num_batches=cfg.model.codebook.init.num_batches,
        )

        if vectors.shape[0] < num_embeddings:
            raise ValueError(
                f'Need at least {num_embeddings} encoder vectors for k-means init, '
                f'but only collected {vectors.shape[0]}. Increase model.codebook.init.num_batches.'
            )

        indices = torch.randperm(vectors.shape[0], device=vectors.device)[:num_embeddings]
        centroids = vectors[indices].clone()

        for _ in tqdm.tqdm(range(cfg.model.codebook.init.num_iters), desc='k-means init iterations'):
            assignments = nearest_centroids(
                vectors,
                centroids,
                chunk_size=cfg.model.codebook.init.chunk_size,
            )
            for code_idx in range(num_embeddings):
                members = vectors[assignments == code_idx]
                if len(members) > 0:
                    centroids[code_idx] = F.normalize(members.mean(dim=0), p=2, dim=0)
                else:
                    replacement_idx = torch.randint(vectors.shape[0], (1,), device=vectors.device).item()
                    centroids[code_idx] = vectors[replacement_idx]
        centroids = F.normalize(centroids, p=2, dim=1).contiguous()
    else:
        centroids = torch.empty(num_embeddings, dim, device=device)

    if is_distributed():
        dist.broadcast(centroids, src=0)

    model.quantizer.embedding.weight.copy_(centroids)
    model.quantizer.reset_ema_state()


def all_reduce_metrics(metrics):
    if not is_distributed():
        return metrics
    reduced = {}
    for key, value in metrics.items():
        if torch.is_tensor(value) and value.numel() == 1 and value.is_floating_point():
            value = value.detach().clone()
            all_reduce_mean(value)
            reduced[key] = value
        elif isinstance(value, (int, float)):
            tensor = torch.tensor(float(value), device='cuda' if torch.cuda.is_available() else 'cpu')
            all_reduce_mean(tensor)
            reduced[key] = tensor.item()
        else:
            reduced[key] = value
    return reduced


def set_requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad_(requires_grad)


def gan_active(disc, step, disc_cfg):
    return disc is not None and step >= disc_cfg['warmup_steps']


def gan_generator_step(disc, out, frames, raw_model, disc_cfg, perceptual_weight, split):
    set_requires_grad(disc, False)
    fake_logits = disc(out['pred'])
    g_loss = generator_hinge_loss(fake_logits)

    if disc_cfg['adaptive']:
        nll = out['recon_loss'] + perceptual_weight * out['perceptual_loss']
        d_w = adaptive_disc_weight(nll, g_loss, raw_model.decoder.net[-2].weight)
    else:
        d_w = torch.tensor(1.0, device=frames.device)

    weight = d_w * disc_cfg['weight']
    extra_loss = weight * g_loss
    return extra_loss, generator_metrics(split, g_loss, weight)


def gan_discriminator_step(disc, disc_optimizer, out, frames, disc_cfg, split):
    set_requires_grad(disc, True)
    use_r1 = disc_cfg['r1_weight'] > 0
    real = frames.detach().requires_grad_(use_r1)
    real_logits = disc(real)
    fake_logits_d = disc(out['pred'].detach())
    d_loss = discriminator_hinge_loss(real_logits, fake_logits_d)

    r1 = None
    if use_r1:
        r1 = r1_gradient_penalty(real_logits, real)
        d_loss = d_loss + 0.5 * disc_cfg['r1_weight'] * r1

    disc_optimizer.zero_grad()
    d_loss.backward()
    disc_optimizer.step()
    return discriminator_metrics(split, d_loss, real_logits, fake_logits_d, r1_penalty=r1)


def run_epoch(
    model,
    loader,
    split,
    device,
    optimizer=None,
    run=None,
    step=0,
    include_reconstructions=False,
    fid=None,
    step_callback=None,
    epoch=None,
    disc=None,
    disc_optimizer=None,
    disc_cfg=None,
    perceptual_weight=0.0,
    log_every_steps_override=20,
):
    is_train = optimizer is not None
    model.train(is_train)
    if disc is not None:
        disc.train(is_train)
    metrics_list = []
    desc = f'[epoch {epoch}] {split} batches' if epoch is not None else f'{split} batches'
    raw_model = unwrap(model)
    raw_disc = unwrap(disc) if disc is not None else None
    log_every_steps = log_every_steps_override if is_train else 0

    recon_batch_idx = None
    if include_reconstructions and is_main_process() and len(loader) > 0:
        recon_batch_idx = int(torch.randint(len(loader), (1,)).item())

    for batch_idx, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc=desc, disable=not is_main_process())):
        frames = move_to_device(batch_frames(batch), device)
        frames = raw_model.flatten_frames(frames)

        with torch.set_grad_enabled(is_train):
            out = model(frames)
            disc_metrics = {}

            if is_train:
                active = gan_active(disc, step, disc_cfg)
                gen_loss = out['loss']

                if active:
                    extra, gen_metrics = gan_generator_step(
                        disc, out, frames, raw_model, disc_cfg, perceptual_weight, split
                    )
                    gen_loss = gen_loss + extra
                    disc_metrics.update(gen_metrics)

                optimizer.zero_grad()
                gen_loss.backward()
                optimizer.step()

                if active and disc_cfg is not None and step % disc_cfg['update_every'] == 0:
                    disc_metrics.update(
                        gan_discriminator_step(disc, disc_optimizer, out, frames, disc_cfg, split)
                    )

                step += 1

        if is_train:
            if log_every_steps > 0 and step % log_every_steps == 0:
                metrics = vqvae_metrics(split, out, frames, num_embeddings=raw_model.quantizer.K)
                metrics[f'{split}/lr'] = optimizer.param_groups[0]['lr']
                metrics.update(disc_metrics)
                if run is not None and is_main_process():
                    run.log(prepare_metrics_for_log(metrics), step=step)
            if step_callback is not None:
                step_callback(step)
                model.train(True)
                if disc is not None:
                    disc.train(True)
        else:
            metrics = vqvae_metrics(
                split,
                out,
                frames,
                num_embeddings=raw_model.quantizer.K,
                include_reconstructions=(batch_idx == recon_batch_idx),
                reconstruction_examples=1,
            )
            if fid is not None:
                fid.update(fid_images(frames), real=True)
                fid.update(fid_images(out['pred']), real=False)
            metrics_list.append(metrics)

    if not is_train:
        metrics = aggregate_metrics(metrics_list)
        metrics = all_reduce_metrics(metrics)
        if fid is not None:
            metrics[f'{split}/fid'] = fid.compute()
        return metrics, step
    return {}, step


@hydra.main(version_base=None, config_path='../configs', config_name='vqvae')
def main(cfg: DictConfig):
    local_rank, _ = init_distributed()
    set_seed(cfg.train.seed + get_rank())
    device = get_device(cfg.train.device)
    if is_distributed() and torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')

    try:
        _run(cfg, device, local_rank)
    finally:
        cleanup_distributed()


def _run(cfg, device, local_rank):
    train_loader, val_loader, test_loader = make_loaders(cfg)
    model = VQVAE(cfg).to(device)

    if cfg.model.codebook.init.type == 'kmeans':
        init_codebook_kmeans(model, train_loader, device, cfg)
    elif cfg.model.codebook.init.type == 'random':
        if is_distributed():
            dist.broadcast(model.quantizer.embedding.weight, src=0)
            model.quantizer.reset_ema_state()
    else:
        raise ValueError(f'Unsupported codebook init type: {cfg.model.codebook.init.type}')

    if is_distributed():
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            broadcast_buffers=False,
        )

    if cfg.optimizer.type != 'adam':
        raise ValueError(f'Unsupported optimizer type: {cfg.optimizer.type}')
    optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    scheduler = build_scheduler(optimizer, cfg.scheduler)

    disc = None
    disc_optimizer = None
    disc_cfg = None
    perceptual_weight = float(cfg.loss.perceptual_weight)
    if cfg.get('discriminator') is not None and cfg.discriminator.get('enabled', False):
        disc = NLayerDiscriminator(cfg).to(device)
        if is_distributed():
            disc = DDP(
                disc,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                broadcast_buffers=False,
            )
        disc_optimizer = optim.Adam(
            disc.parameters(),
            lr=cfg.discriminator.lr,
            betas=(cfg.discriminator.beta1, cfg.discriminator.beta2),
        )
        disc_cfg = {
            'weight': float(cfg.loss.disc_weight),
            'warmup_steps': int(cfg.loss.disc_warmup_steps),
            'adaptive': bool(cfg.loss.disc_adaptive),
            'r1_weight': float(cfg.loss.get('r1_weight', 0.0)),
            'update_every': int(cfg.discriminator.get('update_every', 1)),
        }

    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    run = None
    if is_main_process():
        run = wandb.init(
            project=str(cfg.exp.project),
            name=str(cfg.exp.run_name),
            group=str(cfg.exp.group),
            entity=str(cfg.exp.entity),
            config=wandb_config,
        )

    val_every_steps = int(cfg.train.get('val_every_steps') or 0)

    try:
        best_model_state = None
        step = 0
        next_val_reconstruction_step = VAL_RECONSTRUCTION_EVERY_STEPS

        def validate(at_epoch, at_step):
            nonlocal best_model_state, next_val_reconstruction_step
            include_val_reconstructions = at_step >= next_val_reconstruction_step
            while at_step >= next_val_reconstruction_step:
                next_val_reconstruction_step += VAL_RECONSTRUCTION_EVERY_STEPS

            with torch.no_grad():
                val_metrics, _ = run_epoch(
                    model,
                    val_loader,
                    'val',
                    device,
                    step=at_step,
                    include_reconstructions=include_val_reconstructions,
                    epoch=at_epoch,
                )
            if is_main_process() and run is not None:
                run.log(prepare_metrics_for_log(val_metrics), step=at_step)

            val_loss = val_metrics.get('val/recon_loss')
            if torch.is_tensor(val_loss):
                val_loss = val_loss.item()
            if is_main_process() and val_loss is not None and (
                best_model_state is None or val_loss < best_model_state['val_loss']
            ):
                best_model_state = checkpoint_state(
                    at_epoch, val_loss, unwrap(model), optimizer, wandb_config
                )

        for epoch in range(1, cfg.train.epochs + 1):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            def step_hook(s, _epoch=epoch):
                if val_every_steps > 0 and s % val_every_steps == 0:
                    validate(_epoch, s)

            _, step = run_epoch(
                model,
                train_loader,
                'train',
                device,
                optimizer=optimizer,
                run=run,
                step=step,
                step_callback=step_hook if val_every_steps > 0 else None,
                epoch=epoch,
                disc=disc,
                disc_optimizer=disc_optimizer,
                disc_cfg=disc_cfg,
                perceptual_weight=perceptual_weight,
                log_every_steps_override=int(cfg.train.get('log_every_steps', 20)),
            )

            if val_every_steps == 0:
                validate(epoch, step)

            if scheduler is not None:
                scheduler.step()

            if check_stop_file(STOP_FILE, device):
                if is_main_process():
                    print(f'[STOP] sentinel detected at {STOP_FILE}, ending training after epoch {epoch}')
                    if STOP_FILE.exists():
                        STOP_FILE.unlink()
                break

        if is_main_process() and best_model_state is not None and run is not None:
            checkpoint_path = Path(run.dir) / 'best_model.pt'
            save_checkpoint(best_model_state, checkpoint_path)

        fid = FrechetInceptionDistance(normalize=True).to(device)
        with torch.no_grad():
            test_metrics, step = run_epoch(
                model,
                test_loader,
                'test',
                device,
                step=step,
                include_reconstructions=True,
                fid=fid,
            )
        if is_main_process() and run is not None:
            run.log(prepare_metrics_for_log(test_metrics), step=step)
    finally:
        if run is not None:
            run.finish()


if __name__ == '__main__':
    main()
