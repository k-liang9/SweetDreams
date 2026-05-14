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
from data import AtariEpisodeDataset
from tokenizer import VQVAE, vqvae_metrics


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
):
    is_train = optimizer is not None
    model.train(is_train)
    metrics_list = []
    desc = f'{split} batches'
    raw_model = unwrap(model)

    for batch_idx, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc=desc, disable=not is_main_process())):
        frames = move_to_device(batch_frames(batch), device)
        frames = raw_model.flatten_frames(frames)

        with torch.set_grad_enabled(is_train):
            out = model(frames)
            if is_train:
                optimizer.zero_grad()
                out['loss'].backward()
                optimizer.step()
                step += 1

        if is_train:
            metrics = vqvae_metrics(split, out, frames, num_embeddings=raw_model.quantizer.K)
            metrics[f'{split}/lr'] = optimizer.param_groups[0]['lr']
            if run is not None and is_main_process():
                run.log(prepare_metrics_for_log(metrics), step=step)
        else:
            metrics = vqvae_metrics(
                split,
                out,
                frames,
                num_embeddings=raw_model.quantizer.K,
                include_reconstructions=include_reconstructions and not metrics_list and is_main_process(),
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

    try:
        best_model_state = None
        step = 0
        next_val_reconstruction_step = VAL_RECONSTRUCTION_EVERY_STEPS

        for epoch in range(1, cfg.train.epochs + 1):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            _, step = run_epoch(
                model,
                train_loader,
                'train',
                device,
                optimizer=optimizer,
                run=run,
                step=step,
            )

            include_val_reconstructions = step >= next_val_reconstruction_step
            while step >= next_val_reconstruction_step:
                next_val_reconstruction_step += VAL_RECONSTRUCTION_EVERY_STEPS

            with torch.no_grad():
                val_metrics, step = run_epoch(
                    model,
                    val_loader,
                    'val',
                    device,
                    step=step,
                    include_reconstructions=include_val_reconstructions,
                )
            if is_main_process() and run is not None:
                run.log(prepare_metrics_for_log(val_metrics), step=step)

            val_loss = val_metrics.get('val/recon_loss')
            if torch.is_tensor(val_loss):
                val_loss = val_loss.item()
            if is_main_process() and val_loss is not None and (
                best_model_state is None or val_loss < best_model_state['val_loss']
            ):
                best_model_state = checkpoint_state(
                    epoch, val_loss, unwrap(model), optimizer, wandb_config
                )

            if scheduler is not None:
                scheduler.step()

        if is_main_process() and best_model_state is not None and run is not None:
            checkpoint_path = Path(run.dir) / 'best_model.pt'
            save_checkpoint(best_model_state, checkpoint_path)

        fid = FrechetInceptionDistance(normalize=True).to(device) if is_main_process() else None
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
