from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h5py
import hydra
import torch
import torch.distributed as dist
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
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
    load_tokenizer,
    move_to_device,
    prepare_metrics_for_log,
    save_checkpoint,
    set_seed,
)

STOP_FILE = ROOT / 'STOP'
from data import AtariEpisodeDataset
from world_model import WorldModel, world_model_loss, world_model_metrics


def unwrap(model):
    while True:
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        elif isinstance(model, DDP):
            model = model.module
        else:
            return model


def make_loaders(cfg):
    dataset = AtariEpisodeDataset(
        h5_path=to_absolute_path(cfg.data.h5_path),
        seq_len=cfg.data.seq_len + 1,  # cfg.data.seq_len = context frames; dataset slices context + 1 target frame
        return_tokens=True,
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


def unpack_batch(batch):
    tokens, actions, *_ = batch
    return tokens, actions


def prepare_batch(batch, device, cfg):
    frame_tokens, actions = unpack_batch(move_to_device(batch, device))
    # tokens come from h5 as (B, T, H, W); flatten the spatial dims into per-frame tokens.
    frame_tokens = frame_tokens.flatten(2).contiguous()

    if frame_tokens.size(1) < 2:
        raise ValueError('World-model training needs data.seq_len >= 1 (window must contain context + target frame)')
    if actions.shape[:2] != frame_tokens.shape[:2]:
        raise ValueError(
            f'Actions must have shape (B, T) matching frame tokens; '
            f'got actions {tuple(actions.shape)} and frame tokens {tuple(frame_tokens.shape)}'
        )

    tokens_per_frame = frame_tokens.size(-1)
    if tokens_per_frame != cfg.model.tokens_per_frame:
        raise ValueError(
            f'World-model config expects {cfg.model.tokens_per_frame} tokens per frame, '
            f'but precomputed cache has {tokens_per_frame}. Update model.tokens_per_frame.'
        )

    return frame_tokens, actions[:, :-1].contiguous()


def model_step(world_model, batch, device, cfg):
    frame_tokens, actions = prepare_batch(batch, device, cfg)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        out = world_model(frame_tokens, actions)
        loss_out = world_model_loss(out, frame_tokens)
    if isinstance(out, dict):
        out = {**out, **loss_out}
    else:
        out = {'frame_logits': out, **loss_out}
    return out, frame_tokens


@torch.no_grad()
def precompute_tokens(tokenizer, h5_path, device, cfg):
    # Tokenize every frame in the h5 once and write per-episode `tokens` datasets.
    # Overwrites any existing tokens so reruns pick up tokenizer changes.
    tokenizer.eval()
    batch_size = int(cfg.train.get('tokenize_batch_size', 512))
    with h5py.File(h5_path, 'r+') as f:
        ep_keys = list(f.keys())
        for ep_key in tqdm.tqdm(ep_keys, desc='precompute tokens', disable=not is_main_process()):
            ep = f[ep_key]
            n = ep['frames'].shape[0]

            # Probe one frame to learn token shape.
            probe = torch.from_numpy(ep['frames'][:1]).to(device=device, dtype=torch.float32).div_(255.0)
            probe = probe.permute(0, 3, 1, 2).contiguous()
            probe_tokens = tokenizer.encode(probe)
            H, W = probe_tokens.shape[-2:]

            if 'tokens' in ep:
                del ep['tokens']
            tokens_ds = ep.create_dataset('tokens', shape=(n, H, W), dtype='int32')

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                frames = torch.from_numpy(ep['frames'][start:end]).to(device=device, dtype=torch.float32).div_(255.0)
                frames = frames.permute(0, 3, 1, 2).contiguous()
                indices = tokenizer.encode(frames)
                tokens_ds[start:end] = indices.to(torch.int32).cpu().numpy()

    if is_main_process():
        print(f'precomputed tokens: tokens_per_frame={H * W} ({H}x{W}), num_frame_tokens={cfg.model.num_frame_tokens}')
    if H * W != cfg.model.tokens_per_frame:
        raise ValueError(
            f'World-model config expects {cfg.model.tokens_per_frame} tokens per frame, '
            f'but tokenizer produced {H * W}. Update model.tokens_per_frame.'
        )


def make_optimizer(model, cfg):
    optimizer_type = cfg.optimizer.type.lower()
    weight_decay = cfg.optimizer.get('weight_decay', 0.0)

    if optimizer_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=weight_decay)
    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=cfg.optimizer.lr, weight_decay=weight_decay)

    raise ValueError(f'Unsupported optimizer type: {cfg.optimizer.type}')


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
    world_model,
    loader,
    split,
    device,
    cfg,
    optimizer=None,
    run=None,
    step=0,
    step_callback=None,
    epoch=None,
):
    is_train = optimizer is not None
    world_model.train(is_train)
    metrics_list = []
    desc = f'[epoch {epoch}] {split} batches' if epoch is not None else f'{split} batches'
    progress = tqdm.tqdm(
        loader,
        total=len(loader),
        desc=desc,
        disable=not is_main_process(),
    )
    grad_clip_norm = cfg.train.get('grad_clip_norm')
    log_every_steps = int(cfg.train.get('log_every_steps', 20)) if is_train else 0
    raw_world_model = unwrap(world_model)

    for batch in progress:
        with torch.set_grad_enabled(is_train):
            out, frame_tokens = model_step(world_model, batch, device, cfg)
            loss = out['loss']

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    clip_grad_norm_(raw_world_model.parameters(), grad_clip_norm)
                optimizer.step()
                step += 1

        if is_train:
            # Only sync the loss back / compute metrics / log on logging steps.
            # Every-step .cpu() and wandb.log serialize the GPU on a tiny model.
            if log_every_steps > 0 and step % log_every_steps == 0:
                metrics = world_model_metrics(split, out, frame_tokens)
                progress.set_postfix(loss=f'{float(loss.detach()):.4f}')
                if run is not None and is_main_process():
                    metrics[f'{split}/lr'] = optimizer.param_groups[0]['lr']
                    run.log(prepare_metrics_for_log(metrics), step=step)
            if step_callback is not None:
                step_callback(step)
                world_model.train(True)
        else:
            metrics = world_model_metrics(split, out, frame_tokens)
            metrics_list.append(metrics)

    if not is_train:
        metrics = aggregate_metrics(metrics_list)
        metrics = all_reduce_metrics(metrics)
        return metrics, step
    return {}, step


@hydra.main(version_base=None, config_path='../configs', config_name='world_model')
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
    tokenizer = load_tokenizer(cfg, device)

    # Tokenize the whole dataset once (rank 0 writes; others wait), then the
    # training loop reads precomputed tokens straight from the h5.
    h5_path = to_absolute_path(cfg.data.h5_path)
    if is_main_process():
        precompute_tokens(tokenizer, h5_path, device, cfg)
    if is_distributed():
        dist.barrier()

    train_loader, val_loader, test_loader = make_loaders(cfg)

    world_model = WorldModel(cfg).to(device)
    # Compile the bare module before wrapping in DDP so DDP's gradient hooks see
    # the real parameters and the compile graph isn't broken by DDP's Python wrappers.
    if device.type == 'cuda':
        world_model = torch.compile(world_model)
    if is_distributed():
        world_model = DDP(
            world_model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            broadcast_buffers=False,
        )

    optimizer = make_optimizer(world_model, cfg)
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

    val_every_steps = int(cfg.train.get('val_every_steps') or 0)

    try:
        best_model_state = None
        step = 0

        def validate(at_epoch, at_step):
            nonlocal best_model_state
            with torch.no_grad():
                val_metrics, _ = run_epoch(
                    world_model,
                    val_loader,
                    'val',
                    device,
                    cfg,
                    step=at_step,
                    epoch=at_epoch,
                )

            if val_metrics and is_main_process() and run is not None:
                val_metrics['epoch'] = at_epoch
                run.log(prepare_metrics_for_log(val_metrics), step=at_step)
                val_loss = val_metrics.get('val/loss')
                if torch.is_tensor(val_loss):
                    val_loss = val_loss.item()
                if val_loss is not None and (
                    best_model_state is None or val_loss < best_model_state['val_loss']
                ):
                    best_model_state = checkpoint_state(
                        at_epoch,
                        val_loss,
                        unwrap(world_model),
                        optimizer,
                        wandb_config,
                    )

        for epoch in range(1, cfg.train.epochs + 1):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            def step_hook(s, _epoch=epoch):
                if val_every_steps > 0 and s % val_every_steps == 0:
                    validate(_epoch, s)

            _, step = run_epoch(
                world_model,
                train_loader,
                'train',
                device,
                cfg,
                optimizer=optimizer,
                run=run,
                step=step,
                step_callback=step_hook if val_every_steps > 0 else None,
                epoch=epoch,
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

        if len(test_loader) > 0:
            with torch.no_grad():
                test_metrics, step = run_epoch(
                    world_model,
                    test_loader,
                    'test',
                    device,
                    cfg,
                    step=step,
                )
            if is_main_process() and run is not None:
                run.log(prepare_metrics_for_log(test_metrics), step=step)
    finally:
        if run is not None:
            run.finish()


if __name__ == '__main__':
    main()
