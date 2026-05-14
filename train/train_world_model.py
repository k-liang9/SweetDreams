from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
import tqdm
import wandb

from train.utils import (
    aggregate_metrics,
    build_scheduler,
    checkpoint_state,
    get_device,
    move_to_device,
    prepare_metrics_for_log,
    save_checkpoint,
    set_seed,
)
from data import AtariEpisodeDataset
from tokenizer import VQVAE
from world_model import WorldModel, world_model_loss, world_model_metrics


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

    loader_kwargs = {
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.train.num_workers,
        'pin_memory': torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def load_tokenizer(cfg, device):
    checkpoint_path = Path(to_absolute_path(cfg.tokenizer.checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    tokenizer_cfg = OmegaConf.create(checkpoint['cfg'])
    tokenizer = VQVAE(tokenizer_cfg)
    tokenizer.load_state_dict(checkpoint['model_state_dict'])
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad_(False)

    if tokenizer.quantizer.K != cfg.model.num_frame_tokens:
        raise ValueError(
            f'World model expects {cfg.model.num_frame_tokens} frame-token classes, '
            f'but VQ-VAE checkpoint has {tokenizer.quantizer.K}'
        )

    return tokenizer.to(device)


def unpack_batch(batch):
    frames, actions, *_ = batch
    return frames, actions


@torch.no_grad()
def tokenize_batch(tokenizer, batch, device, cfg):
    frames, actions = unpack_batch(move_to_device(batch, device))
    frame_tokens = tokenizer.encode(frames).flatten(2).contiguous()

    if frame_tokens.size(1) < 2:
        raise ValueError('World-model training needs data.seq_len >= 2')
    if actions.shape[:2] != frame_tokens.shape[:2]:
        raise ValueError(
            f'Actions must have shape (B, T) matching frame tokens; '
            f'got actions {tuple(actions.shape)} and frame tokens {tuple(frame_tokens.shape)}'
        )

    tokens_per_frame = frame_tokens.size(-1)
    if tokens_per_frame != cfg.model.tokens_per_frame:
        raise ValueError(
            f'World-model config expects {cfg.model.tokens_per_frame} tokens per frame, '
            f'but tokenizer produced {tokens_per_frame}. Update model.tokens_per_frame.'
        )

    seq_len = frame_tokens.size(1) * tokens_per_frame + (frame_tokens.size(1) - 1)
    if seq_len > cfg.model.max_seq_len:
        raise ValueError(
            f'Interleaved token sequence length is {seq_len}, but model.max_seq_len is '
            f'{cfg.model.max_seq_len}. Increase model.max_seq_len.'
        )

    return frame_tokens, actions[:, :-1].contiguous()


def model_step(tokenizer, world_model, batch, device, cfg):
    frame_tokens, actions = tokenize_batch(tokenizer, batch, device, cfg)
    out = world_model(frame_tokens, actions)
    loss_out = world_model_loss(out, frame_tokens)
    if isinstance(out, dict):
        out = {**out, **loss_out}
    else:
        out = {'frame_logits': out, **loss_out}
    return out, frame_tokens


def make_optimizer(model, cfg):
    optimizer_type = cfg.optimizer.type.lower()
    weight_decay = cfg.optimizer.get('weight_decay', 0.0)

    if optimizer_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=weight_decay)
    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=cfg.optimizer.lr, weight_decay=weight_decay)

    raise ValueError(f'Unsupported optimizer type: {cfg.optimizer.type}')


def run_epoch(
    tokenizer,
    world_model,
    loader,
    split,
    device,
    cfg,
    optimizer=None,
    run=None,
    step=0,
):
    is_train = optimizer is not None
    world_model.train(is_train)
    tokenizer.eval()
    metrics_list = []
    progress = tqdm.tqdm(loader, total=len(loader), desc=f'{split} batches')
    log_every = cfg.train.get('log_every', 1)
    grad_clip_norm = cfg.train.get('grad_clip_norm')

    for batch in progress:
        with torch.set_grad_enabled(is_train):
            out, frame_tokens = model_step(tokenizer, world_model, batch, device, cfg)
            loss = out['loss']

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    clip_grad_norm_(world_model.parameters(), grad_clip_norm)
                optimizer.step()
                step += 1

        metrics = world_model_metrics(split, out, frame_tokens)
        progress.set_postfix(loss=f'{float(loss.detach().cpu()):.4f}')

        if is_train:
            should_log = log_every <= 1 or step % log_every == 0
            if run is not None and should_log:
                metrics[f'{split}/lr'] = optimizer.param_groups[0]['lr']
                run.log(prepare_metrics_for_log(metrics), step=step)
        else:
            metrics_list.append(metrics)

    if not is_train:
        return aggregate_metrics(metrics_list), step
    return {}, step


@torch.no_grad()
def validate_tokenizer_shape(tokenizer, loader, device, cfg):
    if len(loader) == 0:
        return

    batch = next(iter(loader))
    frame_tokens, _ = tokenize_batch(tokenizer, batch, device, cfg)
    print(
        f'World-model tokenization: {frame_tokens.size(1)} frames, '
        f'{frame_tokens.size(-1)} tokens/frame, {cfg.model.num_frame_tokens} token classes'
    )


@hydra.main(version_base=None, config_path='../configs', config_name='world_model')
def main(cfg: DictConfig):
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)

    train_loader, val_loader, test_loader = make_loaders(cfg)
    tokenizer = load_tokenizer(cfg, device)
    validate_tokenizer_shape(tokenizer, train_loader, device, cfg)

    world_model = WorldModel(cfg).to(device)
    optimizer = make_optimizer(world_model, cfg)
    scheduler = build_scheduler(optimizer, cfg.scheduler)

    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    with wandb.init(
        project=str(cfg.exp.project),
        name=str(cfg.exp.run_name),
        group=str(cfg.exp.group),
        entity=str(cfg.exp.entity),
        config=wandb_config,
    ) as run:
        best_model_state = None
        step = 0

        for epoch in range(1, cfg.train.epochs + 1):
            _, step = run_epoch(
                tokenizer,
                world_model,
                train_loader,
                'train',
                device,
                cfg,
                optimizer=optimizer,
                run=run,
                step=step,
            )

            with torch.no_grad():
                val_metrics, step = run_epoch(
                    tokenizer,
                    world_model,
                    val_loader,
                    'val',
                    device,
                    cfg,
                    step=step,
                )

            if val_metrics:
                val_metrics['epoch'] = epoch
                run.log(prepare_metrics_for_log(val_metrics), step=step)
                val_loss = val_metrics.get('val/loss')
                if torch.is_tensor(val_loss):
                    val_loss = val_loss.item()
                if val_loss is not None and (
                    best_model_state is None or val_loss < best_model_state['val_loss']
                ):
                    best_model_state = checkpoint_state(
                        epoch,
                        val_loss,
                        world_model,
                        optimizer,
                        wandb_config,
                    )

            if scheduler is not None:
                scheduler.step()

        if best_model_state is not None:
            checkpoint_path = Path(run.dir) / 'best_model.pt'
            save_checkpoint(best_model_state, checkpoint_path)

        if len(test_loader) > 0:
            with torch.no_grad():
                test_metrics, step = run_epoch(
                    tokenizer,
                    world_model,
                    test_loader,
                    'test',
                    device,
                    cfg,
                    step=step,
                )
            run.log(prepare_metrics_for_log(test_metrics), step=step)


if __name__ == '__main__':
    main()
