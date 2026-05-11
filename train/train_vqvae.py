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
from torch.utils.data import DataLoader, random_split
import tqdm
import wandb

from common.train_utils import (
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


def run_epoch(model, loader, split, device, optimizer=None, log_every=None, run=None, step=0, max_batches=None):
    is_train = optimizer is not None
    model.train(is_train)
    metrics_list = []
    desc = f'{split} batches'

    total = min(len(loader), max_batches) if max_batches is not None else len(loader)
    for batch_idx, batch in enumerate(tqdm.tqdm(loader, total=total, desc=desc)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = move_to_device(batch, device)
        x, target = model.featurize(batch)

        with torch.set_grad_enabled(is_train):
            out = model(x, target)
            if is_train:
                optimizer.zero_grad()
                out['loss'].backward()
                optimizer.step()
                step += 1

        metrics = model.compute_metrics(split, out, x, target)
        if is_train:
            if run is not None and (log_every is None or step % log_every == 0):
                run.log(prepare_metrics_for_log(metrics), step=step)
        else:
            metrics_list.append(metrics)

    if not is_train:
        return aggregate_metrics(metrics_list), step
    return {}, step


@hydra.main(version_base=None, config_path='../configs', config_name='vqvae')
def main(cfg: DictConfig):
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)

    train_loader, val_loader, test_loader = make_loaders(cfg)
    model = VQVAE(cfg).to(device)
    if cfg.optimizer.type != 'adam':
        raise ValueError(f'Unsupported optimizer type: {cfg.optimizer.type}')
    optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    scheduler = build_scheduler(optimizer, cfg.scheduler)

    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    with wandb.init(
        project=cfg.exp.project, 
        name=cfg.exp.run_name, 
        group=cfg.exp.group, 
        entity=cfg.exp.entity, 
        config=wandb_config
    ) as run:
        best_model_state = None
        step = 0

        for epoch in range(cfg.train.epochs):
            _, step = run_epoch(
                model,
                train_loader,
                'train',
                device,
                optimizer=optimizer,
                log_every=cfg.train.log_every,
                run=run,
                step=step,
                max_batches=cfg.train.limit_train_batches,
            )

            with torch.no_grad():
                val_metrics, step = run_epoch(
                    model,
                    val_loader,
                    'val',
                    device,
                    step=step,
                    max_batches=cfg.train.limit_val_batches,
                )
            run.log(prepare_metrics_for_log(val_metrics), step=step)

            val_loss = val_metrics.get('val/loss')
            if torch.is_tensor(val_loss):
                val_loss = val_loss.item()
            if val_loss is not None and (best_model_state is None or val_loss < best_model_state['val_loss']):
                best_model_state = checkpoint_state(epoch, val_loss, model, optimizer, wandb_config)

            if scheduler is not None:
                scheduler.step()

        if best_model_state is not None:
            checkpoint_path = Path(run.dir) / 'best_model.pt'
            save_checkpoint(best_model_state, checkpoint_path)

        with torch.no_grad():
            test_metrics, step = run_epoch(
                model,
                test_loader,
                'test',
                device,
                step=step,
                max_batches=cfg.train.limit_test_batches,
            )
        run.log(prepare_metrics_for_log(test_metrics), step=step)


if __name__ == '__main__':
    main()
