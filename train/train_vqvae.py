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
from torchmetrics.image.fid import FrechetInceptionDistance
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


VAL_RECONSTRUCTION_EVERY_STEPS = 500


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


def fid_images(images):
    images = images.detach().clamp(0, 1)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    return images


def run_epoch(
    model,
    loader,
    split,
    device,
    optimizer=None,
    run=None,
    step=0,
    max_batches=None,
    include_reconstructions=False,
    fid=None,
):
    is_train = optimizer is not None
    model.train(is_train)
    metrics_list = []
    desc = f'{split} batches'

    total = min(len(loader), max_batches) if max_batches is not None else len(loader)
    for batch_idx, batch in enumerate(tqdm.tqdm(loader, total=total, desc=desc)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, target = model.featurize(batch)
        target_is_x = target is x
        x = move_to_device(x, device)
        target = x if target_is_x else move_to_device(target, device)

        with torch.set_grad_enabled(is_train):
            out = model(x, target)
            if is_train:
                optimizer.zero_grad()
                out['loss'].backward()
                optimizer.step()
                step += 1

        if is_train:
            metrics = model.compute_metrics(split, out, x, target)
            run.log(prepare_metrics_for_log(metrics), step=step)
        else:
            metrics = model.compute_metrics(
                split,
                out,
                x,
                target,
                include_reconstructions=include_reconstructions and not metrics_list,
                reconstruction_examples=1,
            )
            if fid is not None:
                fid.update(fid_images(target), real=True)
                fid.update(fid_images(out['pred']), real=False)
            metrics_list.append(metrics)

    if not is_train:
        metrics = aggregate_metrics(metrics_list)
        if fid is not None:
            metrics[f'{split}/fid'] = fid.compute()
        return metrics, step
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
        project=str(cfg.exp.project),
        name=str(cfg.exp.run_name),
        group=str(cfg.exp.group),
        entity=str(cfg.exp.entity),
        config=wandb_config
    ) as run:
        best_model_state = None
        step = 0
        next_val_reconstruction_step = VAL_RECONSTRUCTION_EVERY_STEPS

        for epoch in range(1, cfg.train.epochs + 1):
            _, step = run_epoch(
                model,
                train_loader,
                'train',
                device,
                optimizer=optimizer,
                run=run,
                step=step,
                max_batches=cfg.train.limit_train_batches,
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
                    max_batches=cfg.train.limit_val_batches,
                    include_reconstructions=include_val_reconstructions,
                )
            run.log(prepare_metrics_for_log(val_metrics), step=step)

            val_loss = val_metrics.get('val/recon_loss')
            if torch.is_tensor(val_loss):
                val_loss = val_loss.item()
            if val_loss is not None and (best_model_state is None or val_loss < best_model_state['val_loss']):
                best_model_state = checkpoint_state(epoch, val_loss, model, optimizer, wandb_config)

            if scheduler is not None:
                scheduler.step()

        if best_model_state is not None:
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
                max_batches=cfg.train.limit_test_batches,
                include_reconstructions=True,
                fid=fid,
            )
        run.log(prepare_metrics_for_log(test_metrics), step=step)


if __name__ == '__main__':
    main()
