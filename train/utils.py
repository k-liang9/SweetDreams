import copy
from pathlib import Path
import random

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import wandb


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device):
    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS was requested, but torch.backends.mps is not available')
    return torch.device(device)


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, (tuple, list)):
        return type(value)(move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def prepare_metrics_for_log(metrics):
    log = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = value.detach()
            if value.numel() == 1:
                value = value.item()
        log[key] = value
    return log


def aggregate_metrics(metrics_list):
    if not metrics_list:
        return {}

    aggregated = {}
    keys = metrics_list[0].keys()
    for key in keys:
        values = [metrics[key] for metrics in metrics_list if key in metrics]
        if not values:
            continue

        first = values[0]
        if isinstance(first, (int, float)):
            aggregated[key] = sum(values) / len(values)
        elif torch.is_tensor(first) and first.numel() == 1:
            aggregated[key] = torch.stack([value.detach() for value in values]).mean()
        else:
            aggregated[key] = first

    return aggregated


def checkpoint_state(epoch, val_loss, model, optimizer, cfg):
    return {
        'epoch': epoch,
        'val_loss': val_loss,
        'model_state_dict': copy.deepcopy(model.state_dict()),
        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
        'cfg': cfg,
    }


def save_checkpoint(state, path, save_to_wandb=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if save_to_wandb:
        wandb.save(str(path))


def build_scheduler(optimizer, scheduler_cfg):
    if scheduler_cfg is None or getattr(scheduler_cfg, 'type', 'none') == 'none':
        return None

    if scheduler_cfg.type == 'step':
        return StepLR(
            optimizer,
            step_size=scheduler_cfg.step_size,
            gamma=scheduler_cfg.gamma,
        )

    if scheduler_cfg.type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.t_max,
            eta_min=scheduler_cfg.get('eta_min', 0.0),
        )

    raise ValueError(f'Unsupported scheduler type: {scheduler_cfg.type}')
