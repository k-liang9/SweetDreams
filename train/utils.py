import copy
import os
from pathlib import Path
import random

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import wandb


def init_distributed():
    if 'LOCAL_RANK' not in os.environ:
        return 0, 1
    local_rank = int(os.environ['LOCAL_RANK'])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    return local_rank, dist.get_world_size()


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed():
    return dist.is_initialized()


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process():
    return get_rank() == 0


def all_reduce_mean(tensor):
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


def check_stop_file(path, device):
    """Rank 0 checks the sentinel file; decision is broadcast so all ranks agree."""
    path = Path(path)
    if not dist.is_initialized():
        return path.exists()
    flag = torch.tensor(int(path.exists()) if dist.get_rank() == 0 else 0, device=device)
    dist.broadcast(flag, src=0)
    return bool(flag.item())


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
    keys = {}  # ordered union via dict insertion order
    for metrics in metrics_list:
        for key in metrics.keys():
            keys[key] = None
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
