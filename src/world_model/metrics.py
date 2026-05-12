import torch
from torch.nn import functional as F

from world_model.losses import IGNORE_INDEX, build_frame_targets


def _frame_logits(out):
    return out['frame_logits'] if isinstance(out, dict) else out


def _frame_targets_and_mask(out, frame_tokens, ignore_index):
    if isinstance(out, dict) and 'frame_targets' in out and 'frame_mask' in out:
        return out['frame_targets'], out['frame_mask']
    return build_frame_targets(frame_tokens, ignore_index=ignore_index)


def frame_token_accuracy(frame_logits, targets, mask):
    if not mask.any():
        return torch.zeros((), device=frame_logits.device)

    pred = frame_logits.argmax(dim=-1)
    return (pred[mask] == targets[mask]).float().mean()


def frame_token_perplexity(frame_loss):
    return torch.exp(frame_loss.detach())


def world_model_metrics(split, out, frame_tokens, ignore_index=IGNORE_INDEX):
    frame_logits = _frame_logits(out)
    targets, mask = _frame_targets_and_mask(out, frame_tokens, ignore_index)

    if isinstance(out, dict) and 'frame_loss' in out:
        frame_loss = out['frame_loss']
    else:
        frame_loss = F.cross_entropy(
            frame_logits.reshape(-1, frame_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=ignore_index,
        )

    if isinstance(out, dict) and 'loss' in out:
        loss = out['loss']
    else:
        loss = frame_loss

    return {
        f'{split}/loss': loss,
        f'{split}/frame_loss': frame_loss,
        f'{split}/frame_accuracy': frame_token_accuracy(frame_logits, targets, mask),
        f'{split}/frame_perplexity': frame_token_perplexity(frame_loss),
        f'{split}/frame_tokens': mask.float().sum(),
    }
