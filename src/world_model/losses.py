import torch
from torch.nn import functional as F


IGNORE_INDEX = -100


def build_frame_targets(frame_tokens, ignore_index=IGNORE_INDEX):
    """
    Build next-frame token targets for the interleaved sequence:
    z0_0 ... z0_15 a0 z1_0 ... z1_15 a1 ...

    frame_tokens: (B, T, N)

    returns:
        targets: (B, S), where ignored positions are ignore_index
        mask:    (B, S), True where frame-token loss is computed
    """
    if frame_tokens.dim() != 3:
        raise ValueError(f'frame_tokens must have shape (B, T, N), got {tuple(frame_tokens.shape)}')

    B, T, N = frame_tokens.shape
    S = T * N + (T - 1)
    targets = frame_tokens.new_full((B, S), ignore_index)
    mask = torch.zeros((B, S), dtype=torch.bool, device=frame_tokens.device)

    # Predict frame t from action t - 1, then autoregressively within frame t.
    for t in range(1, T):
        prev_action_pos = (t - 1) * (N + 1) + N
        frame_start = t * (N + 1)

        targets[:, prev_action_pos] = frame_tokens[:, t, 0]
        mask[:, prev_action_pos] = True

        if N > 1:
            positions = slice(frame_start, frame_start + N - 1)
            targets[:, positions] = frame_tokens[:, t, 1:]
            mask[:, positions] = True

    return targets, mask


def frame_prediction_loss(frame_logits, frame_tokens, ignore_index=IGNORE_INDEX):
    """
    Cross entropy for predicting future VQ frame tokens.

    frame_logits: (B, S, V)
    frame_tokens: (B, T, N)
    """
    targets, mask = build_frame_targets(frame_tokens, ignore_index=ignore_index)
    if frame_logits.shape[:2] != targets.shape:
        raise ValueError(
            f'frame_logits must have shape (B, S, V) matching targets (B, S); '
            f'got logits {tuple(frame_logits.shape)} and targets {tuple(targets.shape)}'
        )
    if not mask.any():
        raise ValueError('No frame-token targets were selected; use seq_len >= 2')

    loss = F.cross_entropy(
        frame_logits.reshape(-1, frame_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )

    return {
        'loss': loss,
        'frame_loss': loss,
        'frame_targets': targets,
        'frame_mask': mask,
    }


def world_model_loss(out, frame_tokens, ignore_index=IGNORE_INDEX):
    frame_logits = out['frame_logits'] if isinstance(out, dict) else out
    return frame_prediction_loss(frame_logits, frame_tokens, ignore_index=ignore_index)
