# Rollout / inference for the world model. No KV cache yet (transformer.py
# TODO), so each generated token re-runs the full forward pass.
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hydra
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from data import AtariEpisodeDataset
from train.utils import get_device, load_tokenizer, load_world_model, move_to_device


class WorldModelEnv:
    """Gym-style wrapper holding the sliding (frame_tokens, actions) window for autoregressive rollouts."""

    def __init__(self, world_model, tokenizer, device, cfg):
        self.tokenizer = tokenizer.to(device)
        self.world_model = world_model.to(device)
        self.device = device
        self.temp = cfg.generate.temperature
        self.top_k = cfg.generate.top_k
        self.greedy = cfg.generate.greedy
        N = cfg.model.tokens_per_frame
        B = cfg.generate.batch_size
        max_T = cfg.data.seq_len
        self.frame_tokens = torch.zeros(B, max_T + 1, N, dtype=torch.long, device=device)
        self.actions = torch.zeros(B, max_T, dtype=torch.long, device=device)

    @torch.no_grad()
    def reset(self, prompt_frames, prompt_actions):
        """
        prompt_frames:  (B, seq_len, C, H, W)
        prompt_actions: (B, seq_len - 1)

        Lays out buffers so the last frame slot and last action slot are empty,
        ready for step() to fill on the next call.
        """
        prompt_tokens = self.tokenizer.encode(prompt_frames).flatten(2).contiguous()  # (B, seq_len, N)

        self.frame_tokens.zero_()
        self.actions.zero_()

        seq_len = self.actions.shape[1]
        self.frame_tokens[:, :seq_len] = prompt_tokens
        self.actions[:, :seq_len - 1] = prompt_actions

    @torch.no_grad()
    def step(self, action):
        """
        action: (B,) — the action conditioning the next frame.
        returns: (B, N) — frame tokens for the newly generated frame.
        """
        self.actions[:, -1] = action

        N = self.frame_tokens.shape[2]
        for n in range(N):
            self.frame_tokens[:, -1, n] = self._sample_token(n)

        generated = self.frame_tokens[:, -1].clone()

        self.frame_tokens = torch.roll(self.frame_tokens, shifts=-1, dims=1)
        self.frame_tokens[:, -1] = 0
        self.actions = torch.roll(self.actions, shifts=-1, dims=1)
        self.actions[:, -1] = 0

        return generated

    @torch.no_grad()
    def _sample_token(self, n):
        """Logit at position seq_len*(N+1) + n - 1 predicts token n of the last frame."""
        out = self.world_model(self.frame_tokens, self.actions)
        logits = out['frame_logits']  # (B, S, V)
        seq_len = self.actions.shape[1]
        N = self.frame_tokens.shape[2]
        pos = seq_len * (N + 1) + n - 1
        logits = logits[:, pos]  # (B, V)

        if self.greedy:
            return logits.argmax(dim=-1)
        if self.temp != 1.0:
            logits = logits / self.temp
        if self.top_k is not None and self.top_k > 0:
            v, _ = torch.topk(logits, k=self.top_k, dim=-1)
            threshold = v[..., -1:]
            logits = torch.where(logits < threshold, torch.full_like(logits, float('-inf')), logits)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def decode(self, frame_tokens):
        """(B, T, N) frame tokens -> (B, T, C, H, W) pixels."""
        B, T, N = frame_tokens.shape
        H = W = int(round(N ** 0.5))
        if H * W != N:
            raise ValueError(f'tokens_per_frame={N} is not a perfect square')
        return self.tokenizer.decode_from_indices(frame_tokens.reshape(B, T, H, W))


def load(cfg, device):
    """Tokenizer, world model, and a dataset whose window covers prompt + rollout."""
    tokenizer = load_tokenizer(cfg, device)
    world_model = load_world_model(cfg, device)
    window = cfg.data.seq_len + cfg.generate.rollout_steps
    dataset = AtariEpisodeDataset(
        h5_path=to_absolute_path(cfg.data.h5_path),
        seq_len=window,
        return_tokens=False,
    )
    return tokenizer, world_model, dataset


def split_batch(frames, actions, seq_len, rollout_steps):
    """
    frames:  (B, seq_len + rollout_steps, C, H, W)
    actions: (B, seq_len + rollout_steps)

    Returns prompt frames/actions, the actions driving generation, and the
    ground-truth future frames for side-by-side comparison.
    """
    prompt_frames = frames[:, :seq_len]
    prompt_actions = actions[:, :seq_len - 1]
    rollout_actions = actions[:, seq_len - 1 : seq_len - 1 + rollout_steps]
    future_frames = frames[:, seq_len : seq_len + rollout_steps]
    return prompt_frames, prompt_actions, rollout_actions, future_frames


@torch.no_grad()
def rollout(env, prompt_frames, prompt_actions, rollout_actions):
    env.reset(prompt_frames, prompt_actions)
    frames = [env.step(rollout_actions[:, t]) for t in range(rollout_actions.shape[1])]
    return torch.stack(frames, dim=1)  # (B, rollout_steps, N)


def log_rollout(imagined, future_frames, run, fps):
    """imagined, future_frames: (B, T, C, H, W) in [0,1]. Logs each sample side-by-side as a wandb video."""
    side_by_side = torch.cat([imagined, future_frames], dim=-1)  # (B, T, C, H, 2W)
    videos = (side_by_side.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    run.log({'rollout': [wandb.Video(v, fps=fps) for v in videos]})


@hydra.main(version_base=None, config_path='../../configs', config_name='world_model')
def main(cfg: DictConfig): 
    device = get_device(cfg.train.device)
    tokenizer, world_model, dataset = load(cfg, device)
    env = WorldModelEnv(world_model, tokenizer, device, cfg)

    loader = DataLoader(dataset, batch_size=cfg.generate.batch_size, shuffle=True)
    frames, actions, _ = move_to_device(next(iter(loader)), device)

    prompt_frames, prompt_actions, rollout_actions, future_frames = split_batch(
        frames, actions, cfg.data.seq_len, cfg.generate.rollout_steps,
    )

    generated = rollout(env, prompt_frames, prompt_actions, rollout_actions)
    imagined = env.decode(generated)

    with wandb.init(
        project=str(cfg.exp.project),
        name=f'{cfg.exp.run_name}-rollout',
        group=str(cfg.exp.group),
        entity=str(cfg.exp.entity),
        tags=[str(cfg.exp.tag), str(cfg.generate.tag)],
        config=OmegaConf.to_container(cfg, resolve=True),
    ) as run:
        log_rollout(imagined, future_frames, run, fps=cfg.generate.fps)


if __name__ == '__main__':
    main()