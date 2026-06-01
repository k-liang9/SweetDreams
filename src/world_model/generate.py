# Rollout / inference for the world model

# TODO:
# 0. RoPE
# 1. kv cache
# 2/3. flash decoding & custom kernel
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hydra
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import tqdm
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from data import AtariEpisodeDataset
from train.utils import get_device, load_tokenizer, load_world_model, move_to_device
from world_model.kv_cache import KVCache


class WorldModelEnv:
    """Gym-style wrapper driving a cached autoregressive rollout."""

    def __init__(self, world_model, tokenizer, device, cfg):
        self.tokenizer = tokenizer.to(device).bfloat16()
        self.world_model = world_model.to(device).bfloat16()
        self.device = device
        self.temp = cfg.generate.temperature
        self.top_k = cfg.generate.top_k
        self.greedy = cfg.generate.greedy
        self.decode_chunk = cfg.generate.decode_chunk
        self.N = cfg.model.tokens_per_frame
        self.B = cfg.generate.batch_size

        self.cache = KVCache(cfg)
        self.cache.allocate(dtype=torch.bfloat16, device=device)

    @torch.no_grad()
    def reset(self, prompt_frames, prompt_actions):
        """
        prompt_frames:  (B, seq_len, C, H, W)
        prompt_actions: (B, seq_len - 1)

        Encodes the prompt and prefills the KV cache. After this call the cache
        holds K/V for every prompt token at every layer.
        """
        prompt_tokens = self.tokenizer.encode(prompt_frames.bfloat16()).flatten(2).contiguous()  # (B, seq_len, N)
        self.cache.reset()
        self.world_model(prompt_tokens, prompt_actions, start=0, cache=self.cache)

    @torch.no_grad()
    def step(self, action):
        """
        action: (B,) — the action conditioning the next frame.
        returns: (B, N) — frame tokens for the newly generated frame.
        """
        new_frame = torch.empty(self.B, self.N, dtype=torch.long, device=self.device)

        # injected action; logits at its position predict frame token 0
        logits = self.world_model.forward_token(action, self.cache.length, self.cache)
        new_frame[:, 0] = self._sample(logits)

        # frame token n's logits predict token n+1
        for i in range(1, self.N):
            logits = self.world_model.forward_token(new_frame[:, i - 1], self.cache.length, self.cache)
            new_frame[:, i] = self._sample(logits)

        # append the last frame token's K/V so the next step's injected action
        # attends to full preceding context; logits are unused
        _ = self.world_model.forward_token(new_frame[:, self.N - 1], self.cache.length, self.cache)

        return new_frame

    @torch.no_grad()
    def _sample(self, logits):
        """logits: (B, V) -> (B,) token ids."""
        if self.greedy:
            return logits.argmax(dim=-1)
        if self.temp != 1.0:
            logits = logits / self.temp
        if self.top_k is not None and self.top_k > 0:
            v, _ = torch.topk(logits, k=self.top_k, dim=-1)
            threshold = v[..., -1:]
            logits = torch.where(logits < threshold, torch.full_like(logits, float('-inf')), logits)
        probs = torch.softmax(logits.float(), dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def decode(self, frame_tokens):
        """(B, T, N) frame tokens -> (B, T, C, H, W) pixels. Chunked along T so the VQ-VAE decoder processes B*decode_chunk frames at a time."""
        B, T, N = frame_tokens.shape
        H = W = int(round(N ** 0.5))
        if H * W != N:
            raise ValueError(f'tokens_per_frame={N} is not a perfect square')
        indices = frame_tokens.reshape(B, T, H, W)
        pieces = [
            self.tokenizer.decode_from_indices(indices[:, t:t + self.decode_chunk])
            for t in range(0, T, self.decode_chunk)
        ]
        return torch.cat(pieces, dim=1)


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
    T = rollout_actions.shape[1]

    # one cycle of 1 wait + 1 warmup + 3 active = 20 steps
    sched = schedule(wait=1, warmup=1, active=3, repeat=1)

    def on_trace_ready(p):
        print(p.key_averages(group_by_input_shape=True).table(
            sort_by='self_cuda_time_total', row_limit=25,
        ))
        p.export_chrome_trace('rollout_trace.json')

    frames = []
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        on_trace_ready=on_trace_ready,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_stack=False,
    ) as prof:
        for t in tqdm.trange(T, desc='rollout'):
            with record_function('env.step'):
                frames.append(env.step(rollout_actions[:, t]))
            prof.step()

    return torch.stack(frames, dim=1)  # (B, rollout_steps, N)


def log_rollout(imagined, future_frames, run, fps):
    """imagined, future_frames: (B, T, C, H, W) in [0,1]. Logs each sample side-by-side as a wandb video."""
    side_by_side = torch.cat([imagined, future_frames], dim=-1)  # (B, T, C, H, 2W)
    videos = (side_by_side.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    run.log({'rollout': [wandb.Video(v, fps=fps, format='gif') for v in videos]})


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
        tags=[str(cfg.generate.tag)],
        config=OmegaConf.to_container(cfg, resolve=True),
        mode='disabled',
    ) as run:
        log_rollout(imagined, future_frames, run, fps=cfg.generate.fps)


if __name__ == '__main__':
    main()