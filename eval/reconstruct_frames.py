"""Reconstruct random dataset frames with a trained VQ-VAE and save a visual grid."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tokenizer import VQVAE  # noqa: E402
from train.utils import get_device  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h5-path', default='data/breakout.h5')
    parser.add_argument('--checkpoint-path', default='weights/vqvae.pt')
    parser.add_argument('--output-dir', default='diagnostics/vqvae_reconstructions')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-episodes', type=int, default=6,
                        help='Number of random episodes to sample from.')
    parser.add_argument('--frames-per-episode', type=int, default=4,
                        help='Number of random frames to sample per episode.')
    parser.add_argument('--tile-scale', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def load_vqvae(checkpoint_path, device):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {path}')

    checkpoint = torch.load(path, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'cfg' not in checkpoint or 'model_state_dict' not in checkpoint:
        raise ValueError(f'Checkpoint {path} missing cfg / model_state_dict.')

    cfg = OmegaConf.create(checkpoint['cfg'])
    model = VQVAE(cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval().to(device)
    return model, cfg


def pick_random_samples(h5_path, num_episodes, frames_per_episode, rng):
    with h5py.File(h5_path, 'r') as file:
        lengths = [(key, len(file[key]['frames'])) for key in file.keys()]
    chosen = rng.sample(lengths, k=min(num_episodes, len(lengths)))
    samples = []
    for ep_key, n_frames in chosen:
        indices = sorted(rng.sample(range(n_frames), k=min(frames_per_episode, n_frames)))
        for fi in indices:
            samples.append((ep_key, fi, n_frames))
    return samples


def load_frames(h5_path, samples):
    frames = np.empty((len(samples), 64, 64, 3), dtype=np.uint8)
    with h5py.File(h5_path, 'r') as file:
        for i, (ep_key, fi, _) in enumerate(samples):
            frames[i] = file[ep_key]['frames'][fi]
    return frames


def reconstruct(model, frames_uint8, device):
    x = torch.from_numpy(frames_uint8).float().div(255.0).permute(0, 3, 1, 2).contiguous().to(device)
    with torch.no_grad():
        indices = model.encode(x)
        recon = model.decode_from_indices(indices)
    recon = recon.clamp(0, 1)
    l1 = (recon - x).abs().mean(dim=(1, 2, 3)).cpu().numpy()
    mse = ((recon - x) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
    psnr = -10.0 * np.log10(np.clip(mse, 1e-12, None))
    recon_uint8 = (recon.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    return recon_uint8, l1, mse, psnr


def upscale(image_uint8, scale):
    pil = Image.fromarray(image_uint8).convert('RGB')
    return pil.resize((pil.width * scale, pil.height * scale), Image.Resampling.NEAREST)


def save_grid(frames, recons, samples, l1, psnr, output_path, scale):
    n = len(frames)
    diffs = np.abs(frames.astype(np.int16) - recons.astype(np.int16)).astype(np.float32)
    per_row_max = diffs.reshape(n, -1).max(axis=1).clip(min=1.0)
    diffs_vis = (diffs / per_row_max[:, None, None, None] * 255.0).astype(np.uint8)

    tile = 64 * scale
    label_w = 220
    header_h = 22
    grid = Image.new('RGB', (label_w + tile * 3, header_h + tile * n), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    for col, name in enumerate(('original', 'reconstruction', '|diff| (norm)')):
        draw.text((label_w + col * tile + 4, 4), name, fill=(0, 0, 0))

    for i in range(n):
        ep_key, fi, n_frames = samples[i]
        y = header_h + i * tile
        text = f'{ep_key}\nframe {fi}/{n_frames - 1}\nL1={l1[i]:.4f}\nPSNR={psnr[i]:.2f} dB'
        draw.multiline_text((4, y + 4), text, fill=(0, 0, 0), spacing=2)
        grid.paste(upscale(frames[i], scale), (label_w + 0 * tile, y))
        grid.paste(upscale(recons[i], scale), (label_w + 1 * tile, y))
        grid.paste(upscale(diffs_vis[i], scale), (label_w + 2 * tile, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    samples = pick_random_samples(args.h5_path, args.num_episodes, args.frames_per_episode, rng)
    if not samples:
        raise RuntimeError('No samples selected; check dataset / args.')

    device = get_device(args.device)
    model, _ = load_vqvae(args.checkpoint_path, device)

    frames = load_frames(args.h5_path, samples)
    recons, l1, mse, psnr = reconstruct(model, frames, device)

    output_dir = Path(args.output_dir)
    grid_path = output_dir / 'reconstructions.png'
    save_grid(frames, recons, samples, l1, psnr, grid_path, args.tile_scale)

    report = {
        'h5_path': args.h5_path,
        'checkpoint_path': args.checkpoint_path,
        'num_samples': len(samples),
        'mean_l1': float(l1.mean()),
        'mean_mse': float(mse.mean()),
        'mean_psnr_db': float(psnr.mean()),
        'samples': [
            {
                'episode': ep_key, 'frame': int(fi), 'episode_length': int(n_frames),
                'l1': float(l1[i]), 'mse': float(mse[i]), 'psnr_db': float(psnr[i]),
            }
            for i, (ep_key, fi, n_frames) in enumerate(samples)
        ],
    }
    report_path = output_dir / 'report.json'
    with report_path.open('w') as f:
        json.dump(report, f, indent=2)

    print(f'Samples: {len(samples)}  mean L1={l1.mean():.4f}  mean PSNR={psnr.mean():.2f} dB')
    print(f'Saved grid: {grid_path}')
    print(f'Saved report: {report_path}')


if __name__ == '__main__':
    main()
