"""Reconstruct every frame in the contact-eval set with a trained VQ-VAE and save a visual grid."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument('--h5-path', default='eval/breakout_contacts.h5',
                        help='Eval h5 produced by eval/build_contact_eval.py (flat list of contact frames).')
    parser.add_argument('--checkpoint-path', default='weights/vqvae.pt')
    parser.add_argument('--output-dir', default='diagnostics/vqvae_reconstructions')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--tile-scale', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
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


def load_eval_set(h5_path):
    """Read the flat contact-eval h5. Returns (frames, meta) where meta is a list of
    (source_episode, source_frame_idx, contact_type) per frame."""
    with h5py.File(h5_path, 'r') as file:
        frames = file['frames'][:]
        eps = [s.decode() for s in file['source_episode'][:]]
        fis = file['source_frame_idx'][:].tolist()
        types = [s.decode() for s in file['contact_type'][:]]
    meta = list(zip(eps, fis, types))
    return frames, meta


def reconstruct(model, frames_uint8, device, batch_size):
    """Encode/decode in batches so we can handle the whole eval set without OOM."""
    n, h, w, _ = frames_uint8.shape
    recon_full = np.empty_like(frames_uint8)
    l1 = np.empty(n, dtype=np.float32)
    mse = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        chunk = frames_uint8[start:start + batch_size]
        x = torch.from_numpy(chunk).float().div(255.0).permute(0, 3, 1, 2).contiguous().to(device)
        with torch.no_grad():
            indices = model.encode(x)
            recon = model.decode_from_indices(indices)
        recon = recon.clamp(0, 1)
        l1[start:start + len(chunk)] = (recon - x).abs().mean(dim=(1, 2, 3)).cpu().numpy()
        mse[start:start + len(chunk)] = ((recon - x) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        recon_full[start:start + len(chunk)] = (
            (recon.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        )
    psnr = -10.0 * np.log10(np.clip(mse, 1e-12, None))
    return recon_full, l1, mse, psnr


def upscale(image_uint8, scale):
    pil = Image.fromarray(image_uint8).convert('RGB')
    return pil.resize((pil.width * scale, pil.height * scale), Image.NEAREST)


def save_grid(frames, recons, meta, l1, psnr, output_path, scale):
    n, h, w, _ = frames.shape
    diffs = np.abs(frames.astype(np.int16) - recons.astype(np.int16)).astype(np.float32)
    per_row_max = diffs.reshape(n, -1).max(axis=1).clip(min=1.0)
    diffs_vis = (diffs / per_row_max[:, None, None, None] * 255.0).astype(np.uint8)

    tile_w = w * scale
    tile_h = h * scale
    label_w = 220
    header_h = 22
    grid = Image.new('RGB', (label_w + tile_w * 3, header_h + tile_h * n), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    for col, name in enumerate(('original', 'reconstruction', '|diff| (norm)')):
        draw.text((label_w + col * tile_w + 4, 4), name, fill=(0, 0, 0))

    for i in range(n):
        ep_key, fi, ctype = meta[i]
        y = header_h + i * tile_h
        text = f'{ctype}\n{ep_key}\nframe {fi}\nL1={l1[i]:.4f}\nPSNR={psnr[i]:.2f} dB'
        draw.multiline_text((4, y + 4), text, fill=(0, 0, 0), spacing=2)
        grid.paste(upscale(frames[i], scale), (label_w + 0 * tile_w, y))
        grid.paste(upscale(recons[i], scale), (label_w + 1 * tile_w, y))
        grid.paste(upscale(diffs_vis[i], scale), (label_w + 2 * tile_w, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def main():
    args = parse_args()

    frames, meta = load_eval_set(args.h5_path)
    if len(frames) == 0:
        raise RuntimeError(f'No frames in {args.h5_path}; rebuild via eval/build_contact_eval.py.')

    device = get_device(args.device)
    model, _ = load_vqvae(args.checkpoint_path, device)

    recons, l1, mse, psnr = reconstruct(model, frames, device, args.batch_size)

    output_dir = Path(args.output_dir)
    grid_path = output_dir / 'reconstructions.png'
    save_grid(frames, recons, meta, l1, psnr, grid_path, args.tile_scale)

    by_type: dict[str, list[int]] = {}
    for i, (_, _, ctype) in enumerate(meta):
        by_type.setdefault(ctype, []).append(i)
    per_type = {
        ctype: {
            'n': len(idxs),
            'mean_l1': float(l1[idxs].mean()),
            'mean_mse': float(mse[idxs].mean()),
            'mean_psnr_db': float(psnr[idxs].mean()),
        }
        for ctype, idxs in by_type.items()
    }

    report = {
        'h5_path': args.h5_path,
        'checkpoint_path': args.checkpoint_path,
        'num_samples': len(frames),
        'mean_l1': float(l1.mean()),
        'mean_mse': float(mse.mean()),
        'mean_psnr_db': float(psnr.mean()),
        'per_contact_type': per_type,
        'samples': [
            {
                'episode': ep_key, 'frame': int(fi), 'contact_type': ctype,
                'l1': float(l1[i]), 'mse': float(mse[i]), 'psnr_db': float(psnr[i]),
            }
            for i, (ep_key, fi, ctype) in enumerate(meta)
        ],
    }
    report_path = output_dir / 'report.json'
    with report_path.open('w') as f:
        json.dump(report, f, indent=2)

    print(f'Samples: {len(frames)}  mean L1={l1.mean():.4f}  mean PSNR={psnr.mean():.2f} dB')
    for ctype, stats in per_type.items():
        print(f"  {ctype:>10}: n={stats['n']:>3}  L1={stats['mean_l1']:.4f}  PSNR={stats['mean_psnr_db']:.2f} dB")
    print(f'Saved grid: {grid_path}')
    print(f'Saved report: {report_path}')


if __name__ == '__main__':
    main()
