from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tokenizer import VQVAE  # noqa: E402
from train.utils import get_device  # noqa: E402


@dataclass
class BallCandidate:
    episode: str
    frame_index: int
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    area: int
    score: float


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Counterfactual VQ-VAE diagnostic: detect small moving bright objects, '
            'erase them, and measure raw/encoder/token/reconstruction sensitivity.'
        )
    )
    parser.add_argument('--h5-path', default='data/breakout.h5')
    parser.add_argument('--checkpoint-path', default=None)
    parser.add_argument('--output-dir', default='diagnostics/vqvae_ball')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-examples', type=int, default=8)
    parser.add_argument('--max-scan-frames', type=int, default=5000)
    parser.add_argument('--motion-threshold', type=int, default=18)
    parser.add_argument('--intensity-threshold', type=int, default=80)
    parser.add_argument('--min-area', type=int, default=1)
    parser.add_argument('--max-area', type=int, default=24)
    parser.add_argument('--max-width', type=int, default=8)
    parser.add_argument('--max-height', type=int, default=8)
    parser.add_argument('--y-min-frac', type=float, default=0.15)
    parser.add_argument('--y-max-frac', type=float, default=0.82)
    parser.add_argument('--erase-radius', type=int, default=2)
    parser.add_argument('--tile-scale', type=int, default=4)
    return parser.parse_args()


def clamp(value, low, high):
    return max(low, min(high, value))


def detect_ball(frames: np.ndarray, frame_index: int, args) -> BallCandidate | None:
    frame = frames[frame_index].astype(np.int16)
    prev_frame = frames[frame_index - 1].astype(np.int16)
    next_frame = frames[frame_index + 1].astype(np.int16)

    motion = np.maximum(np.abs(frame - prev_frame), np.abs(next_frame - frame))
    mask = (motion >= args.motion_threshold) & (frame >= args.intensity_threshold)

    height, width = frame.shape
    y_min = int(args.y_min_frac * height)
    y_max = int(args.y_max_frac * height)
    mask[:y_min] = False
    mask[y_max:] = False

    labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)

    best = None
    for label in range(1, labels):
        x, y, w, h, area = stats[label]
        if area < args.min_area or area > args.max_area:
            continue
        if w > args.max_width or h > args.max_height:
            continue

        x1 = x + w
        y1 = y + h
        component_motion = motion[y:y1, x:x1][mask[y:y1, x:x1]]
        component_intensity = frame[y:y1, x:x1][mask[y:y1, x:x1]]
        score = float(component_motion.mean() + 0.1 * component_intensity.mean())
        candidate = BallCandidate(
            episode='',
            frame_index=frame_index,
            bbox=(int(x), int(y), int(x1), int(y1)),
            center=(float(centroids[label][0]), float(centroids[label][1])),
            area=int(area),
            score=score,
        )
        if best is None or candidate.score > best.score:
            best = candidate

    return best


def collect_candidates(args) -> tuple[list[tuple[BallCandidate, np.ndarray]], int]:
    samples = []
    scanned = 0

    with h5py.File(args.h5_path, 'r') as file:
        for episode in sorted(file.keys()):
            frames = file[episode]['frames']
            if len(frames) < 3:
                continue

            for frame_index in range(1, len(frames) - 1):
                if scanned >= args.max_scan_frames:
                    return samples, scanned
                scanned += 1

                local_frames = frames[frame_index - 1:frame_index + 2]
                candidate = detect_ball(local_frames, 1, args)
                if candidate is None:
                    continue

                candidate.episode = episode
                candidate.frame_index = frame_index
                samples.append((candidate, np.asarray(frames[frame_index], dtype=np.uint8)))
                if len(samples) >= args.num_examples:
                    return samples, scanned

    return samples, scanned


def erase_candidate(frame: np.ndarray, candidate: BallCandidate, radius: int) -> np.ndarray:
    x0, y0, x1, y1 = candidate.bbox
    height, width = frame.shape

    inner_x0 = clamp(x0 - radius, 0, width)
    inner_y0 = clamp(y0 - radius, 0, height)
    inner_x1 = clamp(x1 + radius, 0, width)
    inner_y1 = clamp(y1 + radius, 0, height)

    outer_radius = max(radius * 2 + 1, radius + 2)
    outer_x0 = clamp(x0 - outer_radius, 0, width)
    outer_y0 = clamp(y0 - outer_radius, 0, height)
    outer_x1 = clamp(x1 + outer_radius, 0, width)
    outer_y1 = clamp(y1 + outer_radius, 0, height)

    out = frame.copy()
    outer = out[outer_y0:outer_y1, outer_x0:outer_x1]
    ring_mask = np.ones(outer.shape, dtype=bool)
    ring_mask[
        inner_y0 - outer_y0:inner_y1 - outer_y0,
        inner_x0 - outer_x0:inner_x1 - outer_x0,
    ] = False
    background = np.median(outer[ring_mask]) if np.any(ring_mask) else np.median(out)
    out[inner_y0:inner_y1, inner_x0:inner_x1] = np.uint8(background)
    return out


def load_vqvae(checkpoint_path: str | None, device):
    if checkpoint_path is None:
        return None

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {path}')

    checkpoint = torch.load(path, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'cfg' not in checkpoint:
        raise ValueError(
            f'Checkpoint {path} does not contain a cfg entry. '
            'Expected a training checkpoint saved by train/utils.py.'
        )
    if 'model_state_dict' not in checkpoint:
        raise ValueError(f'Checkpoint {path} does not contain model_state_dict.')

    cfg = OmegaConf.create(checkpoint['cfg'])
    state_dict = checkpoint['model_state_dict']

    model = VQVAE(cfg)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def frame_to_tensor(frame: np.ndarray, device):
    return torch.from_numpy(frame).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().squeeze().clamp(0, 1).numpy()
    return (array * 255.0).round().astype(np.uint8)


def analyze_with_model(model, frame: np.ndarray, erased: np.ndarray, candidate: BallCandidate, device):
    x = frame_to_tensor(frame, device)
    x_erased = frame_to_tensor(erased, device)

    with torch.no_grad():
        z = torch.nn.functional.normalize(model.encoder(x), p=2, dim=1)
        z_erased = torch.nn.functional.normalize(model.encoder(x_erased), p=2, dim=1)
        activation_delta = (z - z_erased).pow(2).sum(dim=1).sqrt()

        tokens = model.encode(x)
        tokens_erased = model.encode(x_erased)
        token_changes = tokens != tokens_erased

        pred = model.decode_from_indices(tokens)
        pred_erased = model.decode_from_indices(tokens_erased)

    _, latent_h, latent_w = tokens.shape
    frame_h, frame_w = frame.shape
    cx, cy = candidate.center
    latent_x = clamp(int(cx * latent_w / frame_w), 0, latent_w - 1)
    latent_y = clamp(int(cy * latent_h / frame_h), 0, latent_h - 1)

    local_y0 = clamp(latent_y - 1, 0, latent_h)
    local_y1 = clamp(latent_y + 2, 0, latent_h)
    local_x0 = clamp(latent_x - 1, 0, latent_w)
    local_x1 = clamp(latent_x + 2, 0, latent_w)
    local_token_changes = token_changes[0, local_y0:local_y1, local_x0:local_x1].sum().item()

    x0, y0, x1, y1 = candidate.bbox
    target_crop = x[:, :, y0:y1, x0:x1]
    pred_crop = pred[:, :, y0:y1, x0:x1]

    metrics = {
        'encoder_delta_mean': float(activation_delta.mean().item()),
        'encoder_delta_max': float(activation_delta.max().item()),
        'encoder_delta_at_ball_cell': float(activation_delta[0, latent_y, latent_x].item()),
        'latent_ball_cell': [int(latent_y), int(latent_x)],
        'token_changes_total': int(token_changes.sum().item()),
        'token_changes_near_ball': int(local_token_changes),
        'target_ball_patch_max': float(target_crop.max().item()),
        'recon_ball_patch_max': float(pred_crop.max().item()),
        'recon_ball_patch_l1': float((pred_crop - target_crop).abs().mean().item()),
    }

    images = {
        'recon': tensor_to_uint8_image(pred),
        'recon_erased': tensor_to_uint8_image(pred_erased),
        'recon_delta': tensor_to_uint8_image((pred - pred_erased).abs() / (pred - pred_erased).abs().max().clamp_min(1e-8)),
    }
    return metrics, images


def draw_tile(image: np.ndarray, bbox=None, scale=4):
    pil = Image.fromarray(image).convert('RGB')
    pil = pil.resize((pil.width * scale, pil.height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(pil)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        box = [x0 * scale, y0 * scale, x1 * scale - 1, y1 * scale - 1]
        draw.rectangle(box, outline=(255, 0, 0), width=max(1, scale // 2))
    return pil


def save_grid(rows, output_path: Path, scale: int):
    labels = ['frame', 'erased']
    if rows and 'recon' in rows[0]['images']:
        labels.extend(['recon', 'recon erased', 'recon delta'])

    tile_w = rows[0]['images']['frame'].shape[1] * scale
    tile_h = rows[0]['images']['frame'].shape[0] * scale
    label_h = 18
    margin = 6
    grid_w = len(labels) * tile_w
    grid_h = label_h + len(rows) * (tile_h + margin)
    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    for col, label in enumerate(labels):
        draw.text((col * tile_w + 4, 3), label, fill=(0, 0, 0))

    for row_idx, row in enumerate(rows):
        y = label_h + row_idx * (tile_h + margin)
        for col, label in enumerate(labels):
            key = label.replace(' ', '_')
            if key == 'frame':
                tile = draw_tile(row['images']['frame'], row['bbox'], scale)
            elif key == 'erased':
                tile = draw_tile(row['images']['erased'], row['bbox'], scale)
            else:
                tile = draw_tile(row['images'][key], row['bbox'], scale)
            grid.paste(tile, (col * tile_w, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, scanned = collect_candidates(args)
    if not samples:
        raise RuntimeError(
            f'No ball candidates found after scanning {scanned} frames. '
            'Try lowering --motion-threshold or --intensity-threshold.'
        )

    device = get_device(args.device)
    model = load_vqvae(args.checkpoint_path, device)

    report = {
        'h5_path': args.h5_path,
        'checkpoint_path': args.checkpoint_path,
        'scanned_frames': scanned,
        'num_examples': len(samples),
        'samples': [],
    }
    rows = []

    for candidate, frame in samples:
        erased = erase_candidate(frame, candidate, args.erase_radius)
        x0, y0, x1, y1 = candidate.bbox
        ball_patch = frame[y0:y1, x0:x1]
        sample_report = {
            **asdict(candidate),
            'raw_ball_patch_max': int(ball_patch.max()),
            'raw_ball_patch_mean': float(ball_patch.mean()),
        }

        row = {
            'bbox': candidate.bbox,
            'images': {
                'frame': frame,
                'erased': erased,
            },
        }

        if model is not None:
            model_metrics, model_images = analyze_with_model(model, frame, erased, candidate, device)
            sample_report.update(model_metrics)
            row['images'].update(model_images)

        report['samples'].append(sample_report)
        rows.append(row)

    json_path = output_dir / 'report.json'
    with json_path.open('w') as file:
        json.dump(report, file, indent=2)

    grid_path = output_dir / 'examples.png'
    save_grid(rows, grid_path, args.tile_scale)

    print(f'Scanned frames: {scanned}')
    print(f'Ball candidates: {len(samples)}')
    print(f'Saved report: {json_path}')
    print(f'Saved examples: {grid_path}')

    if model is None:
        print('No checkpoint passed; skipped encoder/token/reconstruction diagnostics.')
        print('Pass --checkpoint-path path/to/checkpoint.pt to test the trained VQ-VAE.')
        return

    token_changes = [sample['token_changes_total'] for sample in report['samples']]
    encoder_max = [sample['encoder_delta_max'] for sample in report['samples']]
    recon_patch_max = [sample['recon_ball_patch_max'] for sample in report['samples']]
    print(f'Mean token changes: {np.mean(token_changes):.3f}')
    print(f'Mean encoder delta max: {np.mean(encoder_max):.6f}')
    print(f'Mean recon ball patch max: {np.mean(recon_patch_max):.6f}')


if __name__ == '__main__':
    main()
