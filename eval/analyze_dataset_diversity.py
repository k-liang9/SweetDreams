from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min: float | None = None
    max: float | None = None

    def update(self, value: float) -> None:
        value = float(value)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.min = value if self.min is None else min(self.min, value)
        self.max = value if self.max is None else max(self.max, value)

    def to_dict(self) -> dict[str, float | int | None]:
        variance = self.m2 / (self.count - 1) if self.count > 1 else 0.0
        return {
            "count": self.count,
            "mean": self.mean,
            "std": float(np.sqrt(variance)),
            "min": self.min,
            "max": self.max,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stream through an Atari HDF5 dataset and summarize visual diversity, "
            "episode lengths/returns, and Breakout-specific coverage proxies."
        )
    )
    parser.add_argument("--h5-path", default="data/breakout.h5")
    parser.add_argument("--output-dir", default="diagnostics/dataset_diversity")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--sample-frames", type=int, default=4096)
    parser.add_argument("--pair-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--brick-y0-frac", type=float, default=0.26)
    parser.add_argument("--brick-y1-frac", type=float, default=0.46)
    parser.add_argument("--paddle-y0-frac", type=float, default=0.72)
    parser.add_argument("--paddle-y1-frac", type=float, default=0.96)
    parser.add_argument("--save-contact-sheet", action="store_true")
    return parser.parse_args()


def summarize_values(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "max": None,
        }

    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def entropy_from_counts(counts: np.ndarray | list[int]) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def frame_digest(frame: np.ndarray) -> bytes:
    return hashlib.blake2b(frame.tobytes(), digest_size=8).digest()


def coarse_digest(frame: np.ndarray) -> bytes:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    quantized = (small // 16).astype(np.uint8)
    return quantized.tobytes()


def brick_mask(frame: np.ndarray, y0_frac: float, y1_frac: float) -> np.ndarray:
    height, width = frame.shape[:2]
    y0 = int(height * y0_frac)
    y1 = int(height * y1_frac)
    crop = frame[y0:y1, 1 : width - 1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    # Brick rows are colorful and bright after preprocessing; this rejects black
    # background, gray walls, and most scoreboard pixels.
    return (saturation > 40) & (value > 45)


def brick_digest(frame: np.ndarray, y0_frac: float, y1_frac: float) -> tuple[bytes, int]:
    mask = brick_mask(frame, y0_frac, y1_frac)
    small = cv2.resize(mask.astype(np.uint8), (16, 6), interpolation=cv2.INTER_AREA)
    quantized = (small > 0).astype(np.uint8)
    return np.packbits(quantized.reshape(-1)).tobytes(), int(mask.sum())


def detect_ball(frame: np.ndarray, prev_frame: np.ndarray | None, next_frame: np.ndarray | None):
    if prev_frame is None and next_frame is None:
        return None

    def gray(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.int16)

    current = gray(frame)
    motion = np.zeros_like(current)
    if prev_frame is not None:
        motion = np.maximum(motion, np.abs(current - gray(prev_frame)))
    if next_frame is not None:
        motion = np.maximum(motion, np.abs(gray(next_frame) - current))

    height, width = current.shape
    mask = (motion >= 18) & (current >= 80)
    mask[: int(0.15 * height)] = False
    mask[int(0.82 * height) :] = False

    labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    best = None
    for label in range(1, labels):
        x, y, w, h, area = stats[label]
        if area < 1 or area > 24 or w > 8 or h > 8:
            continue
        component_motion = motion[y : y + h, x : x + w][mask[y : y + h, x : x + w]]
        component_intensity = current[y : y + h, x : x + w][mask[y : y + h, x : x + w]]
        score = float(component_motion.mean() + 0.1 * component_intensity.mean())
        if best is None or score > best[0]:
            best = (score, float(centroids[label][0]), float(centroids[label][1]))
    return None if best is None else (best[1], best[2])


def detect_paddle_x(frame: np.ndarray, y0_frac: float, y1_frac: float):
    height, width = frame.shape[:2]
    y0 = int(height * y0_frac)
    y1 = int(height * y1_frac)
    crop = frame[y0:y1]
    red = crop[..., 0].astype(np.int16)
    green = crop[..., 1].astype(np.int16)
    blue = crop[..., 2].astype(np.int16)
    mask = ((red > 90) & (red - np.maximum(green, blue) > 35)).astype(np.uint8)
    labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    best = None
    for label in range(1, labels):
        x, _, component_width, component_height, area = stats[label]
        if x <= 3 or x + component_width >= width - 3:
            continue
        if area < 3 or area > 40 or component_width < 4 or component_height > 3:
            continue
        if best is None or area > best[0]:
            best = (area, float(centroids[label][0]))
    return None if best is None else best[1]


def reservoir_add(sample: list[np.ndarray], frame: np.ndarray, seen: int, max_size: int, rng) -> None:
    if max_size <= 0:
        return
    if len(sample) < max_size:
        sample.append(frame.copy())
        return
    replacement_index = int(rng.integers(seen))
    if replacement_index < max_size:
        sample[replacement_index] = frame.copy()


def sampled_pairwise_l1(frames: list[np.ndarray], n_pairs: int, rng) -> dict[str, float | int | None]:
    if len(frames) < 2 or n_pairs <= 0:
        return summarize_values([])

    stack = np.asarray(frames, dtype=np.int16)
    n = len(stack)
    values = []
    for _ in range(n_pairs):
        i = int(rng.integers(n))
        j = int(rng.integers(n - 1))
        if j >= i:
            j += 1
        values.append(float(np.abs(stack[i] - stack[j]).mean()))
    return summarize_values(values)


def save_heatmap(counts: np.ndarray, output_path: Path, title: str) -> None:
    max_count = int(counts.max())
    scale = 48
    pad = 28
    image = Image.new("RGB", (counts.shape[1] * scale, counts.shape[0] * scale + pad), "white")
    draw = ImageDraw.Draw(image)
    draw.text((4, 6), title, fill=(0, 0, 0))
    for y in range(counts.shape[0]):
        for x in range(counts.shape[1]):
            count = int(counts[y, x])
            strength = 0 if max_count == 0 else count / max_count
            color = (
                int(255 * strength),
                int(40 + 120 * (1.0 - strength)),
                int(255 * (1.0 - strength)),
            )
            x0 = x * scale
            y0 = pad + y * scale
            draw.rectangle((x0, y0, x0 + scale - 1, y0 + scale - 1), fill=color, outline=(230, 230, 230))
            if count:
                draw.text((x0 + 4, y0 + 4), str(count), fill=(0, 0, 0))
    image.save(output_path)


def save_contact_sheet(frames: list[np.ndarray], output_path: Path, rng, cols=8, rows=8) -> None:
    if not frames:
        return
    indices = rng.choice(len(frames), size=min(cols * rows, len(frames)), replace=False)
    tile_h, tile_w = frames[0].shape[:2]
    scale = 3
    sheet = Image.new("RGB", (cols * tile_w * scale, rows * tile_h * scale), "white")
    for offset, frame_index in enumerate(indices):
        row = offset // cols
        col = offset % cols
        tile = Image.fromarray(frames[int(frame_index)]).resize(
            (tile_w * scale, tile_h * scale),
            Image.Resampling.NEAREST,
        )
        sheet.paste(tile, (col * tile_w * scale, row * tile_h * scale))
    sheet.save(output_path)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    exact_hashes = Counter()
    coarse_hashes = Counter()
    brick_hashes = Counter()
    action_counts = Counter()
    sample_frames: list[np.ndarray] = []

    episode_lengths = []
    episode_returns = []
    reward_counts = []
    positive_reward_episodes = 0
    total_frames = 0
    total_rewards = 0
    positive_reward_frames = 0
    adjacent_l1 = RunningStats()
    brightness = RunningStats()
    brick_pixels = RunningStats()
    episode_brick_layout_counts = []
    episode_brick_pixel_ranges = []

    ball_counts = np.zeros((args.grid_size, args.grid_size), dtype=np.int64)
    paddle_counts = np.zeros(args.grid_size, dtype=np.int64)
    ball_detected = 0
    paddle_detected = 0

    with h5py.File(args.h5_path, "r") as file:
        episode_keys = sorted(file.keys())
        if args.max_episodes is not None:
            episode_keys = episode_keys[: args.max_episodes]

        dataset_attrs = json_ready(dict(file.attrs.items()))
        logger.info(
            "Scanning %d episodes from %s (max_episodes=%s)",
            len(episode_keys),
            args.h5_path,
            args.max_episodes,
        )

        scan_start = time.monotonic()
        for episode_index, episode_key in enumerate(episode_keys):
            group = file[episode_key]
            frames = group["frames"]
            actions = group["actions"][:]
            rewards = group["rewards"][:]
            n_frames = len(frames)

            episode_lengths.append(n_frames)
            episode_return = float(group.attrs.get("return", rewards.sum()))
            episode_returns.append(episode_return)
            reward_count = int((rewards > 0).sum())
            reward_counts.append(reward_count)
            positive_reward_frames += reward_count
            total_rewards += float(rewards.sum())
            if episode_return > 0:
                positive_reward_episodes += 1

            for action in actions:
                action_counts[int(action)] += 1

            episode_brick_hashes = set()
            episode_brick_pixels = []
            previous = None
            for frame_index in range(n_frames):
                frame = np.asarray(frames[frame_index], dtype=np.uint8)
                next_frame = (
                    np.asarray(frames[frame_index + 1], dtype=np.uint8)
                    if frame_index + 1 < n_frames
                    else None
                )

                total_frames += 1
                exact_hashes[frame_digest(frame)] += 1
                coarse_hashes[coarse_digest(frame)] += 1

                brick_key, brick_pixel_count = brick_digest(frame, args.brick_y0_frac, args.brick_y1_frac)
                brick_hashes[brick_key] += 1
                episode_brick_hashes.add(brick_key)
                episode_brick_pixels.append(brick_pixel_count)
                brick_pixels.update(brick_pixel_count)

                brightness.update(float(frame.mean()))
                reservoir_add(sample_frames, frame, total_frames, args.sample_frames, rng)

                if previous is not None:
                    adjacent_l1.update(float(np.abs(frame.astype(np.int16) - previous.astype(np.int16)).mean()))

                ball = detect_ball(frame, previous, next_frame)
                if ball is not None:
                    ball_detected += 1
                    ball_x, ball_y = ball
                    x_bin = min(args.grid_size - 1, max(0, int(ball_x * args.grid_size / frame.shape[1])))
                    y_bin = min(args.grid_size - 1, max(0, int(ball_y * args.grid_size / frame.shape[0])))
                    ball_counts[y_bin, x_bin] += 1

                paddle_x = detect_paddle_x(frame, args.paddle_y0_frac, args.paddle_y1_frac)
                if paddle_x is not None:
                    paddle_detected += 1
                    x_bin = min(args.grid_size - 1, max(0, int(paddle_x * args.grid_size / frame.shape[1])))
                    paddle_counts[x_bin] += 1

                previous = frame

            episode_brick_layout_counts.append(len(episode_brick_hashes))
            episode_brick_pixel_ranges.append(int(max(episode_brick_pixels) - min(episode_brick_pixels)))

            elapsed = time.monotonic() - scan_start
            fps = total_frames / elapsed if elapsed > 0 else 0.0
            logger.info(
                "Episode %d/%d (%s): frames=%d return=%.2f rewards=%d brick_layouts=%d | "
                "cumulative frames=%d ball_det=%.1f%% paddle_det=%.1f%% elapsed=%.1fs (%.0f fps)",
                episode_index + 1,
                len(episode_keys),
                episode_key,
                n_frames,
                episode_return,
                reward_count,
                len(episode_brick_hashes),
                total_frames,
                100 * ball_detected / max(1, total_frames),
                100 * paddle_detected / max(1, total_frames),
                elapsed,
                fps,
            )

    logger.info("Scan complete: %d episodes, %d frames in %.1fs", len(episode_lengths), total_frames, time.monotonic() - scan_start)
    logger.info("Computing pairwise L1 distance over %d sampled pairs", args.pair_samples)
    pairwise = sampled_pairwise_l1(sample_frames, args.pair_samples, rng)
    exact_unique = len(exact_hashes)
    coarse_unique = len(coarse_hashes)
    brick_unique = len(brick_hashes)

    report = {
        "h5_path": args.h5_path,
        "dataset_attrs": dataset_attrs,
        "episodes_scanned": len(episode_lengths),
        "total_frames": total_frames,
        "episode_lengths": summarize_values(episode_lengths),
        "episode_returns": summarize_values(episode_returns),
        "positive_reward_episodes": positive_reward_episodes,
        "positive_reward_episode_fraction": positive_reward_episodes / max(1, len(episode_lengths)),
        "positive_reward_frames": positive_reward_frames,
        "total_reward": total_rewards,
        "rewards_per_episode": summarize_values(reward_counts),
        "action_counts": {str(key): int(value) for key, value in sorted(action_counts.items())},
        "frame_uniqueness": {
            "exact_unique_frames": exact_unique,
            "exact_unique_ratio": exact_unique / max(1, total_frames),
            "coarse_16x16_gray_unique_frames": coarse_unique,
            "coarse_16x16_gray_unique_ratio": coarse_unique / max(1, total_frames),
            "most_common_exact_frame_count": int(max(exact_hashes.values())) if exact_hashes else 0,
            "most_common_coarse_frame_count": int(max(coarse_hashes.values())) if coarse_hashes else 0,
        },
        "visual_distance": {
            "adjacent_frame_l1_0_255": adjacent_l1.to_dict(),
            "sampled_pairwise_frame_l1_0_255": pairwise,
            "brightness": brightness.to_dict(),
        },
        "breakout_coverage": {
            "ball_detected_frames": ball_detected,
            "ball_detected_fraction": ball_detected / max(1, total_frames),
            "ball_grid_nonempty_cells": int((ball_counts > 0).sum()),
            "ball_grid_total_cells": int(ball_counts.size),
            "ball_grid_entropy_bits": entropy_from_counts(ball_counts),
            "paddle_detected_frames": paddle_detected,
            "paddle_detected_fraction": paddle_detected / max(1, total_frames),
            "paddle_nonempty_x_bins": int((paddle_counts > 0).sum()),
            "paddle_entropy_bits": entropy_from_counts(paddle_counts),
            "brick_layout_unique_count": brick_unique,
            "brick_layout_unique_ratio": brick_unique / max(1, total_frames),
            "brick_pixels": brick_pixels.to_dict(),
            "brick_layouts_per_episode": summarize_values(episode_brick_layout_counts),
            "brick_pixel_range_per_episode": summarize_values(episode_brick_pixel_ranges),
            "ball_grid_counts": ball_counts.tolist(),
            "paddle_x_bin_counts": paddle_counts.tolist(),
        },
    }

    report_path = output_dir / "report.json"
    with report_path.open("w") as file:
        json.dump(json_ready(report), file, indent=2)

    save_heatmap(ball_counts, output_dir / "ball_heatmap.png", "ball detections")
    save_heatmap(paddle_counts.reshape(1, -1), output_dir / "paddle_heatmap.png", "paddle detections")
    if args.save_contact_sheet:
        save_contact_sheet(sample_frames, output_dir / "sample_frames.png", rng)

    print(f"Scanned episodes: {len(episode_lengths)}")
    print(f"Total frames: {total_frames}")
    print(
        "Episode length mean/median/max: "
        f"{report['episode_lengths']['mean']:.1f}/"
        f"{report['episode_lengths']['median']:.1f}/"
        f"{report['episode_lengths']['max']:.0f}"
    )
    print(
        "Positive-reward episodes: "
        f"{positive_reward_episodes}/{len(episode_lengths)} "
        f"({100 * report['positive_reward_episode_fraction']:.1f}%)"
    )
    print(
        "Unique frames exact/coarse: "
        f"{exact_unique}/{total_frames} ({100 * exact_unique / max(1, total_frames):.1f}%), "
        f"{coarse_unique}/{total_frames} ({100 * coarse_unique / max(1, total_frames):.1f}%)"
    )
    print(f"Adjacent frame L1 mean: {report['visual_distance']['adjacent_frame_l1_0_255']['mean']:.3f}")
    print(f"Sampled pairwise L1 mean: {report['visual_distance']['sampled_pairwise_frame_l1_0_255']['mean']:.3f}")
    print(
        "Ball coverage: "
        f"{report['breakout_coverage']['ball_grid_nonempty_cells']}/"
        f"{report['breakout_coverage']['ball_grid_total_cells']} grid cells, "
        f"detected in {100 * report['breakout_coverage']['ball_detected_fraction']:.1f}% of frames"
    )
    print(
        "Brick layouts: "
        f"{brick_unique} unique, "
        f"per-episode median {report['breakout_coverage']['brick_layouts_per_episode']['median']:.1f}, "
        f"brick-pixel range median {report['breakout_coverage']['brick_pixel_range_per_episode']['median']:.1f}"
    )
    print(f"Saved report: {report_path}")
    print(f"Saved heatmaps: {output_dir / 'ball_heatmap.png'}, {output_dir / 'paddle_heatmap.png'}")


if __name__ == "__main__":
    main()
