import argparse
import multiprocessing as mp
import os
from collections import deque
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import ale_py
import cv2
import gymnasium as gym
import h5py
import numpy as np


gym.register_envs(ale_py)


PRETRAINED_SPECS = {
    "noisy": ("DQN", "sb3/dqn-BreakoutNoFrameskip-v4", "dqn-BreakoutNoFrameskip-v4.zip"),
    "deterministic": ("PPO", "sb3/ppo-BreakoutNoFrameskip-v4", "ppo-BreakoutNoFrameskip-v4.zip"),
}

SB3_LOAD_OVERRIDES = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
    "exploration_schedule": lambda _: 0.0,
    "buffer_size": 1,
    "optimize_memory_usage": False,
    "replay_buffer_kwargs": {"handle_timeout_termination": False},
}


def get_action_ids(env):
    meanings = env.unwrapped.get_action_meanings()
    return {
        "noop": meanings.index("NOOP") if "NOOP" in meanings else 0,
        "fire": meanings.index("FIRE") if "FIRE" in meanings else None,
        "meanings": meanings,
    }


def preprocess_frame(frame, size):
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def push_frame_stack(stack, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA).astype(np.uint8)
    if not stack:
        stack.extend([resized] * 4)
    else:
        stack.append(resized)
    return np.stack(stack, axis=0)[None]


def pretrained_local_path(spec_name):
    from huggingface_hub import hf_hub_download

    _, repo, fname = PRETRAINED_SPECS[spec_name]
    return str(hf_hub_download(repo_id=repo, filename=fname))


def load_pretrained(device):
    from stable_baselines3 import DQN, PPO

    classes = {"DQN": DQN, "PPO": PPO}
    models = {}
    for name, (cls_name, _, _) in PRETRAINED_SPECS.items():
        path = pretrained_local_path(name)
        models[name] = {
            "model": classes[cls_name].load(path, device=device, custom_objects=SB3_LOAD_OVERRIDES),
            "path": path,
            "stack": deque(maxlen=4),
        }
    return models


def run_episode(
    env, action_ids, rng, ep_policy, models, *,
    epsilon, frame_size, fire_on_reset, noop_max, max_steps, seed,
):
    env.reset(seed=seed)
    noop_steps = int(rng.integers(noop_max + 1)) if noop_max > 0 else 0
    for _ in range(noop_steps):
        _, _, term, trunc, _ = env.step(action_ids["noop"])
        if term or trunc:
            env.reset(seed=seed)
    prefire = fire_on_reset and action_ids["fire"] is not None
    if prefire:
        _, _, term, trunc, _ = env.step(action_ids["fire"])
        if term or trunc:
            env.reset(seed=seed)

    agent = None
    if ep_policy.startswith("pretrained"):
        agent = models["deterministic" if ep_policy.endswith("deterministic") else "noisy"]
        agent["stack"].clear()

    frames, actions, rewards, dones = [], [], [], []
    needs_fire = fire_on_reset and not prefire
    ep_return = 0.0
    collector_truncated = False
    terminated = truncated = False

    while not (terminated or truncated):
        raw = env.render()

        if needs_fire:
            if agent is not None:
                push_frame_stack(agent["stack"], raw)
            action = action_ids["fire"]
            needs_fire = False
        elif ep_policy == "random":
            action = int(rng.integers(env.action_space.n))
        elif ep_policy.endswith("noisy") and rng.random() < epsilon:
            push_frame_stack(agent["stack"], raw)
            action = int(rng.integers(env.action_space.n))
        else:
            obs = push_frame_stack(agent["stack"], raw)
            pred, _ = agent["model"].predict(obs, deterministic=True)
            action = int(np.asarray(pred).reshape(-1)[0])

        _, reward, terminated, truncated, _ = env.step(action)
        frames.append(preprocess_frame(raw, frame_size))
        actions.append(action)
        rewards.append(reward)
        dones.append(terminated or truncated)
        ep_return += float(reward)

        if max_steps and len(frames) >= max_steps and not (terminated or truncated):
            collector_truncated = True
            dones[-1] = True
            break

    return {
        "frames": np.asarray(frames, dtype=np.uint8),
        "actions": np.asarray(actions, dtype=np.int32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=bool),
        "return": ep_return,
        "policy": ep_policy,
        "noop_start_steps": noop_steps,
        "prefire_after_noop_start": prefire,
        "collector_truncated": collector_truncated,
    }


def write_episode(group, data):
    group.attrs["policy"] = data["policy"]
    group.attrs["return"] = data["return"]
    group.attrs["collector_truncated"] = data["collector_truncated"]
    group.attrs["noop_start_steps"] = data["noop_start_steps"]
    group.attrs["prefire_after_noop_start"] = data["prefire_after_noop_start"]
    group.create_dataset("frames", data=data["frames"])
    group.create_dataset("actions", data=data["actions"])
    group.create_dataset("rewards", data=data["rewards"])
    group.create_dataset("dones", data=data["dones"])


def _run_worker(args):
    (worker_id, assignments, shard_path, game, frame_size, fire_on_reset,
     noop_max, max_steps, epsilon, seed, needs_pretrained, device) = args

    cv2.setNumThreads(1)
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass

    env = gym.make(game, render_mode="rgb_array")
    env.action_space.seed(seed + worker_id)
    action_ids = get_action_ids(env)
    rng = np.random.default_rng(seed + 7919 * (worker_id + 1))
    models = load_pretrained(device) if needs_pretrained else None

    with h5py.File(shard_path, "w") as f:
        for i, (ep_idx, ep_policy) in enumerate(assignments):
            data = run_episode(
                env, action_ids, rng, ep_policy, models,
                epsilon=epsilon, frame_size=frame_size,
                fire_on_reset=fire_on_reset, noop_max=noop_max,
                max_steps=max_steps, seed=seed + ep_idx,
            )
            if data["return"] == 0.0:
                print(
                    f"[worker {worker_id}] {i + 1}/{len(assignments)} "
                    f"ep={ep_idx} policy={ep_policy} DROPPED (return=0, length={len(data['frames'])})",
                    flush=True,
                )
                continue
            write_episode(f.create_group(f"episode_{ep_idx:04d}"), data)
            print(
                f"[worker {worker_id}] {i + 1}/{len(assignments)} "
                f"ep={ep_idx} policy={ep_policy} "
                f"return={data['return']:.1f} length={len(data['frames'])}",
                flush=True,
            )

    env.close()
    return shard_path


def collect_episodes(
    game="ALE/Breakout-v5",
    n_episodes=300,
    output_path="data/breakout.h5",
    policy="mixed",
    epsilon=0.05,
    seed=0,
    frame_size=84,
    fire_on_reset=True,
    pretrained_device="cpu",
    max_episode_steps=None,
    noop_max=30,
    n_workers=0,
):
    if max_episode_steps is not None and max_episode_steps <= 0:
        max_episode_steps = None
    if n_workers <= 0:
        n_workers = max(1, mp.cpu_count() // 2)
    n_workers = min(n_workers, n_episodes)
    needs_pretrained = policy in {"pretrained", "mixed"}

    plan_rng = np.random.default_rng(seed)
    if policy == "mixed":
        n_random = 0
        n_det = round(0.3 * n_episodes)
        n_noisy = n_episodes - n_random - n_det
        episode_policies = (
            ["pretrained_noisy"] * n_noisy
            + ["pretrained_deterministic"] * n_det
            + ["random"] * n_random
        )
        plan_rng.shuffle(episode_policies)
    elif policy == "pretrained":
        episode_policies = ["pretrained_deterministic"] * n_episodes
    else:
        episode_policies = [policy] * n_episodes

    # round-robin so each worker gets a roughly balanced policy mix
    assignments_per_worker = [[] for _ in range(n_workers)]
    for ep_idx, ep_pol in enumerate(episode_policies):
        assignments_per_worker[ep_idx % n_workers].append((ep_idx, ep_pol))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = output_path.parent / f".{output_path.stem}_shards"
    shard_dir.mkdir(exist_ok=True)
    shard_paths = [shard_dir / f"shard_{w:03d}.h5" for w in range(n_workers)]

    probe_env = gym.make(game)
    action_meanings = probe_env.unwrapped.get_action_meanings()
    probe_env.close()
    pretrained_paths = (
        {name: pretrained_local_path(name) for name in PRETRAINED_SPECS}
        if needs_pretrained else {}
    )

    worker_args = [
        (w, assignments_per_worker[w], str(shard_paths[w]),
         game, frame_size, fire_on_reset, noop_max,
         max_episode_steps or 0, epsilon, seed,
         needs_pretrained, pretrained_device)
        for w in range(n_workers)
    ]

    print(f"Spawning {n_workers} workers for {n_episodes} episodes "
          f"(pretrained_device={pretrained_device})")
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for done_path in pool.imap_unordered(_run_worker, worker_args):
            print(f"[main] shard finished: {done_path}", flush=True)

    print(f"Merging {n_workers} shards into {output_path}")
    with h5py.File(output_path, "w") as out:
        out.attrs["game"] = game
        out.attrs["policy"] = policy
        out.attrs["epsilon"] = epsilon
        out.attrs["seed"] = seed
        out.attrs["frame_size"] = frame_size
        out.attrs["max_episode_steps"] = -1 if max_episode_steps is None else max_episode_steps
        out.attrs["noop_max"] = noop_max
        out.attrs["action_meanings"] = np.array(action_meanings, dtype=h5py.string_dtype())
        if policy == "mixed":
            out.attrs["mixed_policy_split"] = "70% dqn+epsilon, 20% ppo deterministic, 10% random"
        if needs_pretrained:
            out.attrs["pretrained_noisy_model"] = PRETRAINED_SPECS["noisy"][1]
            out.attrs["pretrained_noisy_path"] = pretrained_paths["noisy"]
            out.attrs["pretrained_deterministic_model"] = PRETRAINED_SPECS["deterministic"][1]
            out.attrs["pretrained_deterministic_path"] = pretrained_paths["deterministic"]
        for shard in shard_paths:
            with h5py.File(shard, "r") as f:
                for key in sorted(f.keys()):
                    f.copy(key, out)
            shard.unlink()
    shard_dir.rmdir()
    print(f"Saved to {output_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="ALE/Breakout-v5")
    p.add_argument("--n-episodes", type=int, default=300)
    p.add_argument("--output-path", default="data/breakout.h5")
    p.add_argument("--policy", choices=["random", "pretrained", "mixed"], default="mixed")
    p.add_argument("--epsilon", type=float, default=0.05,
                   help="Random action probability for the 70%% noisy pretrained bucket.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frame-size", type=int, default=84)
    p.add_argument("--max-episode-steps", type=int, default=5000,
                   help="Collector-side cap. Past ~20k frames Breakout's palette cycles and tints the rendered RGB.")
    p.add_argument("--no-fire-on-reset", action="store_true")
    p.add_argument("--pretrained-device", default="cpu",
                   help="Each worker loads its own copy; CPU is recommended for multiprocessing.")
    p.add_argument("--noop-max", type=int, default=30)
    p.add_argument("--n-workers", type=int, default=0,
                   help="0 (default) → cpu_count() // 2. Set explicitly to tune.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_episodes(
        game=args.game,
        n_episodes=args.n_episodes,
        output_path=args.output_path,
        policy=args.policy,
        epsilon=args.epsilon,
        seed=args.seed,
        frame_size=args.frame_size,
        fire_on_reset=not args.no_fire_on_reset,
        pretrained_device=args.pretrained_device,
        max_episode_steps=args.max_episode_steps,
        noop_max=args.noop_max,
        n_workers=args.n_workers,
    )
