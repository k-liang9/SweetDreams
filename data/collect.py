import argparse
from pathlib import Path

import ale_py
import cv2
import gymnasium as gym
import h5py
import numpy as np


gym.register_envs(ale_py)


def preprocess_frame(frame, size=64):
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def get_action_ids(env):
    meanings = env.unwrapped.get_action_meanings()

    def find(name):
        return meanings.index(name) if name in meanings else None

    return {
        "noop": find("NOOP") or 0,
        "fire": find("FIRE"),
        "left": find("LEFT"),
        "right": find("RIGHT"),
        "meanings": meanings,
    }


def random_action(env, rng):
    return int(rng.integers(env.action_space.n))


def object_x(gray, y0, y1, threshold, min_area, max_area, min_width=1, max_width=999):
    crop = gray[y0:y1]
    mask = (crop > threshold).astype(np.uint8)
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    candidates = []
    for label in range(1, n_labels):
        _, _, width, height, area = stats[label]
        if min_area <= area <= max_area and min_width <= width <= max_width:
            candidates.append((area, centroids[label][0]))
    return max(candidates)[1] if candidates else None


def heuristic_action(frame, action_ids, rng, epsilon):
    if rng.random() < epsilon:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    height = gray.shape[0]
    ball_x = object_x(gray, int(0.25 * height), int(0.82 * height), 180, 1, 80, max_width=12)
    paddle_x = object_x(gray, int(0.72 * height), int(0.96 * height), 45, 10, 300, min_width=8)

    if ball_x is None:
        return action_ids["fire"] or action_ids["noop"]
    if paddle_x is None:
        return action_ids["noop"]
    if ball_x < paddle_x - 4 and action_ids["left"] is not None:
        return action_ids["left"]
    if ball_x > paddle_x + 4 and action_ids["right"] is not None:
        return action_ids["right"]
    return action_ids["noop"]


def choose_action(frame, env, action_ids, rng, policy, epsilon, needs_fire):
    if needs_fire and action_ids["fire"] is not None:
        return action_ids["fire"], False

    if policy == "random":
        return random_action(env, rng), False

    action = heuristic_action(frame, action_ids, rng, epsilon)
    if action is None:
        action = random_action(env, rng)
    return action, False


def collect_episodes(
    game="ALE/Breakout-v5",
    n_episodes=2500,
    output_path="data/breakout.h5",
    policy="mixed",
    random_episode_frac=0.25,
    epsilon=0.05,
    seed=0,
    frame_size=64,
    fire_on_reset=True,
):
    rng = np.random.default_rng(seed)
    env = gym.make(game, render_mode="rgb_array")
    env.action_space.seed(seed)
    action_ids = get_action_ids(env)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["game"] = game
        f.attrs["policy"] = policy
        f.attrs["random_episode_frac"] = random_episode_frac
        f.attrs["epsilon"] = epsilon
        f.attrs["seed"] = seed
        f.attrs["frame_size"] = frame_size
        f.attrs["action_meanings"] = np.array(action_ids["meanings"], dtype=h5py.string_dtype())

        for ep_idx in range(n_episodes):
            env.reset(seed=seed + ep_idx)
            ep_policy = policy
            if policy == "mixed":
                ep_policy = "random" if rng.random() < random_episode_frac else "heuristic"

            frames, actions, rewards, dones = [], [], [], []
            needs_fire = fire_on_reset
            episode_return = 0.0
            terminated = truncated = False

            while not (terminated or truncated):
                raw_frame = env.render()
                action, needs_fire = choose_action(
                    raw_frame,
                    env,
                    action_ids,
                    rng,
                    ep_policy,
                    epsilon,
                    needs_fire,
                )
                _, reward, terminated, truncated, _ = env.step(action)

                frames.append(preprocess_frame(raw_frame, size=frame_size))
                actions.append(action)
                rewards.append(reward)
                dones.append(terminated or truncated)
                episode_return += float(reward)

            grp = f.create_group(f"episode_{ep_idx:04d}")
            grp.attrs["policy"] = ep_policy
            grp.attrs["return"] = episode_return
            grp.create_dataset("frames", data=np.asarray(frames, dtype=np.uint8))
            grp.create_dataset("actions", data=np.asarray(actions, dtype=np.int32))
            grp.create_dataset("rewards", data=np.asarray(rewards, dtype=np.float32))
            grp.create_dataset("dones", data=np.asarray(dones, dtype=bool))

            if ep_idx % 50 == 0:
                print(
                    f"Episode {ep_idx}/{n_episodes} "
                    f"policy={ep_policy} return={episode_return:.1f} length={len(frames)}"
                )

    env.close()
    print(f"Saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ALE/Breakout-v5")
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--output-path", default="data/breakout.h5")
    parser.add_argument("--policy", choices=["random", "heuristic", "mixed"], default="mixed")
    parser.add_argument("--random-episode-frac", type=float, default=0.25)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-size", type=int, default=64)
    parser.add_argument("--no-fire-on-reset", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_episodes(
        game=args.game,
        n_episodes=args.n_episodes,
        output_path=args.output_path,
        policy=args.policy,
        random_episode_frac=args.random_episode_frac,
        epsilon=args.epsilon,
        seed=args.seed,
        frame_size=args.frame_size,
        fire_on_reset=not args.no_fire_on_reset,
    )
