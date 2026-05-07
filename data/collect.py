import gymnasium as gym
import ale_py
import numpy as np
import h5py
import cv2
from pathlib import Path

gym.register_envs(ale_py)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    return resized

def collect_episodes(game="ALE/Breakout-v5", n_episodes=500, output_path="data/breakout.h5"):
    env = gym.make(game, render_mode='rgb_array')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, "w") as f:
        for ep_idx in range(n_episodes):
            obs, _ = env.reset()
            
            frames, actions, rewards, dones = [], [], [], []
            
            terminated = truncated = False
            while not (terminated or truncated):
                frame = preprocess_frame(env.render())
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                frames.append(frame)
                actions.append(action)
                rewards.append(reward)
                dones.append(terminated or truncated)
                
            grp = f.create_group(f'episode_{ep_idx:04d}')
            grp.create_dataset("frames",  data=np.array(frames,  dtype=np.uint8))
            grp.create_dataset("actions", data=np.array(actions, dtype=np.int32))
            grp.create_dataset("rewards", data=np.array(rewards, dtype=np.float32))
            grp.create_dataset("dones",   data=np.array(dones,   dtype=bool))
            
            if ep_idx % 50 == 0:
                print(f"Episode {ep_idx}/{n_episodes} — length {len(frames)}")
        
        env.close()
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    collect_episodes()
