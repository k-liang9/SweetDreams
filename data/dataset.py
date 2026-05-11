import h5py
import torch
from torch.utils.data import Dataset

class AtariEpisodeDataset(Dataset):
    def __init__(self, h5_path, seq_len=16):
        self.h5_path = h5_path
        self.file = None
        self.seq_len = seq_len
        
        # build index of all valid (episode, start_frame) pairs
        
        self.index = []
        with h5py.File(self.h5_path, "r") as file:
            for ep_key in file.keys():
                n_frames = len(file[ep_key]['frames'])
                for start in range(n_frames - seq_len + 1):
                    self.index.append((ep_key, start))
                
    def __len__(self):
        return len(self.index)
    
    def _get_file(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")
        return self.file
    
    def __getitem__(self, idx):
        ep_key, start = self.index[idx]
        ep = self._get_file()[ep_key]
        frames = ep['frames'][start:start + self.seq_len] # (T, 64, 64)
        actions = ep['actions'][start:start + self.seq_len] # (T,)
        rewards = ep['rewards'][start:start + self.seq_len] # (T,)
        
        # normalize frames to [0, 1]
        frames = torch.tensor(frames, dtype=torch.float32) / 255.0
        frames = frames.unsqueeze(1) # (T, 1, 64, 64) - channel dim
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        return frames, actions, rewards
