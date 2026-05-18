import h5py
import torch
from torch.utils.data import Dataset

class AtariEpisodeDataset(Dataset):
    def __init__(self, h5_path, seq_len=16, return_tokens=False):
        self.h5_path = h5_path
        self.file = None
        self.seq_len = seq_len
        self.return_tokens = return_tokens

        # build index of all valid (episode, start_frame) pairs

        self.index = []
        with h5py.File(self.h5_path, "r") as file:
            for ep_key in file.keys():
                ep = file[ep_key]
                if return_tokens and 'tokens' not in ep:
                    raise ValueError(
                        f'return_tokens=True but {ep_key} has no precomputed tokens in {h5_path}'
                    )
                n_frames = len(ep['frames'])
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
        actions = torch.tensor(ep['actions'][start:start + self.seq_len], dtype=torch.long)
        rewards = torch.tensor(ep['rewards'][start:start + self.seq_len], dtype=torch.float32)

        if self.return_tokens:
            tokens = ep['tokens'][start:start + self.seq_len]  # (T, H, W)
            tokens = torch.tensor(tokens, dtype=torch.long)
            return tokens, actions, rewards

        frames = ep['frames'][start:start + self.seq_len]
        frames = torch.tensor(frames, dtype=torch.float32) / 255.0
        frames = frames.permute(0, 3, 1, 2).contiguous()  # (T, H, W, C) -> (T, C, H, W)
        return frames, actions, rewards
