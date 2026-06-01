import torch

from world_model.embeddings import compute_max_seq_len


class LayerCache:
    def __init__(self, B, n_heads, max_seq, head_dim, dtype, device):
        self.K = torch.empty(B, n_heads, max_seq, head_dim, dtype=dtype, device=device)
        self.V = torch.empty(B, n_heads, max_seq, head_dim, dtype=dtype, device=device)
        self.max_seq = max_seq
        self.write_pos = 0

    def append(self, k_new, v_new):
        """k_new, v_new: (B, n_heads, n_new, head_dim)"""
        n_new = k_new.shape[2]
        assert n_new <= self.max_seq, f'cannot append {n_new} > max_seq={self.max_seq}'

        start = self.write_pos % self.max_seq
        end = start + n_new
        if end <= self.max_seq:
            self.K[:, :, start:end] = k_new
            self.V[:, :, start:end] = v_new
        else:
            split = self.max_seq - start
            self.K[:, :, start:] = k_new[:, :, :split]
            self.V[:, :, start:] = v_new[:, :, :split]
            wrap_end = end - self.max_seq
            self.K[:, :, :wrap_end] = k_new[:, :, split:]
            self.V[:, :, :wrap_end] = v_new[:, :, split:]
        self.write_pos += n_new

    def read(self):
        filled = min(self.write_pos, self.max_seq)
        if filled < self.max_seq:
            return self.K[:, :, :filled], self.V[:, :, :filled]
        return self.K, self.V

    def reset(self):
        self.write_pos = 0


class KVCache:
    def __init__(self, cfg):
        self.num_layers = cfg.model.num_layers
        self.B = cfg.generate.batch_size
        self.n_heads = cfg.model.num_heads
        self.max_seq = compute_max_seq_len(cfg)
        self.head_dim = cfg.model.d_model // cfg.model.num_heads
        self.caches = []

    def allocate(self, dtype, device):
        self.caches = [
            LayerCache(self.B, self.n_heads, self.max_seq, self.head_dim, dtype, device)
            for _ in range(self.num_layers)
        ]

    def append(self, layer_idx, k_new, v_new):
        self.caches[layer_idx].append(k_new, v_new)

    def read(self, layer_idx):
        return self.caches[layer_idx].read()

    def reset(self):
        for cache in self.caches:
            cache.reset()

    @property
    def length(self):
        return self.caches[0].write_pos if self.caches else 0
