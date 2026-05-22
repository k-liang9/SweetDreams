import torch
import torch.nn as nn

def compute_max_seq_len(cfg):
    """Interleaved length of one training sequence; rollout uses a sliding window of the same size.

    cfg.data.seq_len is the number of past frames in context. The window holds seq_len + 1
    frames total (context + one target frame to predict) and seq_len actions between them.
    """
    seq_len = cfg.data.seq_len
    total_frames = seq_len + 1
    return total_frames * cfg.model.tokens_per_frame + seq_len


class WorldModelEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.model.d_model
        self.max_seq_len = compute_max_seq_len(cfg)
        N = cfg.model.tokens_per_frame
        self.frame_token_embedding = nn.Embedding(cfg.model.num_frame_tokens, self.d_model)
        self.action_embedding = nn.Embedding(cfg.model.num_actions, self.d_model)
        self.type_embedding = nn.Embedding(2, self.d_model)
        self.dropout = nn.Dropout(cfg.model.dropout)

        # 0 = frame token, 1 = action token. Interleave pattern is fixed by N:
        # within each block of N+1 positions, the last position is an action.
        T_max = cfg.data.seq_len + 1
        type_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
        action_positions = torch.arange(1, T_max) * (N + 1) - 1
        type_ids[action_positions] = 1
        self.register_buffer('type_ids', type_ids, persistent=False)

    def forward(self, frame_tokens, actions):
        """
        frame_tokens: (B, T, N)
        actions:      (B, T - 1)

        returns:
            x: (B, S, d_model)
        """
        B, T, N = frame_tokens.shape
        if actions.shape != (B, T - 1):
            raise ValueError(
                f'Expected actions with shape {(B, T - 1)}, got {tuple(actions.shape)}'
            )

        frame_x = self.frame_token_embedding(frame_tokens)  # (B,T,N,d)
        action_x = self.action_embedding(actions)           # (B,T-1,d)

        chunks = []
        for t in range(T):
            chunks.append(frame_x[:, t])                    # (B,N,d)
            if t < T-1:
                chunks.append(action_x[:, t:t + 1])         # (B,1,d)

        x = torch.cat(chunks, dim=1)                        # (B,S,d)
        S = x.size(1)
        if S > self.max_seq_len:
            raise ValueError(
                f'Interleaved sequence length {S} exceeds max_seq_len {self.max_seq_len}'
            )

        x = x + self.type_embedding(self.type_ids[:S])      # (S,d) broadcasts over B
        return self.dropout(x)

class RoPE(nn.Module):
    """Rotary positional embedding (half-rotated convention).

    Pair j of dimensions (x[..., j], x[..., j + head_dim/2]) of a token at
    position m is rotated by angle m * theta_j, where theta_j = base^(-2j/head_dim).
    Apply to Q and K only, never V.
    """

    def __init__(self, cfg, base: float = 10000.0):
        super().__init__()
        head_dim = cfg.model.d_model // cfg.model.num_heads
        max_seq_len = compute_max_seq_len(cfg)
        assert head_dim % 2 == 0, f'head_dim must be even, got {head_dim}'
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # theta_j for j = 0..head_dim/2 - 1
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim/2)

        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def _get_cos_sin(self, start: int, end: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch cos/sin for positions [start, end); compute on the fly past max_seq_len."""
        if end <= self.max_seq_len:
            return self.cos[start:end], self.sin[start:end]
        positions = torch.arange(start, end, device=device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq)
        return angles.cos(), angles.sin()

    def forward(self, x: torch.Tensor, start: int = 0) -> torch.Tensor:
        """
        x:     (B, num_heads, T, head_dim) — Q or K after the multi-head split.
        start: absolute position of x[..., 0, :]. 0 for training/prefill,
               cache_len for single-token decode.
        """
        _, _, T, D = x.shape
        assert D == self.head_dim, f'expected head_dim={self.head_dim}, got {D}'

        cos, sin = self._get_cos_sin(start, start + T, x.device)
        cos = cos.to(x.dtype)[None, None, :, :]
        sin = sin.to(x.dtype)[None, None, :, :]

        x1, x2 = x[..., : D // 2], x[..., D // 2 :]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.cat([out1, out2], dim=-1)