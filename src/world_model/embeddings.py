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
        # Spatial slots: 0..N-1 for the H*W grid positions within a frame, N for action tokens.
        self.spatial_embedding = nn.Embedding(N + 1, self.d_model)
        self.dropout = nn.Dropout(cfg.model.dropout)

        # spatial_ids[p] = p % (N+1):
        #   p % (N+1) ∈ 0..N-1 → frame token at that spatial slot
        #   p % (N+1) == N     → action token
        spatial_ids = torch.arange(self.max_seq_len) % (N + 1)
        self.register_buffer('spatial_ids', spatial_ids, persistent=False)

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

        x = x + self.spatial_embedding(self.spatial_ids[:S])  # (S,d) broadcasts over B
        return self.dropout(x)

class RoPE(nn.Module):
    """Rotary positional embedding on the temporal axis (half-rotated convention).

    All N+1 tokens in one block (N frame tokens + 1 action token) share a single
    time index t. Pair j of dimensions (x[..., j], x[..., j + head_dim/2]) is
    rotated by angle t * theta_j, where theta_j = base^(-2j/head_dim).
    Apply to Q and K only, never V.

    Same-frame attention (Δt = 0) gets RoPE identity → pure content dot.
    Cross-frame attention encodes the temporal offset.
    """

    def __init__(self, cfg, base: float = 10000.0):
        super().__init__()
        head_dim = cfg.model.d_model // cfg.model.num_heads
        max_seq_len = compute_max_seq_len(cfg)
        N = cfg.model.tokens_per_frame
        T_max = cfg.data.seq_len + 1
        assert head_dim % 2 == 0, f'head_dim must be even, got {head_dim}'
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.block_size = N + 1

        # theta_j for j = 0..head_dim/2 - 1
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # cos/sin tables indexed by TIME (frame index), size T_max (= 17 by default)
        # rather than by absolute position (size max_seq_len = 1104).
        times = torch.arange(T_max, dtype=torch.float32)
        angles = torch.outer(times, inv_freq)  # (T_max, head_dim/2)
        self.register_buffer('cos_time', angles.cos(), persistent=False)
        self.register_buffer('sin_time', angles.sin(), persistent=False)

        # time_ids[p] = p // (N+1) — every token in block t shares time t.
        time_ids = torch.arange(max_seq_len) // self.block_size
        self.register_buffer('time_ids', time_ids, persistent=False)

    def _get_cos_sin(self, start: int, end: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch cos/sin for absolute positions [start, end), looked up via time index."""
        if end <= self.max_seq_len:
            t = self.time_ids[start:end]
            return self.cos_time[t], self.sin_time[t]
        # Past the precomputed range (e.g. indefinite-horizon rollout): compute on the fly.
        positions = torch.arange(start, end, device=device, dtype=torch.long)
        t = (positions // self.block_size).to(self.inv_freq.dtype)
        angles = torch.outer(t, self.inv_freq)
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