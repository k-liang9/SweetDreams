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
        # 3D RoPE encodes spatial (row, col) and temporal positions inside attention,
        # so no learned position vectors are added here. A small binary type marker
        # (0=frame, 1=action) guards against the two content tables drifting into
        # overlapping subspaces during training.
        self.type_embedding = nn.Embedding(2, self.d_model)
        self.dropout = nn.Dropout(cfg.model.dropout)

        # type_ids[p] = 1 if position p is the action slot of its block, else 0.
        type_ids = ((torch.arange(self.max_seq_len) % (N + 1)) == N).long()
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
    """3D axial Rotary Position Embedding over (row, col, time).

    head_dim is split into three equal slices of size D/3. Each slice applies
    standard 1D RoPE on D/6 pairs, using one of the three coordinate axes:
      slice 0 (dims 0..D/3)     rotates by row
      slice 1 (dims D/3..2D/3)  rotates by col
      slice 2 (dims 2D/3..D)    rotates by time

    Position assignment for the interleaved (frame_tokens, action) sequence:
      Frame token at within-block index k:  (row=k//W, col=k%W, time=t)
      Action token (last in block):         (row=H,    col=W,    time=t)
    The action sentinel (H, W) is one past the grid in each spatial axis, so
    action vs frame is distinguishable from the rotation alone.

    Apply to Q and K only, never V.
    """

    def __init__(self, cfg, base: float = 10000.0):
        super().__init__()
        head_dim = cfg.model.d_model // cfg.model.num_heads
        assert head_dim % 6 == 0, (
            f'head_dim must be divisible by 6 for 3D RoPE (split into 3 axes, each with '
            f'even pair count); got head_dim={head_dim}'
        )
        N = cfg.model.tokens_per_frame
        H = W = int(round(N ** 0.5))
        assert H * W == N, f'tokens_per_frame={N} must be a perfect square for 3D RoPE'
        T_max = cfg.data.seq_len + 1
        max_seq_len = compute_max_seq_len(cfg)

        self.head_dim = head_dim
        self.axis_dim = head_dim // 3
        self.max_seq_len = max_seq_len
        self.block_size = N + 1
        self.H = H
        self.W = W

        # Per-axis frequencies: theta_j = base^(-2j/axis_dim), j = 0..axis_dim/2 - 1
        axis_dim = self.axis_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # cos/sin tables per axis, indexed by axis position.
        # Row/col include the sentinel value (H, W), so sizes are H+1 and W+1.
        rows = torch.arange(H + 1, dtype=torch.float32)
        cols = torch.arange(W + 1, dtype=torch.float32)
        times = torch.arange(T_max, dtype=torch.float32)
        self.register_buffer('cos_row', torch.outer(rows, inv_freq).cos(), persistent=False)
        self.register_buffer('sin_row', torch.outer(rows, inv_freq).sin(), persistent=False)
        self.register_buffer('cos_col', torch.outer(cols, inv_freq).cos(), persistent=False)
        self.register_buffer('sin_col', torch.outer(cols, inv_freq).sin(), persistent=False)
        self.register_buffer('cos_time', torch.outer(times, inv_freq).cos(), persistent=False)
        self.register_buffer('sin_time', torch.outer(times, inv_freq).sin(), persistent=False)

        # Per-absolute-position (row, col, time) lookup buffers.
        positions = torch.arange(max_seq_len)
        time_ids = positions // self.block_size
        within = positions % self.block_size
        is_action = within == N
        row_ids = torch.where(is_action, torch.full_like(within, H), within // W)
        col_ids = torch.where(is_action, torch.full_like(within, W), within % W)
        self.register_buffer('row_ids', row_ids, persistent=False)
        self.register_buffer('col_ids', col_ids, persistent=False)
        self.register_buffer('time_ids', time_ids, persistent=False)

    def _coords(self, start: int, end: int, device):
        """Return (row_ids, col_ids, time_ids) for absolute positions [start, end)."""
        if end <= self.max_seq_len:
            return self.row_ids[start:end], self.col_ids[start:end], self.time_ids[start:end]
        # Past the precomputed range — compute on the fly (indefinite-horizon decode).
        positions = torch.arange(start, end, device=device, dtype=torch.long)
        time_ids = positions // self.block_size
        within = positions % self.block_size
        N = self.block_size - 1
        is_action = within == N
        row_ids = torch.where(is_action, torch.full_like(within, self.H), within // self.W)
        col_ids = torch.where(is_action, torch.full_like(within, self.W), within % self.W)
        return row_ids, col_ids, time_ids

    def _axis_cos_sin(self, ids, cos_table, sin_table):
        """Look up cos/sin from a precomputed axis table; fall back to on-the-fly compute if needed."""
        if ids.numel() == 0 or int(ids.max()) < cos_table.size(0):
            return cos_table[ids], sin_table[ids]
        angles = torch.outer(ids.to(self.inv_freq.dtype), self.inv_freq)
        return angles.cos(), angles.sin()

    def _rotate(self, x_slice, cos, sin):
        D = x_slice.shape[-1]
        cos = cos.to(x_slice.dtype)[None, None, :, :]
        sin = sin.to(x_slice.dtype)[None, None, :, :]
        x1, x2 = x_slice[..., : D // 2], x_slice[..., D // 2 :]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.cat([out1, out2], dim=-1)

    def forward(self, x: torch.Tensor, start: int = 0) -> torch.Tensor:
        """
        x:     (B, num_heads, T, head_dim) — Q or K after the multi-head split.
        start: absolute position of x[..., 0, :]. 0 for training/prefill,
               cache_len for single-token decode.
        """
        _, _, T, D = x.shape
        assert D == self.head_dim, f'expected head_dim={self.head_dim}, got {D}'

        row_ids, col_ids, time_ids = self._coords(start, start + T, x.device)
        cos_r, sin_r = self._axis_cos_sin(row_ids, self.cos_row, self.sin_row)
        cos_c, sin_c = self._axis_cos_sin(col_ids, self.cos_col, self.sin_col)
        cos_t, sin_t = self._axis_cos_sin(time_ids, self.cos_time, self.sin_time)

        A = self.axis_dim
        x_row = x[..., : A]
        x_col = x[..., A : 2 * A]
        x_time = x[..., 2 * A : 3 * A]

        out_row = self._rotate(x_row, cos_r, sin_r)
        out_col = self._rotate(x_col, cos_c, sin_c)
        out_time = self._rotate(x_time, cos_t, sin_t)
        return torch.cat([out_row, out_col, out_time], dim=-1)
