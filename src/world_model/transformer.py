"""
refer to https://github.com/eloialonso/iris/blob/main/src/models/transformer.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from world_model.embeddings import compute_max_seq_len, RoPE

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.model.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.model.d_model)

    def forward(self, sequences):
        x = sequences
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return x

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = self.d_model = cfg.model.d_model
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.attn = SelfAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(D, 4*D),
            nn.GELU(),
            nn.Linear(4*D, D),
            nn.Dropout(cfg.model.dropout),
        )

    # TODO: add kv
    def forward(self, x):
        x_attn = self.attn(self.ln1(x))
        x = x + x_attn
        x = x + self.mlp(self.ln2(x))
        return x

class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.model.d_model % cfg.model.num_heads == 0
        self.cfg = cfg
        attention = cfg.model.get('attention', 'causal')
        assert attention in ('causal', 'block_causal')
        self.attention_mode = attention
        self.num_heads = cfg.model.num_heads
        D = self.d_model = cfg.model.d_model
        self.head_dim = D // self.num_heads
        self.Wk = nn.Linear(D, D)
        self.Wq = nn.Linear(D, D)
        self.Wv = nn.Linear(D, D)
        self.rope = RoPE(cfg)

        self.resid_dropout = nn.Dropout(cfg.model.dropout)
        self.proj = nn.Linear(D, D)

        if attention == 'block_causal':
            max_seq_len = compute_max_seq_len(cfg)
            tokens_per_block = cfg.model.get('tokens_per_block', cfg.model.tokens_per_frame + 1)
            mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
            for start in range(0, max_seq_len, tokens_per_block):
                end = min(start + tokens_per_block, max_seq_len)
                mask[start:end, start:end] = True
            self.register_buffer('mask', mask)

    # TODO: add kv cache
    def forward(self, x):
        B, T, C = x.size()

        Q = self.Wq(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B,T,C) -> (B,T,nh,hs) -> (B,nh,T,hs)
        K = self.Wk(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q = self.rope(Q)
        K = self.rope(K)

        dropout_p = self.cfg.model.dropout if self.training else 0.0
        if self.attention_mode == 'causal':
            y = F.scaled_dot_product_attention(Q, K, V, is_causal=True, dropout_p=dropout_p)
        else:
            mask = self.mask[:T, :T].unsqueeze(0).unsqueeze(0)
            y = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=dropout_p)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B,nh,T,hs) -> (B,T,nh,hs) -> (B,T,C)

        y = self.resid_dropout(self.proj(y))
        return y
