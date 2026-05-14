"""
refer to https://github.com/eloialonso/iris/blob/main/src/models/transformer.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout(cfg.model.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.model.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.model.d_model)
    
    def forward(self, sequences):
        x = self.dropout(sequences)
        for i, block in enumerate(self.blocks):
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
        attention = cfg.model.get('attention', 'causal')
        assert attention in ('causal', 'block_causal')
        self.num_heads = cfg.model.num_heads
        D = self.d_model = cfg.model.d_model
        self.head_dim = D // self.num_heads
        self.Wk = nn.Linear(D, D)
        self.Wq = nn.Linear(D, D)
        self.Wv = nn.Linear(D, D)
        
        self.attn_dropout = nn.Dropout(cfg.model.dropout)
        self.resid_dropout = nn.Dropout(cfg.model.dropout) # NOTE: using 1 unified dropout for simplicity
        self.proj = nn.Linear(D, D)
        
        max_seq_len = cfg.model.max_seq_len
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        if attention == 'block_causal':
            tokens_per_block = cfg.model.get('tokens_per_block', cfg.model.tokens_per_frame + 1)
            block_causal_mask = causal_mask.clone()
            for start in range(0, max_seq_len, tokens_per_block):
                end = min(start + tokens_per_block, max_seq_len)
                block_causal_mask[start:end, start:end] = True
            causal_mask = block_causal_mask
        self.register_buffer('mask', causal_mask)
        
    # TODO: add kv cache
    def forward(self, x):
        B, T, C = x.size()
        if T > self.mask.size(0):
            raise ValueError(f'Sequence length {T} exceeds attention mask size {self.mask.size(0)}')

        Q = self.Wq(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B,T,C) -> (B,T,nh,hs) -> (B,nh,T,hs)
        K = self.Wk(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B,T,C) -> (B,T,nh,hs) -> (B,nh,T,hs)
        V = self.Wv(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B,T,C) -> (B,T,nh,hs) -> (B,nh,T,hs)
        
        L = 0
        
        att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(K.size(-1))) # (B,nh,T,hs) @ (B,nh,hs,T) -> (B,nh,T,T)
        mask = self.mask[L:L+T, :L+T].unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(~mask, float('-inf')) # (B,nh,T,T)
        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_dropout(att) # (B, nh, T, T)

        y = att @ V # (B,nh,T,T) @ (B,nh,T,hs) -> (B,nh,T,hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B,nh,T,hs) -> (B,T,nh,hs) -> (B,T,C)
        
        y = self.resid_dropout(self.proj(y)) # (B,T,C) -> (B,T,C)
        
        return y
