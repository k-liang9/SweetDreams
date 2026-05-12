import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout(cfg.model.dropout)
        self.blocks = nn.ModuleList([Block(cfg)])
        self.layer_norm = nn.LayerNorm(cfg.model.d_model)
        
class Block(nn.Module):
    pass

class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.model.embed_dim % cfg.model.num_heads == 0
        assert cfg.model.attention in ('causal', 'block_causal')
        self.num_heads = cfg.model.num_heads
        D = self.d_model = cfg.model.d_model
        self.Wk = nn.Linear(D, D)
        self.Wq = nn.Linear(D, D)
        self.Wv = nn.Linear(D, D)
        
        self.attn_dropout = nn.Dropout(cfg.model.dropout)
        self.resid_dropout = nn.Dropout(cfg.model.dropout) # NOTE: using 1 unified dropout for simplicity
        self.proj = nn.Linear(D, D)
        
        causal_mask = torch.tril(torch.ones(cfg.max_tokens, cfg.max_tokens))
        block_causal_mask = torch.max(causal_mask, torch.block_diag(*[torch.ones(cfg.tokens_per_block, cfg.tokens_per_block) for _ in range(cfg.max_blocks)]))
        self.register_buffer('mask', causal_mask if cfg.model.attention == 'causal' else block_causal_mask)
        
    # TODO: add kv cache
    def forward(self, x):
        B, T, C = x.size()
        
        head_dim = C // self.num_heads

        Q = self.Wq(x).view(B, T, self.num_heads, head_dim).transpose(1, 2) # (B,T,C) -> (B,T,nh,hs) -> (B,nh,T,hs)
        K = self.Wk(x).view(B, T, self.num_heads, head_dim).transpose(1, 2) # (B,T,C) -> (B,T,nh,hs) -> (B,nh,T,hs)
        V = self.Wv(x).view(B, T, self.num_heads, head_dim).transpose(1, 2) # (B,T,C) -> (B,T,nh,hs) -> (B,nh,T,hs)
        
        L = 0
        
        att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(K.size(-1))) # (B,nh,T,hs) @ (B,nh,hs,T) -> (B,nh,T,T)
        att = att.masked_fill(self.mask[L:L+T, :L+T] == 0, float('-inf')) # (B,nh,T,T)
        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_dropout(att) # (B, nh, T, T)

        y = att @ V # (B,nh,T,T) @ (B,nh,T,hs) -> (B,nh,T,hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B,nh,T,hs) -> (B,T,nh,hs) -> (B,T,C)
        
        y = self.resid_dropout(self.proj(y)) # (B,T,C) -> (B,T,C)
        
        return y
