import torch.nn as nn
from torch.nn import functional as F
from transformer import Transformer
from embeddings import WorldModelEmbeddings

class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.model.d_model
        self.embeddings = WorldModelEmbeddings(cfg)
        self.transformer = Transformer(cfg)
        self.frame_head = nn.Linear(D, cfg.vq.num_tokens)
        # TODO: reward heads and done_heads
        # self.reward_head = 
        # self.done_head = 

    def forward(self, frame_tokens, actions, rewards=None, dones=None):
        x = self.embeddings(frame_tokens, actions)
        h = self.transformer(x)
        frame_logits = self.frame_head(h)
        
        return frame_logits