import torch
import torch.nn as nn

class WorldModelEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.model.d_model
        self.frame_token_embedding = nn.Embedding(cfg.vq.num_tokens, self.d_model)
        self.action_embedding = nn.Embedding(cfg.data.num_actions, self.d_model)
        self.dropout = nn.Dropout(cfg.model.dropout)
    
    def forward(self, frame_tokens, actions):
        """
        frame_tokens: (B, T, N)
        actions:      (B, T - 1)

        returns:
            x: (B, S, d_model)
        """
        B, T, N = frame_tokens.shape
        
        frame_x = self.frame_token_embedding(frame_tokens)  # (B,T,N,d)
        action_x = self.action_embedding(actions)           # (B,T-1,d)
        
        chunks = []
        for t in range(T):
            chunks.append(frame_x[:, t])                    # (B,N,d)
            if t < T-1:
                chunks.append(action_x[:, t])               # (B,1,d)
        
        x = torch.cat(chunks, dim=1)                        # (B,S,d)
        return self.dropout(x)