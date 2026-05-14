import torch
import torch.nn as nn

class WorldModelEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.model.d_model
        self.max_seq_len = cfg.model.max_seq_len
        self.frame_token_embedding = nn.Embedding(cfg.model.num_frame_tokens, self.d_model)
        self.action_embedding = nn.Embedding(cfg.model.num_actions, self.d_model)
        self.position_embedding = nn.Embedding(cfg.model.max_seq_len, self.d_model)
        self.dropout = nn.Dropout(cfg.model.dropout)
    
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
        if x.size(1) > self.max_seq_len:
            raise ValueError(
                f'Interleaved sequence length {x.size(1)} exceeds max_seq_len {self.max_seq_len}'
            )
        
        positions = torch.arange(x.size(1), device=x.device)
        pos_x = self.position_embedding(positions).unsqueeze(0) # (1,S,d)
        
        x = x + pos_x
        return self.dropout(x)
