import torch.nn as nn

from world_model.embeddings import WorldModelEmbeddings
from world_model.transformer import Transformer


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.model.d_model
        self.embeddings = WorldModelEmbeddings(cfg)
        self.transformer = Transformer(cfg)
        self.frame_head = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, cfg.model.num_frame_tokens),
        )
        # TODO: reward heads and done_heads
        # self.reward_head =
        # self.done_head =

        self.apply(init_weights)

    def forward(self, frame_tokens, actions, rewards=None, dones=None, start=0, cache=None):
        x = self.embeddings(frame_tokens, actions)
        h = self.transformer(x, start=start, cache=cache)
        frame_logits = self.frame_head(h)

        return {'frame_logits': frame_logits}

    def forward_token(self, token, position, cache):
        """Single-token cached decode. position = absolute position of this token."""
        x = self.embeddings.embed_token(token, position)
        h = self.transformer(x, start=position, cache=cache)
        return self.frame_head(h[:, 0])  # (B, num_frame_tokens)
