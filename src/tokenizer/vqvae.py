import torch
import torch.nn as nn
from common.base import Model as Base
from tokenizer.losses import vqvae_loss
from tokenizer.metrics import vqvae_metrics

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg.model.in_out_channels
        hidden_dim = cfg.model.hidden_dim
        latent_dim = cfg.model.latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1), # 64 -> 32
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1), # 32 -> 16
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1), # 16 -> 8
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, latent_dim, kernel_size=4, stride=2, padding=1), # 8 -> 4
        )
    
    def forward(self, x):
        return self.net(x) # (B, latent_dim, 4, 4)
    
class VectorQuantizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.K = cfg.model.num_embeddings
        self.D = cfg.model.embedding_dim
        
        self.embedding = nn.Embedding(self.K, self.D)
        nn.init.uniform_(self.embedding.weight, -1/self.K, 1/self.K)
        
    def forward(self, z):
        # z: (B, D, H, W) -> rearrange to (B*H*W, D)
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D) # (B*H*W, D)
        
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2* z_flat @ self.embedding.weight.T
            + self.embedding.weight.pow(2).sum(1)
        ) # (B*H*W, K)
        
        # find nearest codebook entry
        indices = distances.argmin(1) # (B*H*W,)
        z_q = self.embedding(indices).reshape(B, H, W, D).permute(0, 3, 1, 2) # (B, D, H, W)
        z_q_raw = z_q
        
        # straight-through estimator: lets gradient flow thru nondifferentiable argmin
        z_q = z + (z_q - z).detach()
        
        return z_q, indices.reshape(B, H, W), z_q_raw

class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        out_channels = cfg.model.in_out_channels
        hidden_dim = cfg.model.hidden_dim
        latent_dim = cfg.model.latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # 4 → 8
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # 8 → 16
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # 16 → 32
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2, padding=1), # 32 → 64
            nn.Sigmoid()  # output in [0, 1] to match normalized frames
        )

    def forward(self, z_q):
        return self.net(z_q)  # (B, 1, 64, 64)
    
class VQVAE(Base):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.encoder =      Encoder(cfg)
        self.quantizer =    VectorQuantizer(cfg)
        self.decoder =      Decoder(cfg)
        self.commitment_cost = cfg.model.commitment_cost
        
    def featurize(self, batch):
        frames = batch[0] if isinstance(batch, (tuple, list)) else batch
        if frames.dim() == 5:
            B, T, C, H, W = frames.shape
            frames = frames.reshape(B * T, C, H, W)
        return frames, frames
        
    def forward(self, x, target=None):
        if target is None:
            target = x
        
        z = self.encoder(x)
        z_q, indices, z_q_raw = self.quantizer(z)
        pred = self.decoder(z_q)
        loss_dict = vqvae_loss(pred, target, z, z_q_raw, self.commitment_cost)
        return {
            'pred': pred,
            'indices': indices,
            **loss_dict,
        }
    
    def encode(self, x):
        # at inference - only token indices
        z = self.encoder(x)
        _, indices, _ = self.quantizer(z)
        return indices # (B, 4, 4)
    
    def decode_from_indices(self, indices):
        z_q = self.quantizer.embedding(indices).permute(0, 3, 1, 2)
        return self.decoder(z_q)
    
    def compute_metrics(self, split, out, x, target):
        return vqvae_metrics(
            split,
            out,
            target,
            num_embeddings=self.quantizer.K,
            include_reconstructions=split == 'test',
        )
