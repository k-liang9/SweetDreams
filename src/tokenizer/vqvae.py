import torch
import torch.nn as nn
from torch.nn import functional as F
from common.base import Model as Base

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg.params.in_out_channels
        hidden_dim = cfg.params.hidden_dim=128
        latent_dim = cfg.params.latent_dim=256
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1), # 64 -> 32
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1), # 32 -> 16
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1), # 16 -> 8
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, latent_dim, 4, stride=2, padding=1), # 8 -> 4
        )
    
    def forward(self, x):
        return self.net(x) # (B, latent_dim, 4, 4)
    
class VectorQuantizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.K = cfg.params.num_embeddings
        self.D = cfg.params.embedding_dim
        self.commitment_cost = cfg.params.commitment_cost
        
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
        indices = distances.argmin(1) (B*H*W,)
        z_q = self.embedding(indices).reshape(B, H, W, D).permute(0, 3, 1, 2)
        
        # losses
        codebook_loss = (z_q.detach() - z).pow(2).mean() # moves codebook toward encoder
        commitment_loss = (z_q - z.detach()).pow(2).mean() # moves encoder toward codebook
        loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # straight-through estimator: lets gradient flow thru nondifferentiable argmin
        z_q = z + (z_q - z).detach()
        
        return z_q, indices.reshape(B, H, W), loss

class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        out_channels = cfg.params.in_out_channels
        hidden_dim = cfg.params.hidden_dim=128
        latent_dim = cfg.params.latent_dim=256
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim, 4, stride=2, padding=1),  # 4 → 8
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),  # 8 → 16
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),  # 16 → 32
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, out_channels, 4, stride=2, padding=1), # 32 → 64
            nn.Sigmoid()  # output in [0, 1] to match normalized frames
        )

    def forward(self, z_q):
        return self.net(z_q)  # (B, 1, 64, 64)
    
class VQVAE(Base):
    def __init__(self, cfg):
        self.encoder =      Encoder(cfg)
        self.quantizer =    VectorQuantizer(cfg)
        self.decoder =      Decoder(cfg)
        
    def compute_and_log_metrics(self, pred, feat):
        # train/recon_loss, train/vq_loss, train/commitment_loss, train/total_loss, train/codebook_util, train/codebook_perplexity
        # val: recon_loss, codebook_utilization
        # test: recon_loss, codebook_perplexity, codebook_utilization, reconstruction image grid, Frechet Inception Distance (FID)
        pass
        
    def forward(self, feat):
        z = self.encoder(feat)
        z_q, indices, vq_loss = self.quantizer(z)
        pred = self.decoder(z_q)
        recon_loss = F.mse_loss(pred, feat)
        return {
            'pred': pred,
            'indices': indices,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'loss': recon_loss + vq_loss,
        }
    
    def encode(self, x):
        # at inference - only token indices
        z = self.encoder(x)
        _, indices, _ = self.quantizer(z)
        return indices # (B, 4, 4)
    
    def decode_from_indices(self, indices):
        z_q = self.quantizer.embedding(indices).permute(0, 3, 1, 2)
        return self.decoder(z_q)