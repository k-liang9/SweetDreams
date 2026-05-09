import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
from common.base import Model as Base

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg.params.in_out_channels
        hidden_dim = cfg.params.hidden_dim
        latent_dim = cfg.params.latent_dim
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
        indices = distances.argmin(1) # (B*H*W,)
        z_q = self.embedding(indices).reshape(B, H, W, D).permute(0, 3, 1, 2) # (B, D, H, W)
        
        # losses
        codebook_loss = (z_q - z.detach()).pow(2).mean() # moves codebook toward encoder
        commitment_loss = (z - z_q.detach()).pow(2).mean() # moves encoder toward codebook
        loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # straight-through estimator: lets gradient flow thru nondifferentiable argmin
        z_q = z + (z_q - z).detach()
        
        return z_q, indices.reshape(B, H, W), loss, codebook_loss, commitment_loss

class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        out_channels = cfg.params.in_out_channels
        hidden_dim = cfg.params.hidden_dim
        latent_dim = cfg.params.latent_dim
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
        z_q, indices, vq_loss, codebook_loss, commitment_loss = self.quantizer(z)
        pred = self.decoder(z_q)
        recon_loss = F.mse_loss(pred, target)
        return {
            'pred': pred,
            'indices': indices,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss,
            'loss': recon_loss + vq_loss,
        }
    
    def encode(self, x):
        # at inference - only token indices
        z = self.encoder(x)
        _, indices, _, _, _ = self.quantizer(z)
        return indices # (B, 4, 4)
    
    def decode_from_indices(self, indices):
        z_q = self.quantizer.embedding(indices).permute(0, 3, 1, 2)
        return self.decoder(z_q)
    
    def _codebook_metrics(self, indices):
        counts = torch.bincount(indices.reshape(-1), minlength=self.quantizer.K).float()
        probs = counts / counts.sum().clamp_min(1)
        used = counts.gt(0).float().sum()
        nonzero_probs = probs[probs.gt(0)]
        perplexity = torch.exp(-(nonzero_probs * nonzero_probs.log()).sum())
        return used / self.quantizer.K, perplexity
    
    def _reconstruction_grid(self, pred, target, n=8):
        n = min(n, pred.shape[0], target.shape[0])
        originals = target[:n].detach().cpu().clamp(0, 1)
        reconstructions = pred[:n].detach().cpu().clamp(0, 1)
        grid = torch.cat([originals, reconstructions], dim=0)
        C, H, W = grid.shape[1:]
        grid = grid.reshape(2, n, C, H, W).permute(0, 3, 1, 4, 2)
        grid = grid.reshape(2 * H, n * W, C)
        if C == 1:
            grid = grid.squeeze(-1)
        try:
            return wandb.Image(grid.numpy())
        except wandb.Error:
            return None
    
    def compute_metrics(self, split, out, x, target):
        codebook_utilization, codebook_perplexity = self._codebook_metrics(out['indices'])
        metrics = {
            f'{split}/loss': out['loss'],
            f'{split}/recon_loss': out['recon_loss'],
            f'{split}/vq_loss': out['vq_loss'],
            f'{split}/codebook_utilization': codebook_utilization,
            f'{split}/codebook_perplexity': codebook_perplexity,
        }
        
        if 'codebook_loss' in out:
            metrics[f'{split}/codebook_loss'] = out['codebook_loss']
        if 'commitment_loss' in out:
            metrics[f'{split}/commitment_loss'] = out['commitment_loss']
        if split == 'test':
            reconstruction_grid = self._reconstruction_grid(out['pred'], target)
            if reconstruction_grid is not None:
                metrics[f'{split}/reconstructions'] = reconstruction_grid
        
        return metrics
