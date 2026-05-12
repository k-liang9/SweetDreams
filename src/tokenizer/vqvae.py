import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizer.losses import vqvae_loss

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg.model.in_out_channels
        hidden_dim = cfg.model.hidden_dim
        latent_dim = cfg.model.latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1), # 64 -> 32
            nn.LayerNorm([hidden_dim, 32, 32]),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1), # 32 -> 16
            nn.LayerNorm([hidden_dim, 16, 16]),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1), # 16 -> 8
            nn.LayerNorm([hidden_dim, 8, 8]),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, latent_dim, kernel_size=4, stride=2, padding=1), # 8 -> 4
        )
    
    def forward(self, x):
        return self.net(x) # (B, latent_dim, 4, 4)
    
class VectorQuantizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.K = cfg.model.codebook.num_embeddings
        self.D = cfg.model.latent_dim
        self.decay = cfg.model.codebook.ema.decay
        self.eps = cfg.model.codebook.ema.eps
        self.restart_threshold = cfg.model.codebook.ema.restart_threshold

        self.embedding = nn.Embedding(self.K, self.D)
        nn.init.uniform_(self.embedding.weight, -1/self.K, 1/self.K)
        self.embedding.weight.requires_grad_(False)
        self.register_buffer('ema_count', torch.empty(self.K))
        self.register_buffer('ema_sum', torch.empty(self.K, self.D))
        self.reset_ema_state()

    def normalized_embedding(self):
        return F.normalize(self.embedding.weight, p=2, dim=1)

    @torch.no_grad()
    def reset_ema_state(self):
        embedding = self.normalized_embedding().detach()
        initial_count = max(1.0, self.restart_threshold * 2)
        self.embedding.weight.copy_(embedding)
        self.ema_count.fill_(initial_count)
        self.ema_sum.copy_(embedding * initial_count)

    @torch.no_grad()
    def update_codebook(self, z_flat, indices):
        counts = torch.bincount(indices, minlength=self.K).type_as(z_flat)
        sums = torch.zeros(self.K, self.D, device=z_flat.device, dtype=z_flat.dtype)
        sums.index_add_(0, indices, z_flat)

        self.ema_count.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.ema_sum.mul_(self.decay).add_(sums, alpha=1 - self.decay)

        normalizer = self.ema_count.clamp_min(self.eps).unsqueeze(1)
        embedding = F.normalize(self.ema_sum / normalizer, p=2, dim=1)

        dead_codes = self.ema_count < self.restart_threshold
        if dead_codes.any():
            replacement_count = dead_codes.sum().item()
            replacement_indices = torch.randint(z_flat.shape[0], (replacement_count,), device=z_flat.device)
            replacements = z_flat[replacement_indices]
            embedding[dead_codes] = replacements
            restart_count = max(self.restart_threshold, z_flat.shape[0] / self.K)
            self.ema_count[dead_codes] = restart_count
            self.ema_sum[dead_codes] = replacements * restart_count

        self.embedding.weight.copy_(F.normalize(embedding, p=2, dim=1))

    def forward(self, z):
        # z: (B, D, H, W) -> rearrange to (B*H*W, D)
        z = F.normalize(z, p=2, dim=1)
        embedding = self.normalized_embedding()
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D) # (B*H*W, D)
        
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2* z_flat @ embedding.T
            + embedding.pow(2).sum(1)
        ) # (B*H*W, K)
        
        # find nearest codebook entry
        indices = distances.argmin(1) # (B*H*W,)
        z_q = embedding[indices].reshape(B, H, W, D).permute(0, 3, 1, 2) # (B, D, H, W)
        z_q_raw = z_q

        if self.training:
            self.update_codebook(z_flat.detach(), indices.detach())
        
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
            nn.LayerNorm([hidden_dim, 8, 8]),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # 8 → 16
            nn.LayerNorm([hidden_dim, 16, 16]),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # 16 → 32
            nn.LayerNorm([hidden_dim, 32, 32]),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2, padding=1), # 32 → 64
            nn.Sigmoid()  # output in [0, 1] to match normalized frames
        )

    def forward(self, z_q):
        return self.net(z_q)  # (B, 1, 64, 64)
    
class VQVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder =      Encoder(cfg)
        self.quantizer =    VectorQuantizer(cfg)
        self.decoder =      Decoder(cfg)
        self.commitment_cost = cfg.model.codebook.commitment_cost
        
    def featurize(self, batch):
        frames = batch[0] if isinstance(batch, (tuple, list)) else batch
        if frames.dim() == 5:
            B, T, C, H, W = frames.shape
            frames = frames.reshape(B * T, C, H, W)
        return frames, frames
        
    def forward(self, x, target=None):
        if target is None:
            target = x
        
        z = F.normalize(self.encoder(x), p=2, dim=1)
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
        z = F.normalize(self.encoder(x), p=2, dim=1)
        _, indices, _ = self.quantizer(z)
        return indices # (B, 4, 4)
    
    def decode_from_indices(self, indices):
        z_q = self.quantizer.normalized_embedding()[indices].permute(0, 3, 1, 2)
        return self.decoder(z_q)
