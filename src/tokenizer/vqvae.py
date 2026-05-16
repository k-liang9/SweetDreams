import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F
from tokenizer.losses import PerceptualLoss, vqvae_loss

class SpatialSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0.0):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        x = x.reshape(B, C, -1).transpose(1, 2) # (B, H*W, C)
        
        x, _ = self.attn(x, x, x, need_weights=False)
        
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return residual + x
    

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
            SpatialSelfAttention(hidden_dim, num_heads=4),
            
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1), # 16 -> 8
            nn.LayerNorm([hidden_dim, 8, 8]),
            nn.ReLU(),
            SpatialSelfAttention(hidden_dim, num_heads=4),
            
            nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=0), # 8 -> 6
        )

    def forward(self, x):
        return self.net(x) # (B, latent_dim, 6, 6)
    
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

        if dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)

        self.ema_count.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.ema_sum.mul_(self.decay).add_(sums, alpha=1 - self.decay)

        normalizer = self.ema_count.clamp_min(self.eps).unsqueeze(1)
        embedding = F.normalize(self.ema_sum / normalizer, p=2, dim=1)

        dead_codes = self.ema_count < self.restart_threshold
        if dead_codes.any():
            replacement_count = dead_codes.sum().item()
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            if dist.is_initialized() and dist.get_rank() != 0:
                replacements = torch.empty(replacement_count, self.D, device=z_flat.device, dtype=z_flat.dtype)
            else:
                replacement_indices = torch.randint(z_flat.shape[0], (replacement_count,), device=z_flat.device)
                replacements = z_flat[replacement_indices].contiguous()
            if dist.is_initialized():
                dist.broadcast(replacements, src=0)
            embedding[dead_codes] = replacements
            restart_count = max(self.restart_threshold, world_size * z_flat.shape[0] / self.K)
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
            nn.Upsample(size=(8, 8), mode='nearest'),
            nn.Conv2d(latent_dim, hidden_dim, kernel_size=3, stride=1, padding=1),  # 6 → 8
            nn.LayerNorm([hidden_dim, 8, 8]),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),  # 8 → 8
            nn.LayerNorm([hidden_dim, 8, 8]),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),  # 8 → 16
            nn.LayerNorm([hidden_dim, 16, 16]),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),  # 16 → 32
            nn.LayerNorm([hidden_dim, 32, 32]),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, stride=1, padding=1),  # 32 → 64
            nn.Sigmoid()  # output in [0, 1] to match normalized frames
        )

    def forward(self, z_q):
        return self.net(z_q)  # (B, 3, 64, 64)
    
class VQVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder =      Encoder(cfg)
        self.quantizer =    VectorQuantizer(cfg)
        self.decoder =      Decoder(cfg)
        self.commitment_cost = cfg.model.codebook.commitment_cost
        self.perceptual_weight = float(cfg.loss.perceptual_weight)
        if self.perceptual_weight > 0:
            self.perceptual = PerceptualLoss(net=cfg.loss.perceptual_net)
        else:
            self.perceptual = None

    def flatten_frames(self, frames):
        if frames.dim() == 5:
            B, T, C, H, W = frames.shape
            return frames.reshape(B * T, C, H, W)
        if frames.dim() != 4:
            raise ValueError(f'Expected frames with shape (B, C, H, W) or (B, T, C, H, W), got {tuple(frames.shape)}')
        return frames
        
    def forward(self, frames):
        frames = self.flatten_frames(frames)
        z = F.normalize(self.encoder(frames), p=2, dim=1)
        z_q, indices, z_q_raw = self.quantizer(z)
        pred = self.decoder(z_q)
        loss_dict = vqvae_loss(
            pred, frames, z, z_q_raw, self.commitment_cost,
            perceptual=self.perceptual, perceptual_weight=self.perceptual_weight,
        )
        return {
            'pred': pred,
            'indices': indices,
            **loss_dict,
        }
    
    def encode(self, x):
        leading_shape = None
        if x.dim() == 5:
            leading_shape = x.shape[:2]
        x = self.flatten_frames(x)
        z = F.normalize(self.encoder(x), p=2, dim=1)
        _, indices, _ = self.quantizer(z)
        if leading_shape is not None:
            indices = indices.reshape(*leading_shape, *indices.shape[-2:])
        return indices # (B, 6, 6) or (B, T, 6, 6)
    
    def decode_from_indices(self, indices):
        leading_shape = None
        if indices.dim() == 4:
            leading_shape = indices.shape[:2]
            indices = indices.reshape(-1, *indices.shape[-2:])
        if indices.dim() != 3:
            raise ValueError(f'Expected indices with shape (B, H, W) or (B, T, H, W), got {tuple(indices.shape)}')
        z_q = self.quantizer.normalized_embedding()[indices].permute(0, 3, 1, 2)
        frames = self.decoder(z_q)
        if leading_shape is not None:
            frames = frames.reshape(*leading_shape, *frames.shape[1:])
        return frames
