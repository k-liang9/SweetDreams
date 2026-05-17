import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn import functional as F


class PerceptualLoss(nn.Module):
    def __init__(self, net='vgg'):
        super().__init__()
        import lpips
        self.net = lpips.LPIPS(net=net, verbose=False)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()

    def train(self, mode=True):
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, pred, target):
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        return self.net(pred * 2 - 1, target * 2 - 1).mean()


def reconstruction_loss(pred, target):
    return F.l1_loss(pred, target)


def compute_ball_mask(frames_seq, threshold=0.05, paddle_y_frac=0.85):
    """Frame-differencing ball region mask.

    frames_seq: (B, T, C, H, W) sequence in temporal order.
    Returns: (B, T, 1, H, W) binary mask aligned with frames_seq.
    The first frame in each sequence gets a zero mask (no prior frame).
    """
    diff = (frames_seq[:, 1:] - frames_seq[:, :-1]).abs().sum(dim=2, keepdim=True)
    B, _, _, H, W = frames_seq.shape
    zero = torch.zeros(B, 1, 1, H, W, device=frames_seq.device, dtype=frames_seq.dtype)
    diff = torch.cat([zero, diff], dim=1)
    mask = (diff > threshold).float()
    paddle_y = int(paddle_y_frac * H)
    mask[..., paddle_y:, :] = 0
    return mask


def ball_aux_loss(pred, target, ball_mask):
    """L1 reconstruction averaged over masked (ball-region) pixels only."""
    return ((pred - target).abs() * ball_mask).sum() / ball_mask.sum().clamp_min(1)

def vector_quantization_loss(z, z_q, commitment_cost):
    codebook_loss = F.mse_loss(z_q, z.detach())
    commitment_loss = F.mse_loss(z, z_q.detach())
    vq_loss = commitment_cost * commitment_loss
    return {
        'vq_loss': vq_loss,
        'codebook_loss': codebook_loss,
        'commitment_loss': commitment_loss,
    }


def vqvae_loss(
    pred, target, z, z_q, commitment_cost,
    perceptual=None, perceptual_weight=0.0,
    ball_mask=None, ball_weight=0.0,
):
    recon_loss = reconstruction_loss(pred, target)
    loss_dict = vector_quantization_loss(z, z_q, commitment_cost)
    total = recon_loss + loss_dict['vq_loss']

    if perceptual is not None and perceptual_weight > 0:
        perceptual_loss = perceptual(pred, target)
        total = total + perceptual_weight * perceptual_loss
    else:
        perceptual_loss = torch.zeros((), device=pred.device)

    if ball_mask is not None and ball_weight > 0:
        ball_loss = ball_aux_loss(pred, target, ball_mask)
        total = total + ball_weight * ball_loss
    else:
        ball_loss = torch.zeros((), device=pred.device)

    return {
        'loss': total,
        'recon_loss': recon_loss,
        'perceptual_loss': perceptual_loss,
        'ball_loss': ball_loss,
        **loss_dict,
    }


def discriminator_hinge_loss(real_logits, fake_logits):
    real_loss = F.relu(1.0 - real_logits).mean()
    fake_loss = F.relu(1.0 + fake_logits).mean()
    return 0.5 * (real_loss + fake_loss)


def generator_hinge_loss(fake_logits):
    return -fake_logits.mean()


def adaptive_disc_weight(nll_loss, g_loss, last_layer, max_weight=1e4):
    """VQ-GAN adaptive weight: ||grad(nll)|| / ||grad(g_loss)|| at decoder's last layer."""
    nll_grads = autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    weight = nll_grads.norm() / (g_grads.norm() + 1e-4)
    return weight.clamp(0.0, max_weight).detach()


def r1_gradient_penalty(real_logits, real_images):
    """R1 regularization: penalize discriminator gradient norm on real images.
    Caller must set real_images.requires_grad_(True) before computing real_logits."""
    grad_real = autograd.grad(
        real_logits.sum(), real_images, create_graph=True
    )[0]
    return grad_real.pow(2).flatten(1).sum(1).mean()
