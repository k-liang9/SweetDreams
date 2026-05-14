from torch.nn import functional as F


def reconstruction_loss(pred, target):
    return F.l1_loss(pred, target)
    # TODO: ball-specific isolation
    weight = 1.0 + 10.0 * (target > 0.5).float() # emphasize bright spots (target: pong ball)
    weight = weight / weight.mean()
    recon_loss = (weight * (pred - target).pow(2)).mean()
    return recon_loss

def vector_quantization_loss(z, z_q, commitment_cost):
    codebook_loss = F.mse_loss(z_q, z.detach())
    commitment_loss = F.mse_loss(z, z_q.detach())
    vq_loss = commitment_cost * commitment_loss
    return {
        'vq_loss': vq_loss,
        'codebook_loss': codebook_loss,
        'commitment_loss': commitment_loss,
    }


def vqvae_loss(pred, target, z, z_q, commitment_cost):
    recon_loss = reconstruction_loss(pred, target)
    loss_dict = vector_quantization_loss(z, z_q, commitment_cost)
    return {
        'loss': recon_loss + loss_dict['vq_loss'],
        'recon_loss': recon_loss,
        **loss_dict,
    }
