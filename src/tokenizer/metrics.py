import torch
import wandb


def codebook_metrics(indices, num_embeddings):
    counts = torch.bincount(indices.reshape(-1), minlength=num_embeddings).float()
    probs = counts / counts.sum().clamp_min(1)
    used = counts.gt(0).float().sum()
    nonzero_probs = probs[probs.gt(0)]
    perplexity = torch.exp(-(nonzero_probs * nonzero_probs.log()).sum())
    return used / num_embeddings, perplexity


def reconstruction_grid(pred, target, n=8):
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


def vqvae_metrics(split, out, target, num_embeddings, include_reconstructions=False):
    codebook_utilization, codebook_perplexity = codebook_metrics(out['indices'], num_embeddings)
    metrics = {
        f'{split}/loss': out['loss'],
        f'{split}/recon_loss': out['recon_loss'],
        f'{split}/vq_loss': out['vq_loss'],
        f'{split}/codebook_loss': out['codebook_loss'],
        f'{split}/commitment_loss': out['commitment_loss'],
        f'{split}/codebook_utilization': codebook_utilization,
        f'{split}/codebook_perplexity': codebook_perplexity,
    }

    if include_reconstructions:
        image_grid = reconstruction_grid(out['pred'], target)
        if image_grid is not None:
            metrics[f'{split}/reconstructions'] = image_grid

    return metrics
