import torch
import torch.distributed as dist
import wandb


def codebook_metrics(indices, num_embeddings):
    counts = torch.bincount(indices.reshape(-1), minlength=num_embeddings).float()
    if dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    probs = counts / counts.sum().clamp_min(1)
    used = counts.gt(0).float().sum()
    nonzero_probs = probs[probs.gt(0)]
    perplexity = torch.exp(-(nonzero_probs * nonzero_probs.log()).sum())
    return used / num_embeddings, perplexity


def reconstruction_grid(pred, target, n=1):
    n = min(n, pred.shape[0], target.shape[0])
    indices = torch.randperm(pred.shape[0])[:n]
    originals = target[indices].detach().cpu().clamp(0, 1)
    reconstructions = pred[indices].detach().cpu().clamp(0, 1)
    grid = torch.cat([originals, reconstructions], dim=0)
    C, H, W = grid.shape[1:]
    grid = grid.reshape(2, n, C, H, W).permute(0, 3, 1, 4, 2)
    grid = grid.reshape(2 * H, n * W, C)
    if C == 1:
        grid = grid.squeeze(-1)

    if not hasattr(wandb, 'Image'):
        return None
    return wandb.Image(grid.numpy())


def generator_metrics(split, g_loss, weight):
    return {
        f'{split}/g_loss': g_loss.detach(),
        f'{split}/disc_weight': weight.detach(),
    }


def discriminator_metrics(split, d_loss, real_logits, fake_logits, r1_penalty=None):
    metrics = {
        f'{split}/d_loss': d_loss.detach(),
        f'{split}/d_real': real_logits.detach().mean(),
        f'{split}/d_fake': fake_logits.detach().mean(),
    }
    if r1_penalty is not None:
        metrics[f'{split}/r1_penalty'] = r1_penalty.detach()
    return metrics


def vqvae_metrics(
    split,
    out,
    target,
    num_embeddings,
    include_reconstructions=False,
    reconstruction_examples=1,
):
    codebook_utilization, codebook_perplexity = codebook_metrics(out['indices'], num_embeddings)
    if split in ('train', 'val'):
        metrics = {
            f'{split}/loss': out['loss'],
            f'{split}/recon_loss': out['recon_loss'],
            f'{split}/perceptual_loss': out['perceptual_loss'],
            f'{split}/vq_loss': out['vq_loss'],
            f'{split}/commitment_loss': out['commitment_loss'],
            f'{split}/codebook_perplexity': codebook_perplexity,
            f'{split}/codebook_utilization': codebook_utilization,
        }
    else:
        metrics = {
            f'{split}/recon_loss': out['recon_loss'],
            f'{split}/perceptual_loss': out['perceptual_loss'],
            f'{split}/codebook_perplexity': codebook_perplexity,
            f'{split}/codebook_utilization': codebook_utilization,
        }

    if include_reconstructions:
        image_grid = reconstruction_grid(out['pred'], target, n=reconstruction_examples)
        if image_grid is not None:
            metrics[f'{split}/reconstructions'] = image_grid

    return metrics
