"""Shared weight and feature VAE heads for the unified physical memory unit.

Physical units own serialization and retention-gated Dream publication.
No separate read-only page format or alternate recall implementation lives here.
"""
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().contiguous().cpu()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class DreamWeightVAE(nn.Module):
    """One shared, small codec; parameter count does not grow with page count.

    Chunks share the encoder/decoder, without a dense whole-page bottleneck.
    Posterior sampling is for training; recall always decodes the stored mean.
    FP32 codes are counted as FP32 bytes, not hypothetical entropy-coded bits.
    """
    def __init__(self, chunk_size=1024, hidden_size=128, latent_size=32):
        super().__init__()
        if not (0 < latent_size <= hidden_size < chunk_size):
            raise ValueError('expected 0 < latent <= hidden < chunk_size')
        self.spec = dict(chunk_size=chunk_size, hidden_size=hidden_size, latent_size=latent_size)
        self.encoder = nn.Sequential(nn.Linear(chunk_size, hidden_size), nn.SiLU())
        self.posterior = nn.Linear(hidden_size, 2 * latent_size)
        self.decoder = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.SiLU(),
                                     nn.Linear(hidden_size, chunk_size))

    def chunks(self, delta):
        if delta.dtype != torch.float32 or not torch.isfinite(delta).all():
            raise ValueError('finite FP32 deltas required')
        size = self.spec['chunk_size']
        count = math.ceil(delta.numel() / size)
        flat = F.pad(delta.reshape(-1), (0, count * size - delta.numel())).reshape(count, size)
        mask = (torch.arange(count * size, device=delta.device) < delta.numel()).reshape(count, size)
        scales = (flat.square().sum(1, keepdim=True) / mask.sum(1, keepdim=True)).sqrt().clamp_min(1e-8)
        return flat / scales, scales, mask

    def forward(self, chunks, *, sample=True):
        mean, logvar = self.posterior(self.encoder(chunks)).chunk(2, dim=-1)
        logvar = logvar.clamp(-12, 8)
        z = mean + torch.randn_like(mean) * (logvar * .5).exp() if sample else mean
        return self.decoder(z), mean, logvar

    def reconstruct(self, delta, *, sample=True):
        x, scales, mask = self.chunks(delta)
        decoded, mean, logvar = self(x, sample=sample)
        distortion = ((decoded - x).square() * mask).sum() / mask.sum()
        kl = .5 * (mean.square() + logvar.exp() - 1 - logvar).mean()
        reconstruction = (decoded * scales).reshape(-1)[:delta.numel()].reshape_as(delta)
        return reconstruction, dict(distortion=distortion, kl=kl)

    @property
    def codec_id(self):
        return hashlib.sha256((json.dumps(self.spec, sort_keys=True) +
                               tensor_digest(self.state_dict())).encode()).hexdigest()


class PortFeatureVAE(nn.Module):
    """Independent per-port compression and decoding heads shared across units.

    In the dual-memory architecture these same heads encode persistent latent
    memory during ordinary writes and decode it during ordinary recall. Dreams
    additionally sample source-conditioned posteriors. Native multiport feature
    dimensions stay independent; there is no shared cross-port bottleneck.
    """
    def __init__(self, port_dims, hidden_size=64, latent_size=16):
        super().__init__()
        self.spec = dict(port_dims=dict(port_dims), hidden_size=hidden_size, latent_size=latent_size)
        self.heads = nn.ModuleDict({name: DreamWeightVAE(dim, hidden_size, latent_size)
                                   for name, dim in port_dims.items()})

    def loss(self, name, features, beta=.001):
        normalized = F.layer_norm(features.detach().float(), (features.shape[-1],))
        decoded, mean, logvar = self.heads[name](normalized, sample=True)
        distortion = F.mse_loss(decoded, normalized)
        kl = .5 * (mean.square() + logvar.exp() - 1 - logvar).mean()
        return distortion + beta * kl

    @torch.no_grad()
    def source_replay(self, name, observed_features, *, source_page_id, count=4,
                      temperature=.3, min_cosine=.8):
        if self.training or not source_page_id:
            raise ValueError('frozen feature generator and explicit source identity required')
        if count < 1 or not 0 <= temperature <= 1 or not -1 <= min_cosine <= 1:
            raise ValueError('invalid source replay parameters')
        x = F.layer_norm(observed_features.detach().float(), (observed_features.shape[-1],))
        selected = x[torch.randint(len(x), (count,), device=x.device)]
        head = self.heads[name]
        _, mean, logvar = head(selected, sample=False)
        generated = head.decoder(mean + temperature * torch.randn_like(mean) * (.5*logvar).exp())
        similarity = F.cosine_similarity(generated, selected, dim=-1)
        accepted = torch.isfinite(generated).all(dim=-1) & (similarity >= min_cosine)
        # Rejected samples are counted and omitted, never silently replaced.
        return generated[accepted], dict(source_page_id=source_page_id, port=name,
            proposed=count, accepted=int(accepted.sum()), rejected=int((~accepted).sum()),
            min_cosine=min_cosine, temperature=temperature,
            guarantee='local feature similarity only; not proof of a historical fact')
