"""Persistent per-port VAE codes: a second memory substrate beside Titans.

Every valid observed feature gets one posterior code. There is no pooling,
token pruning, or automatic eviction here. Callers seal/offload complete units
to bound residency. Encoder/decoder weights are shared with dream replay.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LatentPortMemory:
    mu: torch.Tensor
    logvar: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor

    def detach(self):
        return LatentPortMemory(*(v.detach() for v in self.tensors()))

    def tensors(self):return (self.mu,self.logvar,self.mean,self.scale)

    @property
    def count(self):return self.mu.shape[0]


def encode_observations(codec,features,previous=None,*,create_graph=False):
    # Older states remain immutable and retain their values, while training
    # differentiates through the new write, not an unbounded source history.
    result={name:port.detach() for name,port in (previous or {}).items()}
    with torch.set_grad_enabled(create_graph):
        for name,features in features.items():
            # Float inputs alone do not prevent autocast from producing BF16
            # posterior codes. The next native write may run outside AMP.
            with torch.autocast(device_type=features.device.type,enabled=False):
                x=features.detach().float().reshape(-1,features.shape[-1])
                mean=x.mean(-1,keepdim=True);scale=(x.var(-1,unbiased=False,keepdim=True)+1e-5).sqrt()
                head=codec.heads[name]
                mu,logvar=head.posterior(head.encoder((x-mean)/scale)).chunk(2,-1)
                new=LatentPortMemory(mu,logvar.clamp(-12,8),mean,scale)
            if name in result:
                new=LatentPortMemory(*(torch.cat((a,b),dim=0) for a,b in zip(result[name].tensors(),new.tensors())))
            result[name]=new if create_graph else new.detach()
    return result

