"""Checkpoint-owned priors and write rates for caller-owned physical FFN state."""
import math
import torch
from torch import nn


class FFNPrior(nn.Module):
    def __init__(self,inputs,outputs,rank,generator,device):
        super().__init__()
        initial=(torch.rand(rank,inputs,generator=generator,device='cpu')*2-1)/math.sqrt(inputs)
        self.A=nn.Parameter(initial.to(device=device,dtype=torch.float32))
        self.B=nn.Parameter(torch.zeros(outputs,rank,device=device,dtype=torch.float32))


class PhysicalFFNMemory(nn.Module):
    """Slow trained initialization; episodic values never become model parameters.

    Write rates start at one, preserving the existing Adam write. First-order
    meta training differentiates through the prior and rate, with the observed
    source gradient detached. This avoids requiring FA2 double backward.
    """
    def __init__(self,targets,*,rank=16,seed=731,scale=2.,device='cpu'):
        super().__init__();self.targets=dict(targets);self.rank=rank;self.scale=scale
        generator=torch.Generator(device='cpu').manual_seed(seed)
        self.priors=nn.ModuleList([FFNPrior(i,o,rank,generator,device) for i,o in targets.values()])
        self.log_write_scale=nn.Parameter(torch.zeros(len(targets),device=device,dtype=torch.float32))

    def factors(self,*,create_graph=False):
        result={}
        for name,prior in zip(self.targets,self.priors):
            for suffix in ('A','B'):
                value=getattr(prior,suffix)
                result[name+'.'+suffix]=(value.clone() if create_graph else value.detach().clone()).requires_grad_(True)
        return result

    def rate(self,name):
        index=list(self.targets).index(name.rsplit('.',1)[0])
        return self.log_write_scale[index].exp()
