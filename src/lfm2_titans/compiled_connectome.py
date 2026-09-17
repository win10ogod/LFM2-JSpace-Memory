"""Exact finite-step execution plans; the physical genome stays untouched.

Zero-preserving node activations and a fixed K allow dead paths to be omitted
from a port's execution, while keeping all physical weights and their update
normalization. This is an execution optimization, not a smaller experiment.
"""
import numpy as np
import torch
from torch import nn
from .connectome_ops import sparse_layout, SparseMM


def path_support(nodes, src, dst, inputs, outputs, steps):
    forward=np.full(nodes,steps+1,dtype=np.int16);backward=forward.copy()
    forward[list(inputs)]=0;backward[list(outputs)]=0
    for _ in range(steps):
        np.minimum.at(forward,dst,forward.copy()[src]+1)
        np.minimum.at(backward,src,backward.copy()[dst]+1)
    return forward[src]+1+backward[dst]<=steps,forward,backward


class PortExecutionPlan(nn.Module):
    def __init__(self,nodes,indices,spec,steps):
        super().__init__()
        dst,src=indices.detach().cpu().numpy()
        support,_,_=path_support(nodes,src,dst,spec.input_nodes,spec.read_nodes,steps)
        edge_ids=np.flatnonzero(support)
        # np.unique here constructs an index mapping, not corpus deduplication.
        used=np.unique(np.concatenate([src[edge_ids],dst[edge_ids],spec.input_nodes,spec.read_nodes]))
        edges=torch.from_numpy(np.stack([np.searchsorted(used,dst[edge_ids]),np.searchsorted(used,src[edge_ids])]))
        ix,tx,perm,inv,order=sparse_layout(edges,len(used))
        for name,tensor in dict(edge_ids=torch.from_numpy(edge_ids)[order],indices=ix,transposed=tx,
            permutation=perm,inverse=inv,input_nodes=torch.from_numpy(np.searchsorted(used,spec.input_nodes)),
            read_nodes=torch.from_numpy(np.searchsorted(used,spec.read_nodes))).items():
            # Persist plans: HF's meta-device materialization otherwise turns
            # nonpersistent constructor buffers into uninitialized storage.
            self.register_buffer(name,tensor,persistent=True)
        self.nodes=len(used)

    def execute(self,bank,port,x,fast,projection,*,compact=False,read_projection=None):
        c=bank.channels
        projected=projection(x).reshape(len(x),len(self.input_nodes),c)
        drive=x.new_zeros(self.nodes,c,len(x)).index_copy(0,self.input_nodes,projected.permute(1,2,0))
        activity=bank._activate(drive)
        weights=fast if compact else fast[self.edge_ids]
        for _ in range(bank.microsteps):
            recurrent=torch.stack([SparseMM.apply(weights if c==1 else weights[:,i],activity[:,i,:],
                self.indices,self.transposed,self.permutation,self.inverse,bank.sparse_backend) for i in range(c)],dim=1)
            activity=bank._activate((1-bank.leak)*activity+bank.leak*(recurrent+drive))
        reader=port.read_projection if read_projection is None else read_projection
        return reader(activity[self.read_nodes].permute(2,0,1).reshape(len(x),-1))


def build_execution_plans(bank):
    plans=nn.ModuleDict({p.name:PortExecutionPlan(bank.nodes,bank.indices,p,bank.microsteps) for p in bank.specs})
    return plans.to(bank.slow_weights.device)


class RecomputedWriteGradient(torch.autograd.Function):
    """Exact outer VJP of an inner gradient, recomputed one chunk at a time.

All differentiable inputs, including port projections, are explicit. Unlike a
checkpoint around autograd.grad(create_graph=True), this forward retains no
inner derivative graph. The backward reconstructs its exact Hessian-vector
    product. Higher derivative requests keep independent input clones connected
    to their original inputs, at the caller's corresponding memory cost.
"""
    @staticmethod
    def forward(ctx,function,*inputs):
        ctx.function=function;ctx.save_for_backward(*inputs)
        local=[x.detach().requires_grad_(x.requires_grad or i==0) for i,x in enumerate(inputs)]
        with torch.enable_grad():loss,gradient=function(*local)
        return loss.detach(),gradient.detach()

    @staticmethod
    def backward(ctx,loss_grad,gradient_grad):
        higher=torch.is_grad_enabled()
        with torch.enable_grad():
            local=[(x.clone() if higher and x.requires_grad else
                    x.detach().requires_grad_(ctx.needs_input_grad[i+1] or i==0))
                   for i,x in enumerate(ctx.saved_tensors)]
            _,gradient=ctx.function(*local)
            needed=[x for i,x in enumerate(local) if ctx.needs_input_grad[i+1]]
            derivatives=torch.autograd.grad(gradient,needed,grad_outputs=gradient_grad,
                                            allow_unused=True,create_graph=higher)
        iterator=iter(derivatives)
        return (None,*(next(iterator) if ctx.needs_input_grad[i+1] else None for i in range(len(local))))
