"""Sparse neural connections with gradients through an online gradient update.

PyTorch's sparse-value gradient is not differentiable in the installed build.
The explicit bilinear backward below keeps the Titans outer gradient intact.
"""
from __future__ import annotations
import torch


def _csr_matrix(indices,values,nodes):
    """Reuse row pointers for the immutable, sorted physical layout."""
    version=None if indices.is_inference() else indices._version
    cached=getattr(indices,'_csr_row_cache',None)
    if cached is None or cached[:2]!=(version,nodes):
        crow=torch.cat([indices.new_zeros(1),torch.bincount(indices[0],minlength=nodes).cumsum(0)])
        indices._csr_row_cache=(version,nodes,crow)
    else:crow=cached[2]
    return torch.sparse_csr_tensor(crow,indices[1].contiguous(),values.contiguous(),
                                   size=(nodes,nodes),check_invariants=False)


class EdgeDot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, indices, transposed, permutation, inverse, backend='coo'):
        ctx.backend=backend
        ctx.save_for_backward(left, right, indices, transposed, permutation, inverse)
        if backend=='csr':
            # Sample only measured edges; never materialize an N x N product.
            # Explicit backward below retains higher derivatives through SDDMM.
            pattern=_csr_matrix(indices,left.new_zeros(indices.shape[1]),left.shape[0])
            with torch.autocast(device_type=left.device.type,enabled=False):
                return torch.sparse.sampled_addmm(pattern,left.contiguous(),right.T.contiguous(),beta=0).values()
        result = left.new_empty(indices.shape[1])
        # Bound temporary edge-by-observation storage without dropping edges.
        chunk = max(1024, 4_194_304 // max(1, left.shape[1]))
        for start in range(0, result.numel(), chunk):
            ix = indices[:, start:start + chunk]
            result[start:start + chunk] = (left[ix[0]] * right[ix[1]]).sum(-1)
        return result

    @staticmethod
    def backward(ctx, grad):
        left, right, ix, tx, perm, inv = ctx.saved_tensors
        gl = SparseMM.apply(grad, right, ix, tx, perm, inv,ctx.backend)
        gr = SparseMM.apply(grad[perm], left, tx, ix, inv, perm,ctx.backend)
        result=(gl,gr,None,None,None,None)
        return result+(None,) if len(ctx.needs_input_grad)==7 else result


class SparseMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, inputs, indices, transposed, permutation, inverse, backend='coo'):
        ctx.backend=backend
        ctx.save_for_backward(weights, inputs, indices, transposed, permutation, inverse)
        n = inputs.shape[0]
        # cuSPARSE's value array is a packed vector. A channel slice E x C
        # has stride C, so pass an explicit contiguous value array. The custom
        # backward still differentiates w.r.t. the original strided tensor.
        if backend=='csr':
            matrix=_csr_matrix(indices,weights,n)
        elif backend=='coo':
            matrix = torch.sparse_coo_tensor(indices, weights.contiguous(), (n, n),
                                            is_coalesced=True, check_invariants=False)
        else:raise ValueError('unknown sparse execution backend')
        # Recursive calls from the inner/outer backward may run inside a
        # Trainer autocast context. Preserve the bank's explicit precision.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return torch.sparse.mm(matrix, inputs.contiguous())

    @staticmethod
    def backward(ctx, grad):
        weights, inputs, ix, tx, perm, inv = ctx.saved_tensors
        gw = EdgeDot.apply(grad, inputs, ix, tx, perm, inv,ctx.backend) if ctx.needs_input_grad[0] else None
        gx = SparseMM.apply(weights[perm], grad, tx, ix, inv, perm,ctx.backend) if ctx.needs_input_grad[1] else None
        result=(gw,gx,None,None,None,None)
        return result+(None,) if len(ctx.needs_input_grad)==7 else result


def sparse_layout(indices: torch.Tensor, nodes: int):
    """Sort unique (destination, source) edges; return both multiplication orders."""
    indices = indices.long()
    order = (indices[0] * nodes + indices[1]).argsort()
    ix = indices[:, order]
    if ix.numel() and (ix.min() < 0 or ix.max() >= nodes):
        raise ValueError("connection index outside neuron universe")
    codes = ix[0] * nodes + ix[1]
    if len(codes) > 1 and bool((codes[1:] == codes[:-1]).any()):
        raise ValueError("duplicate connection rows; no automatic aggregation")
    perm = (ix[1] * nodes + ix[0]).argsort()
    inverse = perm.argsort()
    return ix, ix.flip(0)[:, perm], perm, inverse, order
