"""Finite-step support and independent edge/channel write ownership.

Support is a structural upper bound, never a count of remembered facts. Owner
allocation changes write permissions, including momentum/decay, not read paths.
"""
import numpy as np
import torch


def structural_support(nodes, indices, ports, microsteps):
    dst, src = indices.detach().cpu().numpy()
    masks = []
    for port in ports:
        forward = np.full(nodes, microsteps + 1, dtype=np.int32)
        backward = forward.copy()
        forward[list(port.input_nodes)] = 0
        backward[list(port.read_nodes)] = 0
        for _ in range(microsteps):
            np.minimum.at(forward, dst, forward.copy()[src] + 1)
            np.minimum.at(backward, src, backward.copy()[dst] + 1)
        masks.append(forward[src] + 1 + backward[dst] <= microsteps)
    return torch.from_numpy(np.stack(masks))


def allocate_ownership(support, channels, shared_fraction, seed=17):
    """One eligible owner per scalar, with an explicit shared association budget.

    Port selection is balanced *on each edge*, rotated across channels. A scalar
    is shared only when multiple ports can reach it. Unsupported scalars have
    no writer. No gradient magnitude or evaluation answers enter allocation.
    """
    if not 0 <= shared_fraction <= 1 or channels < 1:
        raise ValueError('invalid ownership budget')
    support = support.cpu().numpy()
    ports, edges = support.shape
    owners = np.zeros((ports, edges, channels), dtype=bool)
    rng = np.random.default_rng(seed)
    offsets = rng.integers(0, max(ports, 1), size=edges)
    shared = rng.random((edges, channels)) < shared_fraction
    eligible_count = support.sum(axis=0)
    rank = support.cumsum(axis=0) - 1
    for channel in range(channels):
        selected = (offsets + channel) % np.maximum(eligible_count, 1)
        owners[:, :, channel] = support & ((rank == selected) |
            (shared[:, channel] & (eligible_count > 1)))
    return torch.from_numpy(owners)


def capacity_report(bank):
    support = structural_support(bank.nodes, bank.indices, bank.specs, bank.microsteps)
    owners = (bank.write_owners.detach().cpu() if bank.write_ownership == 'partitioned'
              else support[:, :, None].expand(-1, -1, bank.channels))
    writers = owners.sum(0)
    total = bank.slow_weights.numel()
    return dict(edges=bank.indices.shape[1], channels=bank.channels,
        microsteps=bank.microsteps, fast_scalars=total,
        fast_and_momentum_bytes=2*total*bank.slow_weights.element_size(),
        observable_edges=int(support.any(0).sum()),
        observable_scalars=int((writers > 0).sum()),
        exclusive_scalars=int((writers == 1).sum()),
        shared_scalars=int((writers > 1).sum()),
        unreachable_scalars=int((writers == 0).sum()),
        ports=[dict(name=p.name, support_edges=int(support[i].sum()),
                    writable_scalars=int(owners[i].sum()),
                    exclusive_scalars=int((owners[i] & (writers == 1)).sum()))
               for i, p in enumerate(bank.specs)],
        interpretation='Structural accessibility and write permissions; effective recall must be measured.')
