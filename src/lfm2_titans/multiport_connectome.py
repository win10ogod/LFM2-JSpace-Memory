"""Distributed connectome ports with explicit, causal Titans write transactions.

Each port projects directly between its model feature space and its own neuron
sets. There is no shared narrow input/output latent. All reads in one block use
the same fast-weight snapshot; only an explicit commit creates the next state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .connectome_ops import SparseMM, sparse_layout
from .capacity import structural_support, allocate_ownership
from .compiled_connectome import build_execution_plans, RecomputedWriteGradient


@dataclass(frozen=True)
class PortSpec:
    name: str
    feature_dim: int
    input_nodes: tuple[int, ...]
    read_nodes: tuple[int, ...]


@dataclass(frozen=True)
class ConnectomeState:
    fast: torch.Tensor
    momentum: torch.Tensor
    commits: int = 0

    def detach(self):
        """Explicit inference/TBPTT boundary, never used inside an outer update."""
        return ConnectomeState(self.fast.detach().requires_grad_(True),
                               self.momentum.detach(), self.commits)


@dataclass(frozen=True)
class MemoryPage:
    page_id: str
    state: ConnectomeState
    weight: float = 1.0


@dataclass(frozen=True)
class MountedState:
    """One writable working bank plus immutable recalled neural weight pages."""
    active: ConnectomeState
    pages: tuple[MemoryPage, ...] = ()
    active_weight: float = 1.0

    def __post_init__(self):
        weights = [self.active_weight] + [p.weight for p in self.pages]
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError('page read weights must be finite, nonnegative, and have positive sum')
        if len({p.page_id for p in self.pages}) != len(self.pages):
            raise ValueError('duplicate recalled page')

    @property
    def fast(self): return self.active.fast

    @property
    def momentum(self): return self.active.momentum

    @property
    def commits(self): return self.active.commits

    def detach(self):
        return MountedState(self.active.detach(), tuple(MemoryPage(p.page_id, p.state.detach(), p.weight)
                            for p in self.pages), self.active_weight)


class NeuronPort(nn.Module):
    def __init__(self, spec: PortSpec, nodes: int, channels: int = 1, learned_value: bool = False,
                 separate_write_key: bool = False):
        super().__init__()
        for label, selected in (("input", spec.input_nodes), ("read", spec.read_nodes)):
            if not selected or len(set(selected)) != len(selected):
                raise ValueError(f"{spec.name}: {label} nodes must be nonempty and unique")
            if min(selected) < 0 or max(selected) >= nodes:
                raise ValueError(f"{spec.name}: {label} outside graph")
        if spec.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.register_buffer("input_nodes", torch.tensor(spec.input_nodes))
        self.register_buffer("read_nodes", torch.tensor(spec.read_nodes))
        self.input_projection = nn.Linear(spec.feature_dim, len(spec.input_nodes) * channels, bias=False)
        self.write_projection = (nn.Linear(spec.feature_dim, len(spec.input_nodes) * channels, bias=False)
                                 if separate_write_key else None)
        self.read_projection = nn.Linear(len(spec.read_nodes) * channels, spec.feature_dim, bias=False)
        self.value_projection = nn.Linear(spec.feature_dim, spec.feature_dim, bias=False) if learned_value else nn.Identity()
        if learned_value:
            nn.init.eye_(self.value_projection.weight)
        # Per-feature residual gates; every port has its own parameters.
        self.residual_gate = nn.Parameter(torch.full((spec.feature_dim,), -4.0))


class MultiportConnectome(nn.Module):
    def __init__(self, nodes: int, edges: torch.Tensor, ports: list[PortSpec], *,
                 microsteps: int = 3, leak: float = 0.5, write_rate: float = 0.05,
                 momentum: float = 0.9, forgetting: float = 0.001,
                 initial_weights: torch.Tensor | None = None, channels: int = 1,
                 activation: str = "tanh", gradient_normalization: str = "none",
                 residual_recall: bool = False, learned_value: bool = False,
                 feature_chunk_size: int = 256, checkpoint_reads: bool = False,
                 separate_write_key: bool = False, write_association: str = "reconstruct",
                 write_loss: str = "mse", visual_protection: bool = False,
                 retention_anchor: str = "zero", channel_forgetting: list[float] | None = None,
                 checkpoint_write_gradients: bool = False, write_ownership: str = "shared",
                 association_fraction: float = 0.25, ownership_seed: int = 17,
                 decay_enabled: bool = True, compiled_execution: bool = False,
                 surprise_gate: bool = False, surprise_reference: float = 1e-4,
                 sparse_backend: str = 'coo', bounded_inference_writes: bool = False):
        super().__init__()
        if nodes <= 0 or microsteps < 1 or not 0 < leak <= 1:
            raise ValueError("invalid graph dynamics")
        if len(ports) < 2 or len({p.name for p in ports}) != len(ports):
            raise ValueError("at least two independently named ports are required")
        if not 0 < write_rate or not 0 < momentum < 1 or not 0 < forgetting < 1:
            raise ValueError("invalid plasticity coefficients")
        if channels < 1 or activation not in ("tanh", "softsign", "dendritic"):
            raise ValueError("invalid neuron operator")
        if activation == "dendritic" and channels < 2:
            raise ValueError("dendritic operator requires at least two channels")
        if gradient_normalization not in ("none", "rms") or feature_chunk_size < 1:
            raise ValueError("invalid write normalization or chunk size")
        if write_association not in ("reconstruct","next") or write_loss not in ("mse","huber"):
            raise ValueError("invalid associative write objective")
        if retention_anchor not in ("zero","slow"):
            raise ValueError("invalid retention anchor")
        if write_ownership not in ('shared', 'partitioned'):
            raise ValueError('invalid write ownership')
        if channel_forgetting is not None and (channels<2 or len(channel_forgetting)!=channels
                or any(not 0<value<1 for value in channel_forgetting)):
            raise ValueError("channel forgetting requires one coefficient per vector channel")
        ix, tx, permutation, inverse, order = sparse_layout(edges, nodes)
        if ix.shape[1] == 0:
            raise ValueError("graph has no connections")
        for name, tensor in (("indices", ix), ("transposed", tx),
                             ("permutation", permutation), ("inverse", inverse)):
            self.register_buffer(name, tensor)
        if initial_weights is None:
            degree = torch.bincount(ix[0], minlength=nodes).clamp_min(1)
            weights = torch.randn(ix.shape[1], device=ix.device) / degree[ix[0]].sqrt()
        else:
            if initial_weights.shape != (ix.shape[1],):
                raise ValueError("one independent weight per connection is required")
            weights = initial_weights[order].clone()
        if channels > 1:
            weights = weights[:, None].repeat(1, channels)
        self.slow_weights = nn.Parameter(weights)
        self.ports = nn.ModuleDict({p.name: NeuronPort(p, nodes, channels, learned_value, separate_write_key) for p in ports})
        self.log_write_rate = nn.Parameter(torch.tensor(math.log(write_rate)))
        if surprise_reference<=0:raise ValueError('surprise reference must be positive')
        self.surprise_gate=surprise_gate
        self.surprise_reference=surprise_reference
        if surprise_gate:
            self.log_surprise_reference=nn.Parameter(torch.tensor(math.log(surprise_reference)))
        self.momentum_logit = nn.Parameter(torch.tensor(math.log(momentum / (1 - momentum))))
        self.forgetting_logit = nn.Parameter(torch.tensor(math.log(forgetting / (1 - forgetting))))
        if channel_forgetting is not None:
            values=torch.tensor(channel_forgetting)
            self.forgetting_logit=nn.Parameter(torch.logit(values))
        if visual_protection:
            visual=next((p for p in ports if p.name=="visual"),None)
            if visual is None:raise ValueError("visual protection requires an explicit visual port")
            selected=torch.zeros(nodes,dtype=torch.bool,device=ix.device)
            selected[list(visual.input_nodes)]=True
            protected=selected[ix[0]] & selected[ix[1]]
            if not protected.any():raise ValueError("visual input subgraph has no internal edges to protect")
            self.register_buffer("visual_protected_edges",protected)
        self.nodes, self.microsteps, self.leak = nodes, microsteps, leak
        self.specs = ports
        self.channels, self.activation = channels, activation
        self.gradient_normalization, self.residual_recall = gradient_normalization, residual_recall
        self.learned_value = learned_value
        self.feature_chunk_size, self.checkpoint_reads = feature_chunk_size, checkpoint_reads
        self.separate_write_key, self.write_association, self.write_loss = separate_write_key, write_association, write_loss
        self.visual_protection,self.retention_anchor=visual_protection,retention_anchor
        self.channel_forgetting=channel_forgetting
        self.checkpoint_write_gradients=checkpoint_write_gradients
        self.write_ownership, self.association_fraction = write_ownership, association_fraction
        self.ownership_seed, self.decay_enabled = ownership_seed, decay_enabled
        if write_ownership == 'partitioned':
            support = structural_support(nodes, ix, ports, microsteps)
            self.register_buffer('write_owners', allocate_ownership(support, channels,
                                 association_fraction, ownership_seed).to(ix.device))
        self.compiled_execution=compiled_execution
        if sparse_backend not in ('coo','csr'):raise ValueError('invalid sparse backend')
        self.sparse_backend=sparse_backend
        self.bounded_inference_writes=bounded_inference_writes
        self.execution_plans=build_execution_plans(self) if compiled_execution else nn.ModuleDict()

    def _owned_write_gradient(self, write_weights, observations, create_graph):
        gradient = torch.zeros_like(write_weights)
        loss = write_weights.new_zeros(())
        writable = torch.zeros_like(write_weights, dtype=torch.bool)
        names = [p.name for p in self.specs]
        for observation in observations:
            owner = self.write_owners[names.index(observation[0])]
            if self.channels == 1: owner = owner[:, 0]
            if create_graph and self.checkpoint_write_gradients:
                port_loss, port_gradient = self._bounded_write_gradient(write_weights, (observation,))
            elif not create_graph and self.bounded_inference_writes and self.compiled_execution:
                port_loss,port_gradient=self._compact_bounded_write_gradient(write_weights,(observation,),create_graph=False)
            else:
                port_loss, port_gradient = self._write_loss_gradient(write_weights, (observation,), create_graph)
            gradient = gradient + port_gradient * owner / len(observations)
            loss = loss + port_loss / len(observations)
            writable = writable | owner
        return loss, gradient, writable

    def _write_loss_gradient(self, write_weights, observations, create_graph):
        with torch.enable_grad(), torch.autocast(device_type=write_weights.device.type,enabled=False):
            losses=[]
            for name,features,prepared_target in observations:
                keys,values=features,features
                if prepared_target is not None:
                    values=prepared_target
                elif self.write_association=='next' and name!='visual':
                    if features.ndim==3:
                        if features.shape[1]<2:raise ValueError('next-feature writes require at least two observed tokens')
                        keys,values=features[:,:-1],features[:,1:]
                    else:
                        if features.shape[0]<2:raise ValueError('next-feature writes require at least two observed tokens')
                        keys,values=features[:-1],features[1:]
                prediction=self.read(name,keys,write_weights,for_write=True)
                target=F.layer_norm(values.detach().to(prediction.dtype),(features.shape[-1],))
                target=self.ports[name].value_projection(target)
                losses.append(F.mse_loss(prediction,target) if self.write_loss=='mse'
                    else F.smooth_l1_loss(prediction,target,beta=0.5))
            loss=torch.stack(losses).mean()
            gradient=torch.autograd.grad(loss,write_weights,create_graph=create_graph)[0]
        return loss.detach(),gradient

    def _chunk_write_gradient(self, name, write_weights, keys, values):
        with torch.enable_grad(),torch.autocast(device_type=write_weights.device.type,enabled=False):
            prediction=self.read(name,keys,write_weights,for_write=True)
            target=F.layer_norm(values.detach().to(prediction.dtype),(values.shape[-1],))
            target=self.ports[name].value_projection(target)
            loss=(F.mse_loss(prediction,target) if self.write_loss=='mse'
                  else F.smooth_l1_loss(prediction,target,beta=.5))
            gradient=torch.autograd.grad(loss,write_weights,create_graph=True)[0]
        return loss.detach(),gradient

    def _bounded_write_gradient(self, write_weights, observations):
        """Same mean loss/gradient, recomputing each observation chunk separately.

        Checkpointing the entire inner gradient still builds a complete
        second-order graph during its first execution. Chunk checkpointing
        bounds that temporary graph without detaching features or fast weights.
        """
        if self.compiled_execution:
            return self._compact_bounded_write_gradient(write_weights,observations)
        from torch.utils.checkpoint import checkpoint
        from .sft_lora import checkpoint_weight_contexts
        gradient=torch.zeros_like(write_weights)
        loss=write_weights.new_zeros(())
        for name,features,prepared_target in observations:
            keys,values=features,features
            if prepared_target is not None:values=prepared_target
            elif self.write_association=='next' and name!='visual':
                if features.ndim==3:
                    if features.shape[1]<2:raise ValueError('next-feature writes require at least two observed tokens')
                    keys,values=features[:,:-1],features[:,1:]
                else:
                    if features.shape[0]<2:raise ValueError('next-feature writes require at least two observed tokens')
                    keys,values=features[:-1],features[1:]
            keys=keys.reshape(-1,keys.shape[-1]);values=values.reshape(-1,values.shape[-1])
            for key_chunk,value_chunk in zip(keys.split(self.feature_chunk_size),values.split(self.feature_chunk_size)):
                chunk_loss,chunk_grad=checkpoint(self._chunk_write_gradient,name,write_weights,
                    key_chunk,value_chunk,use_reentrant=False,context_fn=checkpoint_weight_contexts)
                scale=len(key_chunk)/(len(keys)*len(observations))
                loss=loss+chunk_loss*scale
                gradient=gradient+chunk_grad*scale
        return loss,gradient

    def _compact_chunk_gradient(self,name,compact_weights,keys,values,input_weight,read_weight,value_weight,create_graph=True):
        # Differentiate the same loss in the exact supported subspace, then
        # scatter once per port. Scattering a 104M-element gradient for every
        # feature chunk otherwise retains dozens of redundant full tensors.
        with torch.enable_grad(),torch.autocast(device_type=keys.device.type,enabled=False):
            port=self.ports[name]
            projection=lambda x:F.linear(x,input_weight)
            x=F.layer_norm(keys.to(self.slow_weights.dtype),(keys.shape[-1],))
            prediction=self.execution_plans[name].execute(self,port,x,compact_weights,projection,compact=True,
                read_projection=lambda x:F.linear(x,read_weight))
            target=F.layer_norm(values.detach().to(prediction.dtype),(values.shape[-1],))
            if value_weight.numel():target=F.linear(target,value_weight)
            loss=(F.mse_loss(prediction,target) if self.write_loss=='mse'
                  else F.smooth_l1_loss(prediction,target,beta=.5))
            grad=torch.autograd.grad(loss,compact_weights,create_graph=create_graph)[0]
        return loss.detach(),grad

    def _compact_bounded_write_gradient(self,write_weights,observations,create_graph=True):
        from torch.utils.checkpoint import checkpoint
        gradient=torch.zeros_like(write_weights);loss=write_weights.new_zeros(())
        for name,features,prepared_target in observations:
            keys,values=features,features
            if prepared_target is not None:values=prepared_target
            elif self.write_association=='next' and name!='visual':
                if features.ndim==3:
                    if features.shape[1]<2:raise ValueError('next-feature writes require two tokens')
                    keys,values=features[:,:-1],features[:,1:]
                else:
                    if features.shape[0]<2:raise ValueError('next-feature writes require two tokens')
                    keys,values=features[:-1],features[1:]
            keys=keys.reshape(-1,keys.shape[-1]);values=values.reshape(-1,values.shape[-1])
            edge_ids=self.execution_plans[name].edge_ids
            compact=write_weights[edge_ids];port_grad=torch.zeros_like(compact)
            port=self.ports[name]
            input_weight=(port.write_projection if port.write_projection is not None else port.input_projection).weight
            # Weight-aware LoRA materializes a dense effective weight. Share it
            # across chunks: each RecomputedWriteGradient saves its inputs until
            # outer backward. Expanding inside this loop retained one full
            # projection per feature chunk, instead of one per observation.
            read_weight=port.read_projection.weight
            value_weight=port.value_projection.weight if self.learned_value else write_weights.new_empty(0)
            for key,value in zip(keys.split(self.feature_chunk_size),values.split(self.feature_chunk_size)):
                fn=lambda *args,_name=name:self._compact_chunk_gradient(_name,*args)
                arguments=(compact,key,value,input_weight,read_weight,value_weight)
                if create_graph:chunk_loss,chunk_grad=RecomputedWriteGradient.apply(fn,*arguments)
                else:chunk_loss,chunk_grad=self._compact_chunk_gradient(name,*arguments,create_graph=False)
                scale=len(key)/(len(keys)*len(observations))
                loss=loss+chunk_loss*scale;port_grad=port_grad+chunk_grad*scale
            gradient=gradient.index_add(0,edge_ids,port_grad)
        return loss,gradient

    def _activate(self, activity):
        if self.activation == "tanh":
            return activity.tanh()
        if self.activation == "softsign":
            return F.softsign(activity)
        return activity.tanh() * torch.roll(activity, 1, dims=1).sigmoid()

    def _read_flat(self, name, x, fast, for_write=False):
        port = self.ports[name]
        c = self.channels
        projection = port.write_projection if for_write and port.write_projection is not None else port.input_projection
        if self.compiled_execution:
            return self.execution_plans[name].execute(self,port,x,fast,projection)
        projected = projection(x).reshape(len(x), len(port.input_nodes), c)
        drive = x.new_zeros(self.nodes, c, len(x)).index_copy(0, port.input_nodes, projected.permute(1, 2, 0))
        activity = self._activate(drive)
        for _ in range(self.microsteps):
            recurrent = torch.stack([SparseMM.apply(
                fast if c == 1 else fast[:, channel], activity[:, channel, :], self.indices,
                self.transposed, self.permutation, self.inverse,self.sparse_backend) for channel in range(c)], dim=1)
            activity = self._activate((1 - self.leak) * activity + self.leak * (recurrent + drive))
        return port.read_projection(activity[port.read_nodes].permute(2, 0, 1).reshape(len(x), -1))

    def initial_state(self):
        return ConnectomeState(self.slow_weights * 1, torch.zeros_like(self.slow_weights))

    def begin(self, state: ConnectomeState):
        return MemoryBlock(self, state)

    def read(self, name: str, features: torch.Tensor, fast: torch.Tensor, for_write: bool = False):
        port = self.ports[name]
        shape = features.shape
        # Sparse edge dynamics stay in the bank's explicit precision. An outer
        # BF16 autocast for LFM must not silently change the inner update.
        with torch.autocast(device_type=features.device.type, enabled=False):
            x = features.to(self.slow_weights.dtype).reshape(-1, shape[-1])
            x = F.layer_norm(x, (shape[-1],))
            results = []
            for chunk in x.split(self.feature_chunk_size):
                if self.checkpoint_reads and torch.is_grad_enabled():
                    from torch.utils.checkpoint import checkpoint
                    from .sft_lora import checkpoint_weight_contexts
                    result = checkpoint(self._read_flat, name, chunk, fast, for_write,
                        use_reentrant=False,context_fn=checkpoint_weight_contexts)
                else:
                    result = self._read_flat(name, chunk, fast, for_write)
                results.append(result)
            result = torch.cat(results, dim=0)
        return result.reshape(shape)

    def architecture(self):
        return {
            "kind": "multiport_connectome_titans", "nodes": self.nodes,
            "edge_parameter_order": "lexicographic destination/source local indices; authoritative indices buffer is saved in memory_state_dict",
            "edges": self.indices.shape[1], "microsteps": self.microsteps,
            "leak": self.leak, "write_schedule": "block_snapshot_then_commit",
            "channels": self.channels, "activation": self.activation,
            "gradient_normalization": self.gradient_normalization,
            "residual_recall": self.residual_recall, "learned_value": self.learned_value,
            "feature_chunk_size": self.feature_chunk_size, "checkpoint_reads": self.checkpoint_reads,
            "separate_write_key": self.separate_write_key,"write_association":self.write_association,
            "write_loss":self.write_loss,
            "visual_protection":self.visual_protection,"retention_anchor":self.retention_anchor,
            "channel_forgetting":self.channel_forgetting,
            "checkpoint_write_gradients":self.checkpoint_write_gradients,
            "write_ownership": self.write_ownership, "association_fraction": self.association_fraction,
            "ownership_seed": self.ownership_seed, "decay_enabled": self.decay_enabled,
            "compiled_execution": self.compiled_execution,
            "sparse_backend": self.sparse_backend,
            "bounded_inference_writes": self.bounded_inference_writes,
            "surprise_gate": self.surprise_gate, "surprise_reference": self.surprise_reference,
            "visual_protected_edge_count":int(self.visual_protected_edges.sum()) if self.visual_protection else 0,
            "shared_input_or_output_latent": False,
            "ports": [dict(name=p.name, feature_dim=p.feature_dim,
                           input_nodes=list(p.input_nodes), read_nodes=list(p.read_nodes))
                      for p in self.specs],
        }


class MemoryBlock:
    """One session's block. Reads are pure, observations explicit, commit once.

    Only causally visible observations may be queued. Keep this object outside
    activation checkpoint recomputation. No model/global session state is used.
    """
    def __init__(self, memory: MultiportConnectome, state: ConnectomeState):
        self.memory, self.snapshot = memory, state
        self._observations: dict[str, tuple[str, torch.Tensor, torch.Tensor | None]] = {}
        self.native_observations = {}
        self._committed = False

    def read(self, name: str, features: torch.Tensor):
        # The snapshot remains valid for autograd recomputation after commit.
        if isinstance(self.snapshot, MountedState):
            weighted = [(self.snapshot.active_weight, self.snapshot.active)]
            weighted.extend((page.weight, page.state) for page in self.snapshot.pages)
            total = sum(weight for weight, _ in weighted)
            # Mix independently evaluated neural banks, never sum unrelated
            # physical fast weights or duplicate the slow prior per page.
            return sum(self.memory.read(name, features, state.fast) * (weight / total)
                       for weight, state in weighted if weight > 0)
        return self.memory.read(name, features, self.snapshot.fast)

    def routing_keys(self):
        """Small neural addresses from observed features; contains no text/images.

        Whole-block keys may route a *later* read only, not earlier tokens in the
        same causal block. They are addressing metadata, not memory contents.
        """
        keys = {}
        for name, features, _ in self._observations.values():
            key = F.layer_norm(features.detach().float(), (features.shape[-1],))
            keys[name] = F.normalize(key.reshape(-1, key.shape[-1]).mean(0), dim=0).cpu()
        return keys

    def residual(self, name: str, features: torch.Tensor):
        value = self.read(name, features)
        if self.memory.residual_recall:
            value = value - self.memory.read(name, features, self.memory.slow_weights)
        gate = self.memory.ports[name].residual_gate.sigmoid()
        return features + (value * gate).to(features.dtype)

    def observe(self, event_id: str, name: str, features: torch.Tensor,
                valid_mask: torch.Tensor | None = None):
        if self._committed:
            raise RuntimeError("block already committed")
        if event_id in self._observations:
            raise ValueError(f"duplicate observation {event_id}; checkpoint/cache writes must be explicit")
        if name not in self.memory.ports:
            raise KeyError(name)
        if valid_mask is not None and valid_mask.shape != features.shape[:-1]:
            raise ValueError("valid mask must match observation dimensions")
        # Capture every valid feature before next-feature Titans association
        # removes the last member of a pair. The VAE stores all observations.
        self.native_observations[name]=(features[valid_mask.bool()] if valid_mask is not None else features)
        prepared_target=None
        if valid_mask is not None:
            if self.memory.write_association=="next" and name!="visual" and features.ndim==3:
                adjacent=valid_mask[:,1:].bool() & valid_mask[:,:-1].bool()
                prepared_target=features[:,1:][adjacent]
                features=features[:,:-1][adjacent]
            else:
                features = features[valid_mask.bool()]
        if features.numel() == 0:
            raise ValueError("observation has no valid features")
        self._observations[event_id] = (name, features, prepared_target)


    def commit(self, *, create_graph: bool):
        if self._committed:
            raise RuntimeError("block already committed")
        if not self._observations:
            raise ValueError("no observations to write")
        bank = self.memory
        with torch.enable_grad():
            # Inner differentiation is partial w.r.t. this functional weight
            # argument, not through the earlier LFM computation of its keys.
            # Outer gradients still reach those keys and the prior fast state.
            write_weights = self.snapshot.fast * 1
            if not write_weights.requires_grad:
                write_weights.requires_grad_(True)
            observations=tuple(self._observations.values())
            writable=None
            if bank.write_ownership == 'partitioned':
                loss,gradient,writable=bank._owned_write_gradient(write_weights,observations,create_graph)
            elif create_graph and bank.checkpoint_write_gradients:
                loss,gradient=bank._bounded_write_gradient(write_weights,observations)
            elif not create_graph and bank.bounded_inference_writes and bank.compiled_execution:
                loss,gradient=bank._compact_bounded_write_gradient(write_weights,observations,create_graph=False)
            else:
                loss,gradient=bank._write_loss_gradient(write_weights,observations,create_graph)
            raw_gradient_norm = float(gradient.detach().norm())
            if bank.visual_protection and not any(name=="visual" for name,_,_ in self._observations.values()):
                permitted=~bank.visual_protected_edges
                if bank.channels>1:permitted=permitted[:,None].expand_as(gradient)
                writable=permitted if writable is None else writable & permitted
                gradient=gradient*writable
            mean_square=(gradient.square().sum()/writable.sum().clamp_min(1)
                         if writable is not None else gradient.square().mean())
            raw_rms=mean_square.clamp_min(torch.finfo(gradient.dtype).tiny).sqrt()
            if bank.gradient_normalization == "rms":
                gradient = gradient / (mean_square.clamp_min(torch.finfo(gradient.dtype).tiny).sqrt() + 1e-6)
            surprise_strength=2*torch.sigmoid(raw_rms.log()-bank.log_surprise_reference) if bank.surprise_gate else 1.
            gradient=gradient*surprise_strength
            momentum = bank.momentum_logit.sigmoid() * self.snapshot.momentum + gradient
            decay = bank.forgetting_logit.sigmoid() if bank.decay_enabled else 0.
            fast = ((1 - decay) * self.snapshot.fast
                    - bank.log_write_rate.exp() * momentum)
            if bank.retention_anchor=="slow":
                fast=fast+decay*bank.slow_weights
            if writable is not None:
                # A protected visual trace must not drift from momentum or
                # decay during text-only writes either. Other edges still learn.
                momentum=torch.where(writable,momentum,self.snapshot.momentum)
                fast=torch.where(writable,fast,self.snapshot.fast)
        state = ConnectomeState(fast, momentum, self.snapshot.commits + 1)
        if isinstance(self.snapshot, MountedState):
            state = MountedState(state, self.snapshot.pages, self.snapshot.active_weight)
        self._committed = True
        receipt = {"observations": len(observations), "write_loss": float(loss.detach()),
                   "write_gradient_norm": float(gradient.detach().norm()),
                   "raw_write_gradient_norm": raw_gradient_norm,
                   "fast_change_norm": float((fast - self.snapshot.fast).detach().norm()),
                   "commits": state.commits}
        if bank.write_ownership == 'partitioned':
            receipt['writable_scalars'] = int(writable.sum())
        receipt['decay_enabled'] = bank.decay_enabled
        if bank.surprise_gate:receipt['surprise_strength']=float(surprise_strength.detach())
        self._observations.clear()
        return (state if create_graph else state.detach()), receipt
