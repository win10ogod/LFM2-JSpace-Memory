"""Caller-owned physical fast adapters; native parameters stay checkpoint-fixed.

Each observation unit learns low-rank FFN weights at every existing language
memory depth. The state contains tensors only, never source strings or tokens.
Reads are functional and may coexist in independent sessions.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass,asdict
import hashlib,json,math,os,time,uuid
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
from .multiport_connectome import ConnectomeState
from .memory_integrity import _checksum
from .latent_memory import LatentPortMemory
from .sequence_memory import (validate_sequence,sequence_tensors,sequence_metadata,
                              sequence_shapes,load_sequences)
from .memory_address import NativeConceptAddress,validate_address,address_tensors,address_shapes,load_address,memory_hash


@dataclass(frozen=True)
class AdapterMemory:
    factors: dict
    first_moment: dict
    second_moment: dict
    commits: int=0

    def detach(self):
        return AdapterMemory({k:v.detach().requires_grad_(True) for k,v in self.factors.items()},
            {k:v.detach() for k,v in self.first_moment.items()},
            {k:v.detach() for k,v in self.second_moment.items()},self.commits)


@dataclass(frozen=True)
class AdapterMixture:
    """Read-only mixture of independently evaluated FFN memory updates."""
    components: tuple

    def __post_init__(self):
        if not self.components or any(not isinstance(s,AdapterMemory) or not math.isfinite(w) or w<=0
                                      for s,w in self.components):
            raise ValueError('adapter mixture requires memories with positive finite weights')
        if not math.isfinite(sum(w for _,w in self.components)):
            raise ValueError('adapter mixture total weight must be finite')


@dataclass(frozen=True)
class AdapterBatch:
    """Independent FFN memories aligned with native model batch rows."""
    rows: tuple


@dataclass(frozen=True)
class PhysicalMemoryUnit:
    adapters: AdapterMemory
    graph: ConnectomeState
    neural_keys: dict | None = None
    latents: dict | None = None
    sequences: tuple = ()
    address: NativeConceptAddress | None = None

    def detach(self):
        return PhysicalMemoryUnit(self.adapters.detach(),self.graph.detach(),self.neural_keys,
            {k:v.detach() for k,v in (self.latents or {}).items()},
            tuple(s.detach() for s in getattr(self,'sequences',())),
            self.address.detach() if getattr(self,'address',None) is not None else None)


class EpisodicAdapterBank:
    def __init__(self,model,rank=16,seed=731,*,architecture_owned=False):
        if type(rank) is not int or rank<1:raise ValueError('rank must be a positive integer')
        if getattr(model,'_episodic_adapter_bank',None) is not None:raise ValueError('adapters already installed')
        if not architecture_owned and any(p.requires_grad for p in model.parameters()):raise ValueError('freeze the inference body before installing fast adapters')
        self.checkpoint_id=getattr(model.config,'memory_checkpoint_id',None)
        if not self.checkpoint_id and not architecture_owned:raise ValueError('a checkpoint identity is required for physical weight memory')
        self.model=model;self.rank=rank;self.seed=seed;self.context=ContextVar(f'physical_adapters_{id(self)}',default=None)
        depths=[int(i) for i in model.config.language_ports]
        targets={n:m for n,m in model.named_modules() if isinstance(m,nn.Linear)
            and any(n.startswith(f'model.language_model.layers.{i}.') for i in depths)
            and n.rsplit('.',1)[-1] in ('w1','w2','w3')}
        if len(targets)!=3*len(depths):raise ValueError('expected all three FFN projections at every memory depth')
        self.targets={n:(m.in_features,m.out_features) for n,m in targets.items()}
        # Hooks keep every native state_dict name unchanged. A saved base model
        # never accidentally absorbs a session's fast weights or wrapper names.
        self.handles=[]
        for name,module in targets.items():
            def inject(module,args,output,name=name):
                state=self.context.get()
                if state is None:return output
                if isinstance(state,AdapterBatch):
                    x=args[0]
                    if x.shape[0]!=len(state.rows):raise ValueError('physical memory batch mismatch')
                    a=torch.stack([row.factors[name+'.A'] for row in state.rows]).to(x.dtype)
                    b=torch.stack([row.factors[name+'.B'] for row in state.rows]).to(x.dtype)
                    return output+torch.bmm(torch.bmm(x,a.transpose(1,2)),b.transpose(1,2))*2.
                components=state.components if isinstance(state,AdapterMixture) else ((state,1.),)
                total=sum(weight for _,weight in components);x=args[0]
                for memory,weight in components:
                    a=memory.factors[name+'.A'].to(x.dtype);b=memory.factors[name+'.B'].to(x.dtype)
                    output=output+F.linear(F.linear(x,a),b)*(2.*weight/total)
                return output
            self.handles.append(module.register_forward_hook(inject))
        model._episodic_adapter_bank=self

    def initial_state(self,*,create_graph=False):
        prior=getattr(self.model,'physical_memory',None)
        if prior is not None:
            values=prior.factors(create_graph=create_graph)
            return AdapterMemory(values,{k:torch.zeros_like(v) for k,v in values.items()},
                {k:torch.zeros_like(v) for k,v in values.items()})
        generator=torch.Generator(device='cpu').manual_seed(self.seed)
        device=next(self.model.parameters()).device;values={}
        for name,(inputs,outputs) in self.targets.items():
            a=(torch.rand(self.rank,inputs,generator=generator)*2-1)/math.sqrt(inputs)
            values[name+'.A']=a.to(device).requires_grad_(True)
            values[name+'.B']=torch.zeros(outputs,self.rank,device=device,requires_grad=True)
        return AdapterMemory(values,{k:torch.zeros_like(v) for k,v in values.items()},
            {k:torch.zeros_like(v) for k,v in values.items()})

    @contextmanager
    def use(self,state):
        if self.context.get() is not None:raise RuntimeError('nested adapter scope')
        token=self.context.set(state)
        try:yield
        finally:self.context.reset(token)

    @contextmanager
    def suspended(self):
        """Honor the native model's explicit use_memory=False in a read scope."""
        token=self.context.set(None)
        try:yield
        finally:self.context.reset(token)

    def commit(self,state,loss,learning_rate=.001,beta1=.9,beta2=.999,*,first_order_graph=False):
        if not math.isfinite(learning_rate) or learning_rate<=0 or not 0<=beta1<1 or not 0<=beta2<1:
            raise ValueError('positive finite learning rate and moment coefficients in [0,1) required')
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite memory write loss')
        gradients=torch.autograd.grad(loss,tuple(state.factors.values()),retain_graph=first_order_graph)
        return self.apply_gradients(state,gradients,learning_rate,beta1,beta2,
            first_order_graph=first_order_graph,loss=float(loss.detach()))

    def apply_gradients(self,state,gradients,learning_rate=.001,beta1=.9,beta2=.999,*,first_order_graph=False,loss=None):
        """Apply a session's already measured first-order source gradient."""
        if len(gradients)!=len(state.factors):raise ValueError('incomplete physical gradients')
        norm=torch.stack([g.float().square().sum() for g in gradients]).sum().sqrt()
        if not torch.isfinite(norm):raise FloatingPointError('nonfinite memory write gradients')
        clip=(1./norm.clamp_min(1.));step=state.commits+1
        factors={};first={};second={}
        prior=getattr(self.model,'physical_memory',None)
        with torch.set_grad_enabled(first_order_graph):
            for (key,value),gradient in zip(state.factors.items(),gradients):
                gradient=gradient.float()*clip
                first[key]=beta1*state.first_moment[key]+(1-beta1)*gradient
                second[key]=beta2*state.second_moment[key]+(1-beta2)*gradient.square()
                update=(first[key]/(1-beta1**step))/(second[key]/(1-beta2**step)).sqrt().add(1e-8)
                rate=prior.rate(key) if prior is not None else 1.
                updated=value-learning_rate*rate*update
                if not torch.isfinite(updated).all():raise FloatingPointError('nonfinite candidate physical weights')
                factors[key]=updated if first_order_graph else updated.detach().requires_grad_(True)
        return AdapterMemory(factors,first,second,step),dict(loss=loss,gradient_norm=float(norm),commits=step,
            write_gradient='first_order' if first_order_graph else 'inference')

    def save(self,state,path):
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        tensors={prefix+'.'+k:v.detach().cpu().contiguous() for prefix,values in (
            ('factor',state.factors),('first',state.first_moment),('second',state.second_moment)) for k,v in values.items()}
        metadata=dict(checkpoint_id=self.checkpoint_id,rank=self.rank,seed=self.seed,targets=self.targets,commits=state.commits)
        save_file(tensors,str(path),metadata={'physical_memory':json.dumps(metadata)})

    def load(self,path):
        from safetensors import safe_open
        with safe_open(str(path),framework='pt',device='cpu') as f:metadata=json.loads(f.metadata()['physical_memory'])
        expected=json.loads(json.dumps(self.targets))
        if (metadata['checkpoint_id']!=self.checkpoint_id or metadata['rank']!=self.rank or metadata['targets']!=expected):
            raise ValueError('physical adapter identity mismatch')
        tensors=load_file(str(path));device=next(self.model.parameters()).device
        groups=[{k.removeprefix(prefix+'.'):v.to(device) for k,v in tensors.items() if k.startswith(prefix+'.')}
            for prefix in ('factor','first','second')]
        expected_keys={name+'.'+suffix for name in self.targets for suffix in ('A','B')}
        if any(set(group)!=expected_keys for group in groups):raise ValueError('incomplete physical adapter state')
        result=AdapterMemory(*groups,metadata['commits']).detach();self.validate(result)
        return result

    def validate(self,state):
        if type(state.commits) is not int or state.commits<0:raise ValueError('invalid adapter commit counter')
        shapes={name+'.'+suffix:shape for name,(inputs,outputs) in self.targets.items()
            for suffix,shape in [('A',(self.rank,inputs)),('B',(outputs,self.rank))]}
        for group in (state.factors,state.first_moment,state.second_moment):
            if set(group)!=set(shapes):raise ValueError('incomplete physical adapter state')
            for name,value in group.items():
                if tuple(value.shape)!=shapes[name] or value.dtype!=torch.float32 or not torch.isfinite(value).all():
                    raise ValueError('invalid physical adapter tensor')
        if any((value<0).any() for value in state.second_moment.values()):raise ValueError('negative second moment')

    def _identity(self):
        if self.checkpoint_id!=self.model.config.memory_checkpoint_id:raise ValueError('body checkpoint changed; open a new memory runtime')
        indices=self.model.memory.indices.detach().cpu().contiguous()
        identity=dict(checkpoint_id=self.checkpoint_id,rank=self.rank,scale=2.,targets=self.targets,
            graph_shape=list(self.model.memory.slow_weights.shape),
            graph_sha256=hashlib.sha256(indices.numpy().tobytes()).hexdigest())
        if self.model.config.latent_memory is not None:identity['latent_memory']=self.model.config.latent_memory
        return identity

    def validate_latents(self,latents):
        if not latents:return
        if self.model.config.latent_memory is None:raise ValueError('latent memory is disabled')
        size=self.model.config.dream_memory['feature_vae']['latent_size']
        for name,port in latents.items():
            if name not in self.model.memory.ports or not isinstance(port,LatentPortMemory):raise ValueError('invalid latent port')
            if port.mu.ndim!=2 or port.mu.shape[1]!=size or port.logvar.shape!=port.mu.shape:
                raise ValueError('invalid latent code shape')
            if port.mean.shape!=(port.count,1) or port.scale.shape!=(port.count,1):raise ValueError('invalid latent normalization shape')
            if any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in port.tensors()):raise ValueError('invalid latent values')
            if (port.scale<=0).any():raise ValueError('latent scales must be positive')

    def _validate_graph(self,state):
        if type(state.commits) is not int or state.commits<0:raise ValueError('invalid graph commit counter')
        for value in (state.fast,state.momentum):
            if value.shape!=self.model.memory.slow_weights.shape or value.dtype!=torch.float32 or not torch.isfinite(value).all():
                raise ValueError('invalid physical graph tensor')

    def initial_unit(self,*,create_graph=False):
        graph=self.model.memory.initial_state()
        return PhysicalMemoryUnit(self.initial_state(create_graph=create_graph),graph if create_graph else graph.detach())

    def save_unit(self,unit,path):
        """One checksummed, append-only file contains every physical weight.

        No source tokens, answers, native KV caches, or raw images are serialized.
        Optional ordered memory includes VAE codes and residual decoder factors.
        Adam moments permit a later continuation of writes after cold loading.
        """
        self.validate(unit.adapters);self._validate_graph(unit.graph);self.validate_latents(unit.latents)
        tensors={prefix+'.'+k:v.detach().cpu().contiguous() for prefix,values in (
            ('factor',unit.adapters.factors),('first',unit.adapters.first_moment),('second',unit.adapters.second_moment))
            for k,v in values.items()}
        tensors.update({'graph.fast':unit.graph.fast.detach().cpu().contiguous(),
                        'graph.momentum':unit.graph.momentum.detach().cpu().contiguous()})
        key_dims={p.name:p.feature_dim for p in self.model.memory.specs}
        for name,key in (unit.neural_keys or {}).items():
            if name not in key_dims or key.shape!=(key_dims[name],) or not torch.isfinite(key).all():
                raise ValueError('invalid physical memory neural address')
            tensors['key.'+name]=key.detach().float().cpu().contiguous()
        for name,port in (unit.latents or {}).items():
            for label,value in zip(('mu','logvar','mean','scale'),port.tensors()):
                tensors['latent.'+name+'.'+label]=value.detach().cpu().contiguous()
        sequences=getattr(unit,'sequences',())
        for segment in sequences:validate_sequence(self.model,segment)
        tensors.update(sequence_tensors(sequences))
        address=getattr(unit,'address',None)
        if address is not None:validate_address(self.model,address)
        tensors.update(address_tensors(address))
        metadata=dict(version=6,identity=self._identity(),checksum=_checksum(tensors),
            latent_counts={name:port.count for name,port in (unit.latents or {}).items()},
            adapter_commits=unit.adapters.commits,graph_commits=unit.graph.commits,
            created_ns=time.time_ns(),unit_id=uuid.uuid4().hex)
        if sequences:metadata['sequences']=sequence_metadata(sequences)
        if address is not None:metadata['address']=address.metadata()
        metadata['memory_hash']=memory_hash(metadata['identity'],metadata['checksum'],metadata.get('address'))
        if len(json.dumps(metadata,sort_keys=True))>16384:raise ValueError('physical unit metadata exceeds load bound')
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.pending')
        lock=path.with_name('.'+path.name+'.writing')
        with lock.open('x'):
            try:
                if path.exists():raise FileExistsError(path)
                save_file(tensors,str(temporary),metadata={'physical_unit':json.dumps(metadata,sort_keys=True)})
                with temporary.open('rb') as stream:os.fsync(stream.fileno())
                os.replace(temporary,path)
            finally:temporary.unlink(missing_ok=True);lock.unlink(missing_ok=True)
        return dict(unit_path=str(path),bytes=path.stat().st_size,checksum=metadata['checksum'],memory_hash=metadata['memory_hash'])

    def load_unit(self,path):
        from safetensors import safe_open
        identity=json.loads(json.dumps(self._identity()))
        with safe_open(str(path),framework='pt',device='cpu') as handle:
            raw=(handle.metadata() or {}).get('physical_unit','')
            if len(raw)>16384:raise ValueError('oversized physical unit metadata')
            meta=json.loads(raw)
            if meta.get('version')!=6 or meta.get('identity')!=identity:raise ValueError('physical unit checkpoint/architecture mismatch')
            sequence_meta=meta.get('sequences',[])
            address_meta=meta.get('address')
            if not sequence_meta or address_meta is None or address_meta.get('version')!=2:
                raise ValueError('a complete ordered unit and native concept address are required')
            shapes={prefix+'.'+name+'.'+suffix:list(shape)
                for prefix in ('factor','first','second') for name,(inputs,outputs) in self.targets.items()
                for suffix,shape in [('A',(self.rank,inputs)),('B',(outputs,self.rank))]}
            shapes.update({'graph.fast':identity['graph_shape'],'graph.momentum':identity['graph_shape']})
            for port in self.model.memory.specs:
                if 'key.'+port.name in handle.keys():shapes['key.'+port.name]=[port.feature_dim]
            latent_counts=meta.get('latent_counts',{})
            for name,count in latent_counts.items():
                if (self.model.config.latent_memory is None or name not in self.model.memory.ports
                        or type(count) is not int or count<0):raise ValueError('invalid latent memory metadata')
                width=self.model.config.dream_memory['feature_vae']['latent_size']
                for label,dim in (('mu',width),('logvar',width),('mean',1),('scale',1)):
                    shapes['latent.'+name+'.'+label]=[count,dim]
            shapes.update(sequence_shapes(self.model,sequence_meta))
            shapes.update(address_shapes(self.model,address_meta))
            if set(handle.keys())!=set(shapes):raise ValueError('incomplete physical unit')
            if any(handle.get_slice(k).get_shape()!=shape for k,shape in shapes.items()):raise ValueError('invalid physical unit shapes')
            tensors={k:handle.get_tensor(k) for k in shapes}
        if _checksum(tensors)!=meta['checksum']:raise ValueError('physical unit checksum mismatch')
        if 'memory_hash' in meta and meta['memory_hash']!=memory_hash(identity,meta['checksum'],address_meta):
            raise ValueError('physical memory address hash mismatch')
        groups=[{k[len(prefix)+1:]:v for k,v in tensors.items() if k.startswith(prefix+'.')}
            for prefix in ('factor','first','second')]
        state=AdapterMemory(*groups,meta['adapter_commits']);self.validate(state)
        graph=ConnectomeState(tensors['graph.fast'],tensors['graph.momentum'],meta['graph_commits']);self._validate_graph(graph)
        device=self.model.memory.slow_weights.device
        state=AdapterMemory(*[{k:v.to(device) for k,v in group.items()} for group in groups],state.commits).detach()
        graph=ConnectomeState(graph.fast.to(device),graph.momentum.to(device),graph.commits).detach()
        keys={name[4:]:value for name,value in tensors.items() if name.startswith('key.')}
        if any(not torch.isfinite(value).all() for value in keys.values()):raise ValueError('nonfinite physical memory key')
        latents={name:LatentPortMemory(*(tensors['latent.'+name+'.'+label].to(device)
            for label in ('mu','logvar','mean','scale'))) for name in latent_counts}
        self.validate_latents(latents)
        return PhysicalMemoryUnit(state,graph,keys,latents,load_sequences(self.model,sequence_meta,tensors),
            load_address(self.model,address_meta,tensors))
