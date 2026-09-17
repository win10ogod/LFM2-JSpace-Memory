"""SFT LoRA across native modules and the complete physical memory module.

Memory writers directly consume projection.weight, so a forward-only PEFT
Linear wrapper would silently omit its adapter from those writers. These
projections expose their effective weight as well as an efficient LoRA forward.
Only low-rank factors train; the PT checkpoint and physical fast-state size stay
intact. Extra factors have an explicit checkpoint alongside native PEFT LoRA.
"""
import json
import math
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from safetensors.torch import save_file,load_file


_custom_disabled=ContextVar('lfm2_custom_lora_disabled',default=False)
_effective_weights=ContextVar('lfm2_effective_lora_weights',default=None)


@contextmanager
def shared_effective_weights():
    """Share exact expanded tensors within a forward, never across updates."""
    if _effective_weights.get() is not None:
        yield
        return
    token=_effective_weights.set({})
    try:yield
    finally:_effective_weights.reset(token)


def warm_effective_weights(model):
    """Expand outside checkpointed blocks so recomputation sees identical ops."""
    for item in getattr(model,'_sft_custom_lora_spec',{}).get('targets',[]):
        parent,name=_split(model,item['path'])
        value=getattr(parent,name)
        if item['kind']=='linear':value=value.weight


class EffectiveWeightContext:
    def __init__(self,snapshot):self.snapshot=snapshot;self.stack=[]
    def __enter__(self):self.stack.append(_effective_weights.set(self.snapshot))
    def __exit__(self,*exc):_effective_weights.reset(self.stack.pop())


def checkpoint_weight_contexts():
    snapshot=_effective_weights.get()
    return EffectiveWeightContext(snapshot),EffectiveWeightContext(snapshot)


@contextmanager
def disable_custom_lora():
    """Thread-local original decoder path; trained reader factors stay loaded."""
    token=_custom_disabled.set(True)
    try:yield
    finally:_custom_disabled.reset(token)


class WeightLoRALinear(nn.Module):
    def __init__(self,base,rank,alpha):
        super().__init__();self.base_layer=base;self.rank=rank;self.alpha=alpha
        self.in_features=base.in_features;self.out_features=base.out_features
        self.adapter_A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=base.weight.dtype))
        self.adapter_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.adapter_A,a=math.sqrt(5));base.requires_grad_(False)

    @property
    def weight(self):
        if _custom_disabled.get():return self.base_layer.weight
        cache=_effective_weights.get();key=(id(self),torch.is_grad_enabled(),self.adapter_A._version,self.adapter_B._version)
        if cache is not None and key in cache:return cache[key]
        with torch.autocast(device_type=self.adapter_A.device.type,enabled=False):
            value=self.base_layer.weight+(self.adapter_B@self.adapter_A)*(self.alpha/self.rank)
        if cache is not None:cache[key]=value
        return value

    @property
    def bias(self):return self.base_layer.bias

    def forward(self,x):
        if _custom_disabled.get():return self.base_layer(x)
        return self.base_layer(x)+F.linear(F.linear(x,self.adapter_A),self.adapter_B)*(self.alpha/self.rank)


class TensorLoRA(nn.Module):
    """LoRA for sparse edge/channel priors, scalar/vector controls and conv kernels."""
    def __init__(self,original,rank,alpha):
        super().__init__();self.shape=tuple(original.shape);self.rank=rank;self.alpha=alpha
        rows=original.shape[0] if original.ndim else 1
        columns=original.numel()//rows
        self.adapter_A=nn.Parameter(torch.empty(rank,columns,device=original.device,dtype=torch.float32))
        self.adapter_B=nn.Parameter(torch.zeros(rows,rank,device=original.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.adapter_A,a=math.sqrt(5))

    def forward(self,original):
        if _custom_disabled.get():return original
        cache=_effective_weights.get();key=(id(self),torch.is_grad_enabled(),self.adapter_A._version,self.adapter_B._version,original._version)
        if cache is not None and key in cache:return cache[key]
        with torch.autocast(device_type=original.device.type,enabled=False):
            delta=(self.adapter_B@self.adapter_A)*(self.alpha/self.rank)
            value=original+delta.reshape(self.shape).to(original.dtype)
        if cache is not None:cache[key]=value
        return value


def _split(model,path):
    parent,_,name=path.rpartition('.')
    return model.get_submodule(parent) if parent else model,name


def install_custom_lora(native,spec):
    existing=getattr(native,'_sft_custom_lora_spec',None)
    if existing is not None:
        if existing!=spec:raise ValueError('custom SFT adapter structure differs from checkpoint')
        return
    for item in spec['targets']:
        parent,name=_split(native,item['path'])
        if item['kind']=='linear':
            base=getattr(parent,name)
            if not isinstance(base,nn.Linear):raise TypeError(f'expected original linear: {item["path"]}')
            setattr(parent,name,WeightLoRALinear(base,item['rank'],item['alpha']))
        else:
            original=getattr(parent,name)
            if list(original.shape)!=item['shape']:raise ValueError(f'LoRA tensor shape changed: {item["path"]}')
            original.requires_grad_(False)
            parametrize.register_parametrization(parent,name,TensorLoRA(original,item['rank'],item['alpha']))
    native._sft_custom_lora_spec=spec


def custom_factors(native):
    return {n:p for n,p in native.named_parameters() if n.endswith(('.adapter_A','.adapter_B'))}


def apply_sft_lora(native,factory,rank=16,memory_edge_rank=2):
    native_targets=[n for n,m in native.named_modules() if n.startswith('model.') and isinstance(m,nn.Linear)]
    fine=factory.FinetuningArguments(stage='sft',finetuning_type='lora',lora_rank=rank,lora_alpha=rank*2,
        lora_dropout=0.,lora_target=','.join(native_targets),additional_target=None,
        freeze_vision_tower=False,freeze_multi_modal_projector=False,freeze_language_model=False,disable_shuffling=True)
    model_args=factory.ModelArguments(model_name_or_path=str(native.config._name_or_path or factory.WORK/'initial'),trust_remote_code=True)
    model=factory.init_adapter(native.config,native,model_args,fine,is_trainable=True)
    native=model.get_base_model();spec=dict(version=1,targets=[])
    linear_paths=[]
    for name,module in list(native.memory.named_modules()):
        if isinstance(module,nn.Linear):
            linear_paths.append(name)
            spec['targets'].append(dict(path='memory.'+name,kind='linear',rank=rank,alpha=rank*2))
    for name,parameter in list(native.memory.named_parameters()):
        if any(name.startswith(path+'.') for path in linear_paths):continue
        maximum=min(parameter.shape) if parameter.ndim==2 else 1
        r=min(memory_edge_rank,maximum) if name=='slow_weights' else 1
        spec['targets'].append(dict(path='memory.'+name,kind='tensor',shape=list(parameter.shape),rank=r,alpha=r*2))
    if hasattr(native,'physical_memory'):
        for name,parameter in list(native.physical_memory.named_parameters()):
            r=min(memory_edge_rank,min(parameter.shape)) if parameter.ndim==2 else 1
            spec['targets'].append(dict(path='physical_memory.'+name,kind='tensor',shape=list(parameter.shape),rank=r,alpha=r*2))
    if hasattr(native,'dream_memory'):
        dream_linear=[]
        for name,module in native.dream_memory.named_modules():
            if isinstance(module,nn.Linear):
                dream_linear.append(name)
                spec['targets'].append(dict(path='dream_memory.'+name,kind='linear',rank=rank,alpha=rank*2))
        for name,parameter in native.dream_memory.named_parameters():
            if any(name.startswith(path+'.') for path in dream_linear):continue
            r=min(memory_edge_rank,min(parameter.shape)) if parameter.ndim==2 else 1
            spec['targets'].append(dict(path='dream_memory.'+name,kind='tensor',shape=list(parameter.shape),rank=r,alpha=r*2))
    # Cached LFM short convolution calls F.conv1d(..., conv.weight), so adapt
    # the tensor itself, including that native cached execution path.
    for name,module in list(native.named_modules()):
        if name.startswith('model.') and isinstance(module,(nn.Conv1d,nn.Conv2d)):
            shape=list(module.weight.shape);width=module.weight.numel()//shape[0]
            r=min(rank,max(1,min(shape[0],width)//2))
            spec['targets'].append(dict(path=name+'.weight',kind='tensor',shape=shape,rank=r,alpha=r*2))
    original_memory_parameters=sum(p.numel() for p in native.memory.parameters())
    install_custom_lora(native,spec)
    custom_ids={id(p) for p in custom_factors(native).values()}
    for name,p in model.named_parameters():
        expected='.lora_' in name or id(p) in custom_ids
        if p.requires_grad!=expected:raise RuntimeError(f'unexpected SFT trainability: {name}')
    if model.peft_config['default'].modules_to_save:raise RuntimeError('SFT must not fully unfreeze memory')
    counts={}
    for name,p in native.named_parameters():
        if not p.requires_grad:continue
        group=('dream' if name.startswith('dream_memory.') else 'memory' if name.startswith(('memory.','physical_memory.')) else 'vision' if '.vision_tower.' in name
               else 'projector' if '.multi_modal_projector.' in name else 'language')
        counts[group]=counts.get(group,0)+p.numel()
    audit=dict(method='native PEFT LoRA + weight-aware memory LoRA + native short-convolution tensor LoRA',
        native_rank=rank,memory_projection_rank=rank,memory_edge_rank=memory_edge_rank,
        scalar_vector_control_rank=1,full_memory_unfreezing=False,trainable_by_group=counts,
        original_memory_parameters=original_memory_parameters,
        physical_fast_scalars=native.memory.slow_weights.numel(),physical_fast_state_low_rank=False,
        custom_spec=spec,normalization_embedding_and_native_biases='frozen, as in standard LoRA',
        scope='all native linear modules and convolution kernels; all graph and physical FFN memory parameters have LoRA factors',
        physical_ffn_architecture=native.config.physical_memory,dream_architecture=native.config.dream_memory)
    return model,fine,model_args,native_targets,audit


def save_custom_lora(model,directory):
    native=model.get_base_model() if hasattr(model,'get_base_model') else model
    spec=getattr(native,'_sft_custom_lora_spec',None)
    if spec is None:return
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    save_file({n:p.detach().cpu().contiguous() for n,p in custom_factors(native).items()},str(directory/'custom_lora.safetensors'))
    (directory/'custom_lora.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2))


def load_custom_lora(model,directory,*,required=False,trainable=None):
    directory=Path(directory);path=directory/'custom_lora.json'
    if not path.exists():
        if required:raise FileNotFoundError(path)
        return
    native=model.get_base_model() if hasattr(model,'get_base_model') else model
    install_custom_lora(native,json.loads(path.read_text()))
    saved=load_file(str(directory/'custom_lora.safetensors'));expected=custom_factors(native)
    if saved.keys()!=expected.keys():raise ValueError('custom LoRA checkpoint keys differ')
    with torch.no_grad():
        for name,p in expected.items():
            if saved[name].shape!=p.shape:raise ValueError(f'custom LoRA checkpoint shape differs: {name}')
            p.copy_(saved[name].to(p))
            if trainable is not None:p.requires_grad_(trainable)


def merge_sft_lora(model):
    native=model.get_base_model()
    spec=getattr(native,'_sft_custom_lora_spec',None)
    if spec is not None:
        for item in spec['targets']:
            parent,name=_split(native,item['path'])
            with torch.no_grad():
                if item['kind']=='linear':
                    layer=getattr(parent,name);weight=layer.weight
                    if not torch.isfinite(weight).all():raise FloatingPointError('nonfinite merged memory projection')
                    layer.base_layer.weight.copy_(weight);setattr(parent,name,layer.base_layer)
                else:
                    if not torch.isfinite(getattr(parent,name)).all():raise FloatingPointError('nonfinite merged tensor LoRA')
                    parametrize.remove_parametrizations(parent,name,leave_parametrized=True)
        del native._sft_custom_lora_spec
    return model.merge_and_unload(safe_merge=True)
