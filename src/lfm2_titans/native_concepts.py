"""Model-native sparse concept addresses; no language generation or JSON tags.

Coordinates are nonnegative fits to a calibrated Jacobian-lens dictionary.
Vocabulary IDs identify dictionary directions, not user-assigned categories.
All observation positions are retained. Text decoding is display-only.
"""
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,re,time,threading
import torch
from torch.nn import functional as F
from safetensors.torch import load_file


@dataclass(frozen=True)
class SparseConcepts:
    indices: torch.Tensor
    coefficients: torch.Tensor
    residual_fraction: torch.Tensor

    def detach(self):
        return SparseConcepts(*(v.detach().cpu().contiguous() for v in
            (self.indices,self.coefficients,self.residual_fraction)))


@dataclass(frozen=True)
class NativeConceptAddress:
    codes: dict
    checkpoint_id: str
    lens_id: str
    vocab_size: int
    recorded_ns: int
    logical_time: int
    parents: tuple=()
    event_ns: int | None=None

    def detach(self):
        return NativeConceptAddress({n:c.detach() for n,c in self.codes.items()},self.checkpoint_id,
            self.lens_id,self.vocab_size,self.recorded_ns,self.logical_time,self.parents,self.event_ns)

    def metadata(self):
        return dict(version=2,origin='native_jacobian_sparse_nonnegative',checkpoint_id=self.checkpoint_id,
            lens_id=self.lens_id,vocab_size=self.vocab_size,recorded_ns=self.recorded_ns,
            logical_time=self.logical_time,parents=list(self.parents),event_ns=self.event_ns,
            code_shapes={n:list(c.indices.shape) for n,c in self.codes.items()},
            position_semantics='all native contextual observation positions in order',
            text_labels_required=False)


def validate_native_address(model,address):
    if address.checkpoint_id!=model.config.memory_checkpoint_id:raise ValueError('concept checkpoint mismatch')
    if not re.fullmatch('[a-f0-9]{64}',address.lens_id):raise ValueError('invalid concept lens identity')
    if address.vocab_size!=model.config.text_config.vocab_size:raise ValueError('concept vocabulary mismatch')
    if any(type(t) is not int or t<0 for t in (address.recorded_ns,address.logical_time)):
        raise ValueError('invalid concept time')
    if address.event_ns is not None and (type(address.event_ns) is not int or address.event_ns<0):
        raise ValueError('invalid concept event time')
    if any(not re.fullmatch('[a-f0-9]{64}',p) for p in address.parents):raise ValueError('invalid causal parent')
    if not address.codes or set(address.codes)-set(model.config.language_ports.values()):
        raise ValueError('invalid native concept ports')
    for c in address.codes.values():
        if c.indices.ndim!=2 or min(c.indices.shape)<1 or c.indices.dtype!=torch.int32:
            raise ValueError('invalid native concept indices')
        if (c.indices<0).any() or (c.indices>=address.vocab_size).any():raise ValueError('concept outside vocabulary')
        if c.coefficients.shape!=c.indices.shape or c.residual_fraction.shape!=(len(c.indices),):
            raise ValueError('invalid native concept shapes')
        if any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in (c.coefficients,c.residual_fraction)):
            raise ValueError('invalid native concept values')
        if (c.coefficients<0).any() or (c.residual_fraction<0).any():raise ValueError('negative concept coordinate')
        ordered=c.indices.sort(dim=-1).values
        if (ordered[:,1:]==ordered[:,:-1]).any():raise ValueError('duplicate sparse concept coordinate')


def native_tensors(address):
    return {'native_concept.'+n+'.'+k:getattr(c,k).detach().cpu().contiguous()
        for n,c in address.codes.items() for k in ('indices','coefficients','residual_fraction')}


def native_shapes(model,metadata):
    if (metadata.get('origin')!='native_jacobian_sparse_nonnegative'
            or metadata.get('checkpoint_id')!=model.config.memory_checkpoint_id
            or metadata.get('vocab_size')!=model.config.text_config.vocab_size
            or not re.fullmatch('[a-f0-9]{64}',metadata.get('lens_id',''))):
        raise ValueError('incompatible native concept address')
    result={};shapes=metadata.get('code_shapes',{})
    if not shapes or set(shapes)-set(model.config.language_ports.values()):raise ValueError('invalid concept ports')
    for name,shape in shapes.items():
        if len(shape)!=2 or any(type(v) is not int or v<1 for v in shape) or shape[1]>metadata['vocab_size']:
            raise ValueError('invalid sparse concept shape')
        for key in ('indices','coefficients','residual_fraction'):
            result['native_concept.'+name+'.'+key]=shape if key!='residual_fraction' else shape[:1]
    return result


def load_native_address(model,metadata,tensors):
    codes={n:SparseConcepts(*(tensors['native_concept.'+n+'.'+k] for k in
        ('indices','coefficients','residual_fraction'))) for n in metadata['code_shapes']}
    result=NativeConceptAddress(codes,metadata['checkpoint_id'],metadata['lens_id'],metadata['vocab_size'],
        metadata['recorded_ns'],metadata['logical_time'],tuple(metadata['parents']),metadata.get('event_ns'))
    validate_native_address(model,result);return result


@torch.no_grad()
def sparse_nonnegative(features,dictionary,*,sparsity=16,iterations=48):
    """Positive matching pursuit with a projected NNLS refit on each support.

    Dictionary rows must have unit norm. Support choices come from native
    residual correlations. No concept words or task-specific labels enter.
    This is an approximate sparse nonnegative fit, not a unique decomposition.
    """
    x=features.float();d=dictionary
    if not 1<=sparsity<=len(d) or iterations<1:raise ValueError('invalid sparse fitting configuration')
    if not torch.isfinite(x).all() or not torch.isfinite(d).all():raise ValueError('nonfinite concept features')
    n=len(x);indices=torch.empty(n,sparsity,dtype=torch.long,device=x.device)
    coefficients=torch.zeros(n,0,device=x.device);residual=x.clone()
    rows=torch.arange(n,device=x.device)
    for step in range(sparsity):
        scores=residual@d.T
        if step:scores.scatter_(1,indices[:,:step],-torch.inf)
        indices[:,step]=scores.argmax(dim=-1)
        vectors=d[indices[:,:step+1]]
        gram=vectors@vectors.transpose(1,2)
        targets=(vectors*x[:,None]).sum(-1)
        coefficients=F.pad(coefficients,(0,1))
        rate=gram.abs().sum(-1).amax(-1).clamp_min(1e-8).reciprocal()
        for _ in range(iterations):
            gradient=targets-torch.bmm(gram,coefficients[:,:,None]).squeeze(-1)
            coefficients=(coefficients+rate[:,None]*gradient).clamp_min(0)
        residual=x-(coefficients[:,:,None]*vectors).sum(1)
    fraction=residual.square().sum(-1)/x.square().sum(-1).clamp_min(1e-12)
    return SparseConcepts(indices.to(torch.int32),coefficients.float(),fraction.float())


class NativeConceptLens:
    """Frozen per-checkpoint dictionary, loaded separately from model weights."""
    def __init__(self,model,directory=None,*,sparsity=16,feature_chunk=32,cache_layers=3):
        self.model=model
        if directory is None:
            spec=model.config.concept_memory
            if not spec:raise ValueError('this checkpoint has no configured native concept lens')
            from transformers.utils.hub import cached_file
            filename=cached_file(model.config._name_or_path,spec['lens_file'])
            report=dict(protocol=spec['protocol'],body_fixed=True,native_outputs_identical=True)
            raw=Path(filename).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=spec['lens_sha256']:raise ValueError('native concept lens checksum mismatch')
            sparsity=spec['sparsity'];feature_chunk=spec.get('feature_chunk',feature_chunk)
        else:
            root=Path(directory);filename=root/'jacobians.safetensors';raw=Path(filename).read_bytes()
            report=json.loads((root/'result.json').read_text())
        if report['protocol']['checkpoint_id']!=model.config.memory_checkpoint_id:
            raise ValueError('Jacobian lens belongs to another checkpoint')
        if not report['body_fixed'] or not report['native_outputs_identical']:
            raise ValueError('lens diagnostic did not preserve the native model')
        self.matrices=load_file(str(filename));self.source_path=Path(filename)
        self.lens_id=hashlib.sha256(raw).hexdigest();self.sparsity=sparsity
        if feature_chunk<1 or cache_layers<1:raise ValueError('invalid concept scheduling')
        self.feature_chunk=feature_chunk;self.cache_layers=cache_layers;self.cache=OrderedDict()
        self._cache_lock=threading.RLock()
        self._parameter_versions={n:p._version for n,p in model.named_parameters()}
        self.depths={int(n):port for n,port in model.config.language_ports.items()}
        if any(str(n) not in self.matrices for n in self.depths):raise ValueError('lens lacks a memory depth')
        self.protocol=report['protocol']

    @torch.no_grad()
    def directions(self,depth,indices):
        model=self.model;device=model.lm_head.weight.device
        rows=model.lm_head.weight[indices].float()
        rows=rows*model.model.language_model.embedding_norm.weight.float()
        return rows@self.matrices[str(depth)].to(device)

    @torch.no_grad()
    def dictionary(self,depth):
        with self._cache_lock:return self._dictionary(depth)

    def _dictionary(self,depth):
        if depth in self.cache:
            self.cache.move_to_end(depth);return self.cache[depth]
        while len(self.cache)>=self.cache_layers:self.cache.popitem(last=False)
        weight=self.model.lm_head.weight;vocab,width=weight.shape
        dictionary=torch.empty(vocab,width,device=weight.device,dtype=torch.float32)
        for start in range(0,vocab,4096):
            idx=torch.arange(start,min(start+4096,vocab),device=weight.device)
            dictionary[start:start+len(idx)]=F.normalize(self.directions(depth,idx),dim=-1)
        self.cache[depth]=dictionary;return dictionary

    @torch.no_grad()
    def encode_features(self,features):
        result={}
        for depth,name in self.depths.items():
            values=features[name]
            if not len(values):raise ValueError('empty native concept observation')
            dictionary=self.dictionary(depth);chunks=[]
            for start in range(0,len(values),self.feature_chunk):
                chunks.append(sparse_nonnegative(values[start:start+self.feature_chunk].to(dictionary.device),
                    dictionary,sparsity=self.sparsity).detach())
            result[name]=SparseConcepts(*(torch.cat([getattr(c,k) for c in chunks])
                for k in ('indices','coefficients','residual_fraction')))
        return result

    @torch.no_grad()
    def encode_inputs(self,inputs):
        from .concept_capture import capture_native_features
        if (self.model.training or self.protocol['checkpoint_id']!=self.model.config.memory_checkpoint_id
                or any(p.requires_grad or p._version!=self._parameter_versions[n] for n,p in self.model.named_parameters())):
            raise ValueError('concept lens requires its unchanged frozen checkpoint; recalibrate after training')
        if any(k.startswith('memory_') or k in ('use_memory','physical_memory_state','past_key_values') for k in inputs):
            raise ValueError('native concept encoder owns memory and cache scope')
        with capture_native_features(self.model,inputs.get('attention_mask')) as features:
            self.model(**dict(inputs,use_memory=False,use_cache=False,logits_to_keep=1))
        return self.encode_features(features)

    @torch.no_grad()
    def address_memory(self,state,*,event_ns=None,logical_time=0,parents=()):
        from .sequence_memory import decode_segment
        if not state.sequences:raise ValueError('native concept extraction requires preserved ordered observations')
        length=sum(s.count for s in state.sequences)
        if length>self.model.config.text_config.max_position_embeddings:
            raise ValueError('observation exceeds native concept context; no silent truncation')
        features=torch.cat([decode_segment(self.model,s) for s in state.sequences])
        features=features.to(self.model.get_input_embeddings().weight)[None]
        codes=self.encode_inputs(dict(inputs_embeds=features))
        address=NativeConceptAddress(codes,self.model.config.memory_checkpoint_id,self.lens_id,
            self.model.config.text_config.vocab_size,time.time_ns(),logical_time,tuple(parents),event_ns)
        validate_native_address(self.model,address);return address

    def close(self):
        with self._cache_lock:self.cache.clear()


def sparse_similarity(query,stored,*,device='cpu',row_chunk=128):
    """Late interaction on actual sparse coefficients; no decoded text.

    Mean over valid query positions of their best matching memory position,
    then mean over ports. All stored positions are visited in bounded chunks.
    """
    scores=[]
    for name,q in query.items():
        if name not in stored:continue
        m=stored[name];qi=q.indices.to(device).long();qc=q.coefficients.to(device)
        mi=m.indices.to(device).long();mc=m.coefficients.to(device)
        active=torch.unique(torch.cat((qi.flatten(),mi.flatten())),sorted=True)
        qdense=torch.zeros(len(qi),len(active),device=device)
        qdense.scatter_(1,torch.searchsorted(active,qi),qc)
        qdense=F.normalize(qdense,dim=-1);best=torch.zeros(len(qi),device=device)
        for start in range(0,len(mi),row_chunk):
            mdense=torch.zeros(min(row_chunk,len(mi)-start),len(active),device=device)
            mdense.scatter_(1,torch.searchsorted(active,mi[start:start+row_chunk]),mc[start:start+row_chunk])
            best=torch.maximum(best,(qdense@F.normalize(mdense,dim=-1).T).amax(-1))
        valid=qc.square().sum(-1)>0
        if valid.any():scores.append(best[valid].mean())
    return float(torch.stack(scores).mean()) if scores else 0.
