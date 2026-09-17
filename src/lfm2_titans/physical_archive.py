"""Query neural addresses first; load only selected complete physical weights."""
import heapq,json,math,os,re,time,uuid
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors import safe_open
from .episodic_adapters import AdapterMixture
from .multiport_connectome import MemoryPage,MountedState
from .sequence_memory import prepare_ordered_inputs
from .memory_address import memory_hash
from .native_concepts import load_native_address,native_shapes,sparse_similarity


class PhysicalMemoryArchive:
    """Append-only disk catalogue with explicitly bounded active composition.

    Query scans sparse native concept addresses on CPU or GPU. Titans weights
    and persistent VAE codes load only after routing. Total units have no software count limit; disk
    capacity, index quality, residency, and latency remain measurable costs.
    """
    def __init__(self,model,directory,*,max_readers=1,max_resident_units=8):
        if type(max_resident_units) is not int or max_resident_units<1:raise ValueError('positive resident unit bound required')
        self.max_resident_units=max_resident_units
        self.model=model;self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=True)
        self.session=model.open_physical_memory_session(max_readers=max_readers)
        self.concept_lens=self.session.concept_lens
        self.identity=json.loads(json.dumps(self.session.bank._identity()))
        self.key_dims={p.name:p.feature_dim for p in model.memory.specs}

    def _path(self,unit_id):
        if not isinstance(unit_id,str) or not re.fullmatch(r'[0-9]{20}_[a-f0-9]{32}',unit_id):raise ValueError('invalid physical unit ID')
        return self.root/(unit_id+'.safetensors')

    def append(self,*,start_new=True):
        with self.session._writes:
            unit_id=f'{time.time_ns():020d}_{uuid.uuid4().hex}'
            result=self.session.seal(self._path(unit_id),start_new=start_new)
            return dict(result,unit_id=unit_id)

    def append_concepts(self,lens,*,start_new=True,**options):
        """Seal model-derived sparse coordinates; no tokenizer or JSON output."""
        with self.session._writes:
            address=self.session.extract_concepts(lens,**options)
            return dict(self.append(start_new=start_new),address=address)

    def query_native_concepts(self,inputs,*,lens,top_k=8,device='cpu',encoded_query=None,before_ns=None,min_score=None):
        if lens.model is not self.model:raise ValueError('concept lens model mismatch')
        if type(top_k) is not int or top_k<1:raise ValueError('positive candidate count required')
        if min_score is not None and not math.isfinite(min_score):raise ValueError('finite concept score threshold required')
        query=lens.encode_inputs(inputs) if encoded_query is None else encoded_query
        best=[];scanned=legacy=address_bytes=0
        for unit_id,metadata,f in self._metadata_records():
            if before_ns is not None and metadata['created_ns']>before_ns:continue
            meta=metadata.get('address')
            if meta is None or meta.get('version')!=2:legacy+=1;continue
            if meta['lens_id']!=lens.lens_id:raise ValueError('archive uses a different concept lens; re-extract addresses')
            shapes=native_shapes(self.model,meta);tensors={}
            for key,shape in shapes.items():
                if f.get_slice(key).get_shape()!=shape:raise ValueError('invalid native concept index shape')
                tensors[key]=f.get_tensor(key);address_bytes+=tensors[key].numel()*tensors[key].element_size()
            address=load_native_address(self.model,meta,tensors)
            reference=memory_hash(self.identity,metadata['checksum'],meta)
            if metadata.get('memory_hash')!=reference:raise ValueError('native concept hash mismatch')
            score=sparse_similarity(query,address.codes,device=device);scanned+=1
            if min_score is not None and score<min_score:continue
            row=dict(unit_id=unit_id,memory_hash=reference,score=score,recorded_ns=address.recorded_ns,
                event_ns=address.event_ns,logical_time=address.logical_time,parents=list(address.parents))
            item=(score,unit_id,row)
            if len(best)<top_k:heapq.heappush(best,item)
            elif item[:2]>best[0][:2]:heapq.heapreplace(best,item)
        return dict(candidates=[r for _,_,r in sorted(best,reverse=True)],scanned_units=scanned,
            legacy_units_not_indexed=legacy,coverage_complete=legacy==0,address_bytes_read=address_bytes,physical_weight_tensors_loaded=0,
            representation='native_jacobian_sparse_nonnegative',text_generation_calls=0,
            score_is_calibrated_probability=False,device=device)

    def _metadata_records(self):
        with os.scandir(self.root) as entries:
            for entry in entries:
                if not entry.is_file() or not re.fullmatch(r'[0-9]{20}_[a-f0-9]{32}\.safetensors',entry.name):continue
                with safe_open(entry.path,framework='pt',device='cpu') as f:
                    raw=(f.metadata() or {}).get('physical_unit','')
                    if len(raw)>16384:raise ValueError('oversized physical unit metadata')
                    metadata=json.loads(raw)
                    if metadata.get('identity')!=self.identity:raise ValueError('incompatible physical archive')
                    yield entry.name[:-12],metadata,f

    def resolve_hash(self,reference):
        if not isinstance(reference,str) or not re.fullmatch('[a-f0-9]{64}',reference):
            raise ValueError('a full SHA-256 memory reference is required')
        found=[]
        for unit_id,metadata,_ in self._metadata_records():
            actual=memory_hash(self.identity,metadata['checksum'],metadata.get('address'))
            if metadata.get('memory_hash',actual)!=actual:raise ValueError('invalid archived address hash')
            if actual==reference:found.append(unit_id)
        if not found:raise KeyError('memory hash not found')
        # Identical immutable content can have several archived copies. This
        # selects a stable path; it never deletes or merges those copies.
        return dict(memory_hash=reference,unit_id=min(found),copies=found)

    def mount_hash_async(self,reference):return self.mount_async(self.resolve_hash(reference)['unit_id'])

    def encode_query(self,inputs):
        with self.session.pin():return self.concept_lens.encode_inputs(inputs)

    def query(self,inputs,**options):
        mode=options.pop('index_mode','exact')
        if mode not in ('scan','exact'):raise ValueError('native concept index supports exact scan')
        result=self.query_native_concepts(inputs,lens=self.concept_lens,**options)
        if not result['coverage_complete']:raise ValueError('incompatible memory; use offline conversion')
        return dict(result,matches=result['candidates'],algorithm='native sparse concept late interaction')

    def generate(self,query_inputs,*,generation_inputs,top_k=1,index_options=None,encoded_query=None,**generation_options):
        if type(top_k) is not int or not 1<=top_k<=self.max_resident_units:
            raise ValueError('top_k exceeds configured resident unit capacity')
        forbidden={'memory_state','physical_memory_state','latent_memory_state','memory_write','use_memory','memory_read_mode','recall_mode'}
        if forbidden.intersection(generation_inputs) or forbidden.intersection(generation_options):
            raise ValueError('archive owns one ordered dual-memory reader')
        keys=self.encode_query(query_inputs) if encoded_query is None else encoded_query
        route=self.query({},encoded_query=keys,top_k=top_k,**(index_options or {}))
        if not route['matches']:raise LookupError('no physical memory matches the query')
        with self.session.pin() as (active,_),torch.no_grad():
            units=[self.session.bank.load_unit(self._path(m['unit_id'])) for m in route['matches']]
            adapters=AdapterMixture(tuple((unit.adapters,1.) for unit in units))
            graph=MountedState(active.graph,tuple(MemoryPage(m['unit_id'],unit.graph,1.)
                for m,unit in zip(route['matches'],units)),active_weight=0.)
            ordered=sorted(zip(route['matches'],units),key=lambda pair:pair[0]['unit_id'])
            prepared=prepare_ordered_inputs(self.model,tuple(s for _,u in ordered for s in u.sequences),
                generation_inputs,max_new_tokens=generation_options.get('max_new_tokens',generation_inputs.get('max_new_tokens')),
                max_length=generation_options.get('max_length',generation_inputs.get('max_length')))
            with self.session.bank.use(adapters):
                tokens=self.model.generate(**prepared,**generation_options,memory_state=graph)
        return dict(query=route,tokens=tokens,loaded_units=[m['unit_id'] for m in route['matches']],
                    composition='independently evaluated graph and FFN memories',latent_recall='ordered',generation_calls=1)

    def mount_async(self,unit_id):return self.session.mount_async(self._path(unit_id))

    def unload(self):return self.session.unload()
    def close(self):self.session.close()
