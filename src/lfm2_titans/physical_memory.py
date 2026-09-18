"""Bounded, transactional runtime for graph + direct FFN physical memory."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import threading
import random,time
import uuid
from pathlib import Path
import torch
from torch.nn import functional as F
from .episodic_adapters import EpisodicAdapterBank,PhysicalMemoryUnit
from .sequence_memory import capture_native_inputs,encode_sequence,prepare_ordered_inputs,decode_segment


class PhysicalMemorySession:
    """One resident unit, one staging load, and a bounded number of readers.

    All weights are caller-owned; a generate call pins one complete unit.
    Writes publish a new version after gradients finish. An unsaved version
    cannot be replaced by mounting another page. Archives are append-only.
    Routing is deliberately external: mounting a supplied ID is not evidence
    that the model can autonomously choose the right historical memory.
    """
    def __init__(self,model,*,rank=16,max_readers=1,_initial_unit=None,concept_lens=None):
        if model.training:raise ValueError('physical sessions require model.eval()')
        if type(max_readers) is not int or max_readers<1:raise ValueError('positive reader bound required')
        bank=getattr(model,'_episodic_adapter_bank',None)
        if bank is None:bank=EpisodicAdapterBank(model,rank=rank)
        if bank.rank!=rank:raise ValueError('another adapter rank is already installed')
        self.model,self.bank=model,bank
        self.concept_lens=concept_lens
        self._lock=threading.Lock();self._writes=threading.RLock()
        self._readers=threading.BoundedSemaphore(max_readers)
        self._mount_slot=threading.BoundedSemaphore(1)
        self._loader=ThreadPoolExecutor(max_workers=1,thread_name_prefix='physical-unit-loader')
        self._state=bank.initial_unit() if _initial_unit is None else _initial_unit
        self.generation=0;self.dirty=False;self.closed=False
        self._dream_checked_generation=0
        self.last_memory_action=None;self.last_consolidation=None
        self._history_root=None;self._consolidation_parent=None

    @contextmanager
    def pin(self):
        with self._readers:
            with self._lock:
                if self.closed:raise RuntimeError('session closed')
                if self.model.training or any(p.requires_grad for p in self.model.parameters()):
                    raise ValueError('the inference body must remain frozen and in eval mode')
                if self.model.config.memory_checkpoint_id!=self.bank.checkpoint_id:
                    raise ValueError('body checkpoint changed; open a new memory runtime')
                state,generation=self._state,self.generation
            yield state,generation

    def generate(self,**inputs):
        if 'recall_mode' in inputs or 'memory_read_mode' in inputs:
            raise ValueError('this architecture has one ordered dual-memory reader')
        if 'memory_state' in inputs or inputs.get('memory_write'):
            raise ValueError('session owns its state; use observe to write')
        self._consolidate_pending()
        from .autonomous_memory import autonomous_generate
        query={k:inputs.pop(k) for k in ('input_ids','attention_mask') if k in inputs}
        with self.pin() as (state,_),torch.no_grad():
            tokens,_,receipt=autonomous_generate(self.model,self.bank,state.adapters,state.graph,
                state.sequences,query,**inputs)
        with self._lock:self.last_memory_action=receipt
        return tokens

    def _consolidate_pending(self):
        """Ordinary seal/generate calls service writes; callers need no Dream API."""
        from .autonomous_memory import autonomous_consolidate
        with self._writes:
            if not self.dirty or self._dream_checked_generation==self.generation:return
            with self.pin() as (state,generation):
                if not state.sequences:return
                candidate,receipt=autonomous_consolidate(self.model,state)
            if receipt['accepted']:
                if self._history_root is None:
                    # A standalone live session acquires its persistent root
                    # at seal(). Never discard the pre-Dream physical weights.
                    receipt.update(accepted=False,deferred_until_seal=True)
                else:
                    original=state
                    if self.concept_lens is not None and original.address is None:
                        address=self.concept_lens.address_memory(original,
                            logical_time=original.graph.commits+original.adapters.commits)
                        original=replace(original,address=address)
                    path=self._history_root/f'{time.time_ns():020d}_{uuid.uuid4().hex}.safetensors'
                    history=self.bank.save_unit(original,path)
                    receipt['preserved_previous_unit']=history
                    self._consolidation_parent=history['memory_hash']
            with self._lock:
                if self.generation!=generation:raise RuntimeError('memory changed during autonomous consolidation')
                if receipt['accepted']:self._state=candidate;self.generation+=1
                if not receipt.get('deferred_until_seal'):self._dream_checked_generation=self.generation
                self.last_consolidation=receipt

    def observe(self,*,preserve_sequence=None,sequence_chunk_size=128,**inputs):
        if any(k in inputs for k in ('memory_state','memory_write')):raise ValueError('session owns write options')
        if preserve_sequence is not None and type(preserve_sequence) is not bool:raise ValueError('preserve_sequence must be boolean or None')
        with self._writes,self.pin() as (state,_),self.bank.use(state.adapters),torch.no_grad():
            # Once a unit records ordered observations, subsequent ordinary
            # writes keep doing so. A default must not silently omit history.
            if preserve_sequence is False:raise ValueError('ordered observations are required by this architecture')
            with capture_native_inputs(self.model,inputs.get('attention_mask')) as captured:
                out=self.model(**dict(inputs,memory_capture_keys=True),memory_state=state.graph,latent_memory_state=state.latents,memory_write=True)
            sequences=getattr(state,'sequences',())
            if len(captured)!=1:raise ValueError('expected one native multimodal observation')
            sequences=sequences+encode_sequence(self.model,captured[0],chunk_size=sequence_chunk_size)
            keys=dict(state.neural_keys or {})
            for name,key in out.memory_keys.items():keys[name]=F.normalize(keys.get(name,torch.zeros_like(key))+key,dim=0)
            replacement=replace(state,graph=out.memory_state.detach(),neural_keys=keys,latents=out.latent_memory_state,sequences=sequences,address=None)
            with self._lock:self._state=replacement;self.generation+=1;self.dirty=True
            return dict(out.memory_receipt,mount_generation=self.generation,ordered_features=sum(s.count for s in sequences),
                        ordered_bytes=sum(s.bytes for s in sequences))

    def learn(self,**inputs):
        """One supervised, observation-derived write; probes must stay external.

        Labels must come from information actually observed. This method does
        not claim that arbitrary free-form text can produce reliable recall
        without an appropriate observation-derived consolidation objective.
        """
        learning_rate=inputs.pop('learning_rate',.001)
        preservation=inputs.pop('preservation',None)
        self.model._check_training_attention()
        if torch.is_inference_mode_enabled():raise ValueError('physical writes require autograd; use no_grad instead of inference_mode')
        if 'labels' not in inputs:raise ValueError('source-derived labels are required')
        if inputs.get('use_memory') is False:raise ValueError('physical writes require memory to be enabled')
        if any(k in inputs for k in ('memory_state','memory_write')):raise ValueError('session owns write options')
        inputs['use_cache']=False
        with self._writes,self.pin() as (state,_),self.bank.use(state.adapters),torch.enable_grad():
            out=self.model(**inputs,memory_state=state.graph,latent_memory_state=state.latents)
            loss=out.loss;recall_loss=float(loss.detach());kl=None
            if preservation is not None:
                preserved=self.model(**preservation['inputs'],memory_state=state.graph)
                actual=preserved.logits.float().log_softmax(-1)
                reference=preservation['log_probs']
                mask=preservation['inputs'].get('attention_mask',torch.ones(actual.shape[:2],device=actual.device))
                if actual.shape!=reference.shape:raise ValueError('preservation reference shape mismatch')
                kl=(F.kl_div(actual,reference,log_target=True,reduction='none').sum(-1)*mask).sum()/mask.sum()
                loss=loss+preservation['weight']*kl
            adapters,receipt=self.bank.commit(state.adapters,loss,learning_rate=learning_rate)
            receipt.update(recall_loss=recall_loss,preservation_kl=float(kl.detach()) if kl is not None else None)
            with self._lock:
                self._state=replace(state,adapters=adapters,address=None);self.generation+=1;self.dirty=True
            return dict(receipt,mount_generation=self.generation)

    def seal(self,path,*,start_new=False):
        with self._writes:
            if self._history_root is None:self._history_root=Path(path).parent
            self._consolidate_pending()
            if self.concept_lens is not None:
                with self.pin() as (state,_):address=getattr(state,'address',None)
                if address is None or getattr(address,'lens_id',None)!=self.concept_lens.lens_id:
                    self.extract_concepts(self.concept_lens,
                        parents=(self._consolidation_parent,) if self._consolidation_parent else ())
            with self.pin() as (state,_):
                receipt=self.bank.save_unit(state,path)
                replacement=self.bank.initial_unit() if start_new else state
                with self._lock:
                    self._state=replacement;self.generation+=1;self.dirty=False
                    if start_new:self._consolidation_parent=None
                return dict(receipt,mount_generation=self.generation,autonomous_consolidation=self.last_consolidation)

    def extract_concepts(self,lens,*,event_ns=None,logical_time=None,parents=()):
        """Publish continuous native concepts without generating any labels."""
        if lens.model is not self.model:raise ValueError('concept lens model differs from session')
        with self._writes,self.pin() as (state,generation):
            clock=state.graph.commits+state.adapters.commits if logical_time is None else logical_time
            address=lens.address_memory(state,event_ns=event_ns,logical_time=clock,parents=parents)
            with self._lock:
                if self.closed or self.generation!=generation:raise RuntimeError('memory changed during concept extraction')
                self._state=replace(state,address=address);self.generation+=1;self.dirty=True
            return address.metadata()

    def dream_consolidate(self,processor=None,*,source_id,validator,sleep_state=None,signals=None):
        """One source-grounded VAE replay path with an atomic retention gate."""
        if not source_id or validator is None:raise ValueError('source identity and retention validator are required')
        dream=self.model.dream_memory
        with self._writes,self.pin() as (original,generation),torch.no_grad():
            from .autonomous_memory import replay_candidate,consolidation_observations,rank_reads
            candidate,replay=replay_candidate(self.model,original,create_graph=False)
            terms=replay['codec']
            accepted=bool(validator(original,candidate))
            if signals is not None:raise ValueError('policy signals are derived from actual source predictions')
            logits,errors,retention=consolidation_observations(self.model,original,candidate)
            ranking,updated_sleep,_=rank_reads(self.model,logits,errors=errors,state=sleep_state)
            if accepted:
                with self._lock:
                    if self.closed or self.generation!=generation:raise RuntimeError('memory changed during consolidation')
                    self._state=candidate;self.generation+=1;self.dirty=True
                updated_sleep=dream.controller.complete(updated_sleep,consolidated_work=1.)
            return dict(published=accepted,source_id=source_id,functional_validator=accepted,
                replay_source=replay['replay'],
                vae={k:float(v) for k,v in terms.items()},source_positions=replay['source_positions'],
                sleep_state=updated_sleep.detach(),sleep_priority=ranking.detach().cpu().tolist(),mount_generation=self.generation)

    def unload(self):
        """Release a sealed active unit without deleting its archived weights."""
        with self._writes,self._readers,self._lock:
            if self.closed:raise RuntimeError('session closed')
            if self.dirty:raise RuntimeError('seal unsaved memory before unloading')
            self._state=self.bank.initial_unit();self.generation+=1
            self._consolidation_parent=None
            return dict(mount_generation=self.generation,archived_files_deleted=False)

    def mount_async(self,path):
        with self._lock:
            if self.closed:raise RuntimeError('session closed')
            if self.dirty:raise RuntimeError('seal unsaved memory before mounting another unit')
        if not self._mount_slot.acquire(blocking=False):raise RuntimeError('one physical mount is already pending')
        try:return self._loader.submit(self._mount,path)
        except BaseException:self._mount_slot.release();raise

    def _mount(self,path):
        try:
            state=self.bank.load_unit(path)
            if state.graph.fast.is_cuda:torch.cuda.current_stream(state.graph.fast.device).synchronize()
            with self._writes,self._lock:
                if self.closed:raise RuntimeError('session closed before mount completed')
                if self.dirty:raise RuntimeError('a concurrent write must be sealed before mounting')
                self._state=state;self.generation+=1
                self._consolidation_parent=None
                return dict(path=str(path),mount_generation=self.generation)
        finally:self._mount_slot.release()

    def close(self):
        with self._writes,self._lock:
            if self.dirty:raise RuntimeError('seal unsaved memory before closing')
            self.closed=True
        self._loader.shutdown(wait=True)
        with self._readers,self._lock:self._state=None
