"""Modified native LFM2-VL: distributed c042 MaleCNS–Titans neural memory.

The original language, vision, projector and generation implementations are
inherited from Transformers. All native weights and memory weights are saved
together. Observations are committed only after the causal forward completes.
"""
from contextvars import ContextVar
from contextlib import contextmanager,nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import os
import uuid
import threading

import torch
from transformers.models.lfm2_vl.modeling_lfm2_vl import (
    Lfm2VlForConditionalGeneration, Lfm2VlPreTrainedModel, Lfm2VlCausalLMOutputWithPast,
)

from .configuration_lfm2_titans import Lfm2TitansConfig
# Direct import makes this transitive dependency discoverable by HF's local
# dynamic-module copier as well as by save_pretrained's source collector.
from .connectome_ops import SparseMM
from .compiled_connectome import PortExecutionPlan, build_execution_plans
from .capacity import structural_support, allocate_ownership
from .multiport_connectome import ConnectomeState, MultiportConnectome, PortSpec
from .lfm_multiport import ChunkedLfmConv, LanguagePortLayer, VisionPortTower
from .multiport_connectome import MountedState
from .physical_memory import PhysicalMemorySession
from .physical_archive import PhysicalMemoryArchive
from .episodic_adapters import EpisodicAdapterBank,PhysicalMemoryUnit
from .physical_ffn import PhysicalFFNMemory
from .integrated_dream import IntegratedDreamMemory
# Explicit imports also include the complete dream graph in AutoClass exports.
from .dream_memory import DreamWeightVAE,PortFeatureVAE
from .sleep_controller import ReplayController
from .latent_memory import LatentPortMemory,encode_observations
from .sequence_memory import SequenceMemory,prepare_ordered_inputs,install_sequence_capture
from .memory_address import NativeConceptAddress
from .native_concepts import NativeConceptLens
from .memory_integrity import _checksum
from .concept_capture import install_concept_capture
from .native_sft_memory import NativeSFTMemory,NativeMemoryCheckpoint
from .batched_memory import BatchedMemoryBlock
from .sft_lora import load_custom_lora,save_custom_lora,checkpoint_weight_contexts


@dataclass
class Lfm2TitansOutput(Lfm2VlCausalLMOutputWithPast):
    memory_state: ConnectomeState | None = None
    memory_receipt: dict | None = None
    memory_keys: dict | None = None
    physical_memory_state: PhysicalMemoryUnit | None = None
    dream_features: dict | None = None
    latent_memory_state: dict | tuple | None = None


class Lfm2TitansForConditionalGeneration(Lfm2VlForConditionalGeneration):
    config_class = Lfm2TitansConfig
    _keep_in_fp32_modules_strict = []

    def __init__(self, config, native_model=None):
        if native_model is None:
            super().__init__(config)
        else:
            # Reuse the actual loaded native modules, without allocating a
            # second 3B model or reinitializing any pretrained parameter.
            Lfm2VlPreTrainedModel.__init__(self, config)
            self.model, self.lm_head = native_model.model, native_model.lm_head
            self.generation_config = deepcopy(native_model.generation_config)
            self.post_init()
        architecture = config.memory_architecture
        if not architecture:
            raise ValueError("memory_architecture is required; load a complete Lfm2_Titans checkpoint")
        # Explicit CPU construction also works inside HF's meta-device load.
        # The saved state dict then supplies all learned parameters/buffers.
        with torch.device("cpu"):
            ports = [PortSpec(p["name"], p["feature_dim"], tuple(p["input_nodes"]),
                              tuple(p["read_nodes"])) for p in architecture["ports"]]
            if 'edge_indices' in architecture:
                edges = torch.tensor(architecture["edge_indices"], dtype=torch.long)
            else:
                from safetensors.torch import load_file
                from transformers.utils.hub import cached_file
                filename=architecture.get('edge_index_file')
                if filename!='memory_graph.safetensors':raise ValueError('invalid external graph artifact')
                path=cached_file(config._name_or_path,filename)
                edges=load_file(path,device='cpu')['indices'].long()
            self.memory = MultiportConnectome(architecture["nodes"], edges, ports,
                                              **config.memory_parameters).float()
        self._memory_context = ContextVar(f"lfm2_titans_{id(self)}", default=None)
        self._install_ports()
        install_sequence_capture(self)
        install_concept_capture(self)
        if config.physical_memory is not None:self._install_physical_memory(config.physical_memory)
        if config.dream_memory is not None:self._install_dream_memory(config.dream_memory)
        self._native_sft_memory=NativeSFTMemory(self)
        self._configured_concept_lens=None
        self._concept_lens_init_lock=threading.RLock()

    def native_sft_callback(self,resume=None):
        return NativeMemoryCheckpoint(self,resume)

    def open_concept_lens(self,directory,**options):
        """Load a derived native concept dictionary without text-label generation."""
        return NativeConceptLens(self,directory,**options)

    def configured_concept_lens(self):
        """Share immutable dictionaries across independently owned sessions."""
        if self.config.concept_memory is None:return None
        with self._concept_lens_init_lock:
            if self._configured_concept_lens is None:self._configured_concept_lens=NativeConceptLens(self)
        return self._configured_concept_lens

    def load_sft_memory_adapters(self,directory):
        from .sft_lora import load_custom_lora
        load_custom_lora(self,directory,required=True,trainable=True)

    def _get_dtype_plan(self, dtype):
        plan = super()._get_dtype_plan(dtype)
        # A module-wide policy also casts integer/bool buffers in HF 5.9.
        # Only learned floating parameters need the FP32 memory policy.
        plan.update({f"memory.{name}": torch.float32 for name, _ in self.memory.named_parameters()})
        if hasattr(self,'physical_memory'):
            plan.update({f"physical_memory.{name}":torch.float32 for name,_ in self.physical_memory.named_parameters()})
        if hasattr(self,'dream_memory'):
            plan.update({f"dream_memory.{name}":torch.float32 for name,_ in self.dream_memory.named_parameters()})
            plan.update({f"dream_memory.{name}":torch.float32 for name,value in self.dream_memory.named_buffers() if value.is_floating_point()})
        return plan

    def _install_dream_memory(self,spec):
        with torch.device('cpu'):
            dimensions={p.name:p.feature_dim for p in self.memory.specs}
            codec_port=getattr(self.config,'sequence_codec_port',None)
            if codec_port is not None:dimensions[codec_port]=self.config.text_config.hidden_size
            self.dream_memory=IntegratedDreamMemory(spec,dimensions).float()
        self.dream_memory.to(self.memory.slow_weights.device)

    def enable_dream_memory(self,controller_graph,*,weight_vae=None,feature_vae=None,seed=731,kl_weight=.001):
        if not hasattr(self,'physical_memory'):raise ValueError('enable physical memory before dreams')
        if hasattr(self,'dream_memory'):raise ValueError('dream architecture already enabled')
        spec=dict(version=1,weight_vae=weight_vae or dict(chunk_size=1024,hidden_size=128,latent_size=32),
            feature_vae=feature_vae or dict(hidden_size=64,latent_size=16),controller_graph=deepcopy(controller_graph),
            seed=seed,kl_weight=kl_weight,publication='verify_new_and_retained_sources_before_atomic_commit',
            history='append_only',replay='source_conditioned_native_features')
        self._install_dream_memory(spec);self.config.dream_memory=spec
        return spec

    def _install_physical_memory(self,spec):
        bank=getattr(self,'_episodic_adapter_bank',None)
        if bank is None:bank=EpisodicAdapterBank(self,rank=spec['rank'],seed=spec['seed'],architecture_owned=True)
        if bank.rank!=spec['rank']:raise ValueError('physical memory rank differs from installed bank')
        self.physical_memory=PhysicalFFNMemory(bank.targets,rank=spec['rank'],seed=spec['seed'],scale=spec['scale'],
            device=self.memory.slow_weights.device)

    def enable_latent_memory(self):
        if not hasattr(self,'dream_memory'):raise ValueError('VAE heads are required')
        if self.config.latent_memory is not None:raise ValueError('latent memory already enabled')
        self.config.latent_memory=dict(version=1,representation='per-feature posterior and ordered native inputs',
            storage='immutable physical unit',recall='ordered native input replay',codec='dream_memory.feature_vae')
        return self.config.latent_memory

    def enable_physical_memory(self,*,rank=16,seed=731):
        """Promote episodic FFN memory to a saved, trainable architecture component."""
        if hasattr(self,'physical_memory'):raise ValueError('physical memory architecture already enabled')
        if type(rank) is not int or rank<1:raise ValueError('positive physical memory rank required')
        spec=dict(version=1,rank=rank,seed=seed,scale=2.,write_gradient='first_order',
            layers=sorted(map(int,self.config.language_ports)),projections=['w1','w2','w3'],
            publication='verified_atomic',state='caller_owned',precision='float32')
        self._install_physical_memory(spec);self.config.physical_memory=spec
        return spec

    def initial_physical_memory(self,*,create_graph=False):
        if not hasattr(self,'physical_memory'):raise ValueError('enable the physical memory architecture first')
        return self._episodic_adapter_bank.initial_unit(create_graph=create_graph)

    def write_physical_memory(self,inputs,state=None,*,learning_rate=.001,create_graph=False,update_graph=True,capture_dream=False):
        """Source-only write; returned state is unpublished until the caller commits.

        During meta training, source gradients are first order. The following
        recall loss trains the initialization and per-projection write rate.
        Native attention therefore needs only its supported first derivative.
        """
        self._check_training_attention()
        if not hasattr(self,'physical_memory'):raise ValueError('physical memory architecture is disabled')
        if 'labels' not in inputs:raise ValueError('explicit source-derived labels required')
        if any(k in inputs for k in ('memory_state','physical_memory_state','memory_write','use_memory')):
            raise ValueError('writer owns memory scope')
        state=self.initial_physical_memory(create_graph=create_graph) if state is None else state
        with torch.enable_grad():
            output=self(**dict(inputs,use_cache=False),physical_memory_state=state,memory_write=update_graph,
                memory_create_graph=create_graph and update_graph,
                dream_capture=capture_dream or self.config.latent_memory is not None,
                memory_capture_keys=True,
                memory_write_gradient_scope='local' if create_graph and update_graph else 'full')
            adapters,receipt=self._episodic_adapter_bank.commit(state.adapters,output.loss,
                learning_rate=learning_rate,first_order_graph=create_graph)
        graph=output.memory_state if create_graph else output.memory_state.detach()
        if capture_dream:receipt['dream_features']=output.dream_features
        if not update_graph and self.config.latent_memory is not None:
            output.latent_memory_state=encode_observations(self.dream_memory.feature_vae,output.dream_features,
                state.latents,create_graph=create_graph)
        return PhysicalMemoryUnit(adapters,graph,output.memory_keys,output.latent_memory_state,
            getattr(state,'sequences',())),receipt

    def _install_ports(self):
        layers = self.model.language_model.layers
        placements = {int(k): v for k, v in self.config.language_ports.items()}
        names = list(placements.values()) + ([self.config.vision_port] if self.config.vision_port else [])
        if len(placements) < 2 or len(set(names)) != len(names) or set(names) != set(self.memory.ports):
            raise ValueError("independent memory ports must match distributed model placements")
        specs = {p.name: p for p in self.memory.specs}
        for depth, name in placements.items():
            if not 0 <= depth < len(layers) or specs[name].feature_dim != self.config.text_config.hidden_size:
                raise ValueError(f"invalid language port {depth}/{name}")
        if self.config.vision_port and specs[self.config.vision_port].feature_dim != self.config.vision_config.hidden_size:
            raise ValueError("native vision dimension mismatch")
        for layer in layers:
            if not layer.is_attention_layer:
                layer.conv = ChunkedLfmConv(layer.conv)
        for depth, name in placements.items():
            layers[depth] = LanguagePortLayer(layers[depth], name, self._memory_context)
        if self.config.vision_port:
            self.model.vision_tower = VisionPortTower(self.model.vision_tower, self.config.vision_port,
                                                     self._memory_context)

    def _check_training_attention(self):
        if self.config.diagnostic_allow_other_attention:
            return
        for cfg in (self.config.text_config, self.config.vision_config):
            backend = cfg._attn_implementation or ""
            if backend != "flash_attention_2" and not backend.startswith("kernels-community/flash-attn2@"):
                raise ValueError(f"Training requires FlashAttention-2, got {backend!r}")

    def gradient_checkpointing_enable(self,gradient_checkpointing_kwargs=None):
        options=dict(gradient_checkpointing_kwargs or {})
        if self.config.physical_memory is not None:
            if options.get('use_reentrant',True):
                raise ValueError('physical memory training requires use_reentrant=False')
            original=options.get('context_fn',lambda:(nullcontext(),nullcontext()))
            def contexts():
                forward,_=original();bank=self._episodic_adapter_bank;snapshot=bank.context.get()
                cache_forward,cache_recompute=checkpoint_weight_contexts()
                class Restore:
                    # A first-order memory gradient and Trainer.backward may
                    # both recompute the same checkpoint. Generator contexts
                    # are single-use, so create a new one on every entry.
                    def __init__(self):self.stack=[]
                    def __enter__(self):
                        token=bank.context.set(snapshot);_,context=original()
                        self.stack.append((token,context));cache_recompute.__enter__();return context.__enter__()
                    def __exit__(self,*exc):
                        token,context=self.stack.pop()
                        try:return context.__exit__(*exc)
                        finally:cache_recompute.__exit__(*exc);bank.context.reset(token)
                return forward,Restore()
            options['context_fn']=contexts
        return super().gradient_checkpointing_enable(gradient_checkpointing_kwargs=options)

    @classmethod
    def from_native(cls, native_model, memory, specification, origin=None):
        values = native_model.config.to_dict()
        values.pop("model_type", None)
        values.pop("architectures", None)
        values.pop("auto_map", None)
        architecture = memory.architecture()
        architecture["edge_indices"] = memory.indices.detach().cpu().tolist()
        config = Lfm2TitansConfig(**values, memory_architecture=architecture,
            memory_parameters=specification["memory"],
            language_ports={str(d): f"language_{d}" for d in specification["language_layers"]},
            vision_port="visual", memory_origin=origin)
        config._attn_implementation = native_model.config._attn_implementation
        config.text_config._attn_implementation = native_model.config.text_config._attn_implementation
        config.vision_config._attn_implementation = native_model.config.vision_config._attn_implementation
        model = cls(config, native_model=native_model)
        model.memory = memory
        return model

    def forward(self, input_ids=None, pixel_values=None, spatial_shapes=None,
                pixel_attention_mask=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, memory_state=None, memory_write=False,
                memory_create_graph=False, use_memory=True, memory_capture_keys=False,
                physical_memory_state=None, dream_capture=False,
                latent_memory_state=None,**kwargs):
        if 'memory_read_mode' in kwargs:raise ValueError('only the unified dual-memory reader is available')
        if 'memory_write_gradient_scope' in kwargs:raise ValueError('the old local-gradient writer branch was removed')
        if any(k in kwargs for k in ('memory_record','memory_start','memory_end','memory_total','memory_overlap')):
            raise ValueError('SFT windows were removed; pass ordinary native training examples')
        if self.training and labels is not None and getattr(self.config,'native_joint_sft',False):
            if physical_memory_state is not None or memory_state is not None:
                raise ValueError('SFT memory is owned by the model training session')
            self._check_training_attention()
            inputs=dict(input_ids=input_ids,labels=labels,attention_mask=attention_mask,
                pixel_values=pixel_values,pixel_attention_mask=pixel_attention_mask,spatial_shapes=spatial_shapes,
                position_ids=position_ids,past_key_values=past_key_values,inputs_embeds=inputs_embeds,
                logits_to_keep=logits_to_keep,**kwargs)
            returned=inputs.pop('return_dict',getattr(self.config,'return_dict',True))
            output=self._native_sft_memory.forward(super().forward,inputs)
            return output if returned is not False else output.to_tuple()
        if latent_memory_state and self.config.latent_memory is None:raise ValueError('latent memory architecture is disabled')
        if physical_memory_state is not None:
            if not use_memory or memory_state is not None:raise ValueError('one complete physical memory state is required')
            if not hasattr(self,'physical_memory'):raise ValueError('physical memory architecture is disabled')
            if not isinstance(physical_memory_state,PhysicalMemoryUnit):raise TypeError('expected PhysicalMemoryUnit')
            returned=kwargs.pop('return_dict',getattr(self.config,'return_dict',True))
            if returned is None:returned=getattr(self.config,'return_dict',True)
            if latent_memory_state is not None:raise ValueError('complete physical unit already owns latent state')
            with self._episodic_adapter_bank.use(physical_memory_state.adapters):
                output=self.forward(input_ids=input_ids,pixel_values=pixel_values,spatial_shapes=spatial_shapes,
                    pixel_attention_mask=pixel_attention_mask,attention_mask=attention_mask,position_ids=position_ids,
                    past_key_values=past_key_values,inputs_embeds=inputs_embeds,labels=labels,use_cache=use_cache,
                    logits_to_keep=logits_to_keep,memory_state=physical_memory_state.graph,memory_write=memory_write,
                    memory_create_graph=memory_create_graph,use_memory=True,memory_capture_keys=memory_capture_keys,
                    dream_capture=dream_capture,
                    latent_memory_state=physical_memory_state.latents,return_dict=True,**kwargs)
            output.physical_memory_state=PhysicalMemoryUnit(physical_memory_state.adapters,output.memory_state,
                physical_memory_state.neural_keys,output.latent_memory_state,
                getattr(physical_memory_state,'sequences',()),
                None if memory_write else getattr(physical_memory_state,'address',None))
            return output if returned else output.to_tuple()
        if self.training:
            self._check_training_attention()
        if self._memory_context.get() is not None:
            raise RuntimeError("nested memory forward is not allowed")
        return_dict = kwargs.pop("return_dict", None)
        if return_dict is None:
            return_dict = getattr(self.config, "return_dict", True)
        native = dict(input_ids=input_ids, pixel_values=pixel_values, spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            labels=labels, use_cache=use_cache, logits_to_keep=logits_to_keep, **kwargs)
        if not use_memory:
            if memory_write:
                raise ValueError("memory_write requires use_memory=True")
            adapter_bank=getattr(self,'_episodic_adapter_bank',None)
            if adapter_bank is not None:
                with adapter_bank.suspended():return super().forward(**native,return_dict=return_dict)
            return super().forward(**native, return_dict=return_dict)
        state = memory_state if memory_state is not None else self.memory.initial_state()
        if state.fast.device != self.memory.slow_weights.device or state.fast.shape != self.memory.slow_weights.shape:
            raise ValueError("memory state device/shape does not match this model")
        # A fast state is one caller-owned session. Writing multiple unrelated
        # batch entries into it would silently mix their memories.
        source = input_ids if input_ids is not None else inputs_embeds
        if memory_write and source.shape[0] != 1:
            raise ValueError("write one session per call; batch reads may share an explicit state")
        block = self.memory.begin(state)
        mask = attention_mask
        if mask is not None:
            if mask.ndim != 2 or mask.shape[1] < source.shape[1]:
                raise ValueError("expected 2D validity mask including all new tokens")
            mask = mask[:, -source.shape[1]:]
        token = self._memory_context.set((block, memory_write or memory_capture_keys or dream_capture, mask))
        try:
            output = super().forward(**native, return_dict=True)
        finally:
            self._memory_context.reset(token)
        keys = block.routing_keys() if memory_capture_keys else None
        native_observations=block.native_observations
        features={name:value.detach() for name,value in native_observations.items()} if dream_capture else None
        if memory_write:
            state, receipt = block.commit(create_graph=memory_create_graph)
        else:
            receipt = {"observations": 0, "commits": state.commits}
        if memory_write and self.config.latent_memory is not None:
            if latent_memory_state is not None and not isinstance(latent_memory_state,dict):
                raise ValueError('write one latent unit; mounted mixtures are read-only')
            latent_memory_state=encode_observations(self.dream_memory.feature_vae,native_observations,
                latent_memory_state,create_graph=memory_create_graph)
            receipt['latent_features']={k:v.count for k,v in latent_memory_state.items()}
        result = Lfm2TitansOutput(**dict(output), memory_state=state, memory_receipt=receipt, memory_keys=keys,
            dream_features=features,latent_memory_state=latent_memory_state)
        return result if return_dict else result.to_tuple()

    def forward_block(self, inputs, state, *, write, create_graph):
        output = self(**inputs, memory_state=state, memory_write=write,
                      memory_create_graph=create_graph, return_dict=True)
        return output, output.memory_state, output.memory_receipt

    def generate(self, *args, **kwargs):
        if kwargs.get("memory_write", False):
            raise ValueError("Write observations with forward(memory_write=True), then generate with the returned memory_state")
        return super().generate(*args, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        lens_source=None
        if self.config.concept_memory is not None:
            from transformers.utils.hub import cached_file
            lens_source=Path(cached_file(self.config._name_or_path,self.config.concept_memory['lens_file'])).read_bytes()
        # Every published weight snapshot gets its own session identity.
        self.config.memory_checkpoint_id = uuid.uuid4().hex
        result=super().save_pretrained(save_directory, **kwargs)
        # Binary graph artifact avoids millions of JSON integers. Save after
        # the native shard writer, which may clean older safetensors files.
        if self.config.memory_architecture.get('edge_index_file'):
            from safetensors.torch import save_file
            save_file({'indices':self.memory.indices.detach().cpu().to(torch.int32).contiguous()},
                      str(Path(save_directory)/'memory_graph.safetensors'))
        if lens_source is not None:
            # Keep the calibration provenance. A new trained checkpoint ID
            # intentionally fails lens validation until it is recalibrated.
            (Path(save_directory)/self.config.concept_memory['lens_file']).write_bytes(lens_source)
        return result

    def open_physical_memory_session(self,*,rank=None,max_readers=1):
        """Graph + FFN fast weights, with immutable disk units and hot mounting.

        Freeze the model with eval().requires_grad_(False) before opening.
        Observation-derived consolidation labels are explicit in session.learn.
        """
        rank=rank if rank is not None else (self.config.physical_memory or {}).get('rank',16)
        bank=getattr(self,'_episodic_adapter_bank',None)
        if bank is not None and bank.checkpoint_id is None:
            bank.checkpoint_id=getattr(self.config,'memory_checkpoint_id',None)
        lens=self.configured_concept_lens()
        if lens is None:raise ValueError('calibrate the native concept lens before opening inference memory')
        return PhysicalMemorySession(self,rank=rank,max_readers=max_readers,concept_lens=lens)

    def open_physical_archive(self,directory,*,max_readers=1,max_resident_units=8):
        return PhysicalMemoryArchive(self,directory,max_readers=max_readers,max_resident_units=max_resident_units )

    @property
    def base(self):
        """Compatibility with explicit block evaluation; no duplicate module."""
        return self


Lfm2TitansForConditionalGeneration.register_for_auto_class("AutoModelForImageTextToText")
