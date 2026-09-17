"""Modified LFM2-VL configuration: adds the complete c042 memory architecture."""
from copy import deepcopy
from transformers import Lfm2VlConfig


class Lfm2TitansConfig(Lfm2VlConfig):
    model_type = "lfm2_titans"
    has_no_defaults_at_init = True

    def __init__(self, memory_architecture=None, memory_parameters=None,
                 language_ports=None, vision_port="visual", memory_origin=None,
                 diagnostic_allow_other_attention=False, physical_memory=None, dream_memory=None, latent_memory=None,
                 concept_memory=None, **kwargs):
        super().__init__(**kwargs)
        self.memory_architecture = deepcopy(memory_architecture)
        self.memory_parameters = deepcopy(memory_parameters or {})
        self.language_ports = {str(k): v for k, v in (language_ports or {}).items()}
        self.vision_port = vision_port
        self.memory_origin = deepcopy(memory_origin or {})
        self.diagnostic_allow_other_attention = diagnostic_allow_other_attention
        self.memory_precision = "float32"
        self.memory_state_policy = "caller_owned; causal block snapshot then one commit"
        self.physical_memory=deepcopy(physical_memory)
        self.dream_memory=deepcopy(dream_memory)
        self.latent_memory=deepcopy(latent_memory)
        self.concept_memory=deepcopy(concept_memory)
        if self.concept_memory is not None:
            c=self.concept_memory
            if (c.get('version')!=1 or c.get('lens_file')!='memory_concept_lens.safetensors'
                    or type(c.get('sparsity')) is not int or c['sparsity']<1
                    or c.get('storage')!='native_sparse_concepts_v6'
                    or c.get('recall')!='ordered'):
                raise ValueError('unsupported native concept memory configuration')
        if self.latent_memory is not None and (self.dream_memory is None or self.latent_memory.get('version')!=1):
            raise ValueError('latent memory requires the saved per-port VAE heads')
        if self.dream_memory is not None and (self.physical_memory is None or self.dream_memory.get('version')!=1):
            raise ValueError('integrated dreams require the physical memory architecture')
        if self.physical_memory is not None:
            p=self.physical_memory
            if (p.get('version')!=1 or type(p.get('rank')) is not int or p['rank']<1
                    or p.get('write_gradient')!='first_order' or p.get('scale')!=2.):
                raise ValueError('unsupported physical FFN memory architecture')


Lfm2TitansConfig.register_for_auto_class()
