"""Small native VL fixture for the current joint-memory architecture."""
import torch
from transformers import Lfm2VlConfig, Lfm2VlForConditionalGeneration
from lfm2_titans.modeling_lfm2_titans import Lfm2TitansForConditionalGeneration
from lfm2_titans.multiport_connectome import MultiportConnectome, PortSpec

def native_and_memory():
    torch.manual_seed(51)
    cfg = Lfm2VlConfig(image_token_id=31, text_config=dict(hidden_size=16,
        intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
        num_key_value_heads=1, vocab_size=32, eos_token_id=30, bos_token_id=1,
        pad_token_id=0, layer_types=["conv", "full_attention", "conv"]),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, patch_size=2))
    cfg._attn_implementation = "eager"
    native = Lfm2VlForConditionalGeneration(cfg).eval()
    ports = [PortSpec("language_0", 16, (0,1,2,3), (2,3,4,5)),
             PortSpec("language_2", 16, (4,5,6,7), (0,1,6,7)),
             PortSpec("visual", 16, (0,2,4,6), (1,3,5,7))]
    params = dict(channels=4, activation="dendritic", microsteps=2,
        gradient_normalization="rms", visual_protection=True, retention_anchor="slow",
        checkpoint_reads=True, feature_chunk_size=3)
    mem = MultiportConnectome(8, torch.cartesian_prod(torch.arange(8), torch.arange(8)).T,
                             ports, **params)
    return native, mem, dict(memory=params, language_layers=[0,2])


def make_model():
    native, memory, spec = native_and_memory()
    model = Lfm2TitansForConditionalGeneration.from_native(native, memory, spec)
    model.config.diagnostic_allow_other_attention = True
    return model



def dream_model():
    model=make_model();model.enable_physical_memory(rank=2)
    types=['MBON12','ER5','MBON03','ExR1','FB6A_a','DPM','PAM01']
    graph=dict(types=types,src=list(range(7)),dst=[1,2,3,4,5,6,0],weight=[1.]*7)
    model.enable_dream_memory(graph,weight_vae=dict(chunk_size=32,hidden_size=8,latent_size=2),
        feature_vae=dict(hidden_size=8,latent_size=2))
    model.enable_latent_memory()
    return model


