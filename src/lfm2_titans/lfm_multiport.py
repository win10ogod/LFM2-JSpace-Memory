"""LFM language/vision integration without mutable writes inside checkpoints."""
from __future__ import annotations

from contextvars import ContextVar
import torch
from torch import nn
from torch.nn import functional as F

from .multiport_connectome import MultiportConnectome, ConnectomeState


class ChunkedLfmConv(nn.Module):
    """Exact causal convolution for multi-token continuation of a hybrid cache.

    Transformers 5.9's LFM slow cached path computes only one convolution output
    and broadcasts it for T>1. Preserve the original weights and the complete
    chunk by prepending the previous L-1 inputs and doing the actual convolution.
    """
    def __init__(self, original):
        super().__init__()
        self.original=original

    def forward(self, hidden_states, past_key_values=None, attention_mask=None):
        original=self.original
        if (past_key_values is None or hidden_states.shape[1]==1
                or not past_key_values.has_previous_state(original.layer_idx)):
            return original(hidden_states,past_key_values=past_key_values,attention_mask=attention_mask)
        if attention_mask is not None:
            hidden_states=hidden_states*attention_mask[:,-hidden_states.shape[1]:,None].to(hidden_states.dtype)
        b,c,x=original.in_proj(hidden_states).transpose(-1,-2).chunk(3,dim=-2)
        bx=b*x
        prior=past_key_values.layers[original.layer_idx].conv_states
        width=original.L_cache
        joined=torch.cat([prior[...,-(width-1):].to(bx.dtype),bx],dim=-1) if width>1 else bx
        conv=F.conv1d(joined,original.conv.weight,original.conv.bias,groups=bx.shape[1])
        # Same fixed-size state as the native layer; no dropped chunk tokens.
        past_key_values.layers[original.layer_idx].conv_states.copy_(joined[...,-width:])
        return original.out_proj((c*conv).transpose(-1,-2).contiguous())


class LanguagePortLayer(nn.Module):
    def __init__(self, original, name, context):
        super().__init__()
        self.original = original
        self.port_name, self.context = name, context
        self.is_attention_layer = original.is_attention_layer

    def forward(self, *args, **kwargs):
        # The original layer owns any activation checkpoint. Port reads and
        # observation registration occur outside that checkpointed call.
        hidden = self.original(*args, **kwargs)
        active = self.context.get()
        if active is None:
            return hidden
        block, write, mask = active
        if write:
            block.observe(self.port_name, self.port_name, hidden, mask)
        return block.residual(self.port_name, hidden)


class VisionPortTower(nn.Module):
    def __init__(self, original, name, context):
        super().__init__()
        self.original = original
        self.port_name, self.context = name, context

    def forward(self, *args, **kwargs):
        result = self.original(*args, **kwargs)
        active = self.context.get()
        if active is None:
            return result
        block, write, _ = active
        hidden = result.last_hidden_state
        mask = kwargs.get("pixel_attention_mask")
        if write:
            # Preserve every valid tile/patch feature, with native positional
            # encoding already applied. No caption or pooled image surrogate.
            block.observe(self.port_name, self.port_name, hidden, mask)
        result.last_hidden_state = block.residual(self.port_name, hidden)
        return result


class LfmMultiportMemory(nn.Module):
    """Explicit block API; caller owns each session's fast state and base cache.

    `forward_block` processes new causal input once. Reusing a generation cache
    does not authorize re-presenting and writing old images/tokens. This wrapper
    intentionally exposes state rather than hiding a shared global memory.
    """
    def __init__(self, base, memory: MultiportConnectome,
                 language_ports: dict[int, str], vision_port: str | None = None, *,
                 diagnostic_allow_other_attention: bool = False):
        super().__init__()
        for config in (base.config.text_config, base.config.vision_config):
            backend = config._attn_implementation or ""
            fa2 = (backend == "flash_attention_2"
                   or backend.startswith("kernels-community/flash-attn2@"))
            if not fa2 and not diagnostic_allow_other_attention:
                raise ValueError(f"FlashAttention-2 is required; selected backend is {backend!r}")
        if len(language_ports) < 2:
            raise ValueError("distributed integration requires at least two language depths")
        names = list(language_ports.values()) + ([vision_port] if vision_port else [])
        if len(set(names)) != len(names) or set(names) != set(memory.ports):
            raise ValueError("every memory port must have one independent model placement")
        layers = base.model.language_model.layers
        text_dim = base.config.text_config.hidden_size
        specs = {p.name: p for p in memory.specs}
        # Validate everything before modifying the selected base.
        for depth, name in language_ports.items():
            if not 0 <= depth < len(layers) or specs[name].feature_dim != text_dim:
                raise ValueError(f"invalid language placement {depth}/{name}")
        if vision_port and specs[vision_port].feature_dim != base.config.vision_config.hidden_size:
            raise ValueError("vision feature dimension does not match native tower")
        base.requires_grad_(False)
        self.base, self.memory = base, memory
        self.language_ports, self.vision_port = language_ports, vision_port
        self._context = ContextVar(f"lfm_memory_{id(self)}", default=None)
        for layer in layers:
            if not layer.is_attention_layer:
                layer.conv=ChunkedLfmConv(layer.conv)
        for depth, name in language_ports.items():
            layers[depth] = LanguagePortLayer(layers[depth], name, self._context)
        if vision_port:
            base.model.vision_tower = VisionPortTower(base.model.vision_tower, vision_port, self._context)

    def forward_block(self, inputs: dict, state: ConnectomeState, *,
                      write: bool, create_graph: bool):
        if self._context.get() is not None:
            raise RuntimeError("nested memory forward is not allowed")
        block = self.memory.begin(state)
        mask = inputs.get("attention_mask")
        # A generation mask can include cached prefix positions; observations
        # here correspond only to the newly provided tokens.
        length = (inputs["input_ids"].shape[1] if "input_ids" in inputs
                  else inputs["inputs_embeds"].shape[1])
        if mask is not None:
            if mask.ndim != 2 or mask.shape[1] < length:
                raise ValueError("expected a 2D token validity mask")
            mask = mask[:, -length:]
        token = self._context.set((block, write, mask))
        try:
            output = self.base(**inputs)
        finally:
            self._context.reset(token)
        if write:
            next_state, receipt = block.commit(create_graph=create_graph)
        else:
            next_state, receipt = state, {"observations": 0, "commits": state.commits}
        return output, next_state, receipt
