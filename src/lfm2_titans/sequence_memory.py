"""Ordered VAE recall with measured, unit-local reconstruction corrections.

The shared VAE remains the codec prior. Residual factors and sparse numerical
patches pay explicitly for information that this prior cannot yet reconstruct.
No source strings, token IDs, pixels, or native KV caches are stored here.
This is not a fixed-size lossless compressor: incompressible observations cost
more bytes. Exact feature reconstruction does not certify correct answers.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import re
import torch
from .latent_memory import LatentPortMemory


DTYPES = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
LABELS = ('mu', 'logvar', 'mean', 'scale', 'left', 'right', 'patch_indices', 'patch_values')


def feature_checksum(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class SequenceMemory:
    codec_port: str
    posterior: LatentPortMemory
    left: torch.Tensor
    right: torch.Tensor
    patch_indices: torch.Tensor
    patch_values: torch.Tensor
    dtype: str
    checksum: str

    def tensors(self):
        return (*self.posterior.tensors(), self.left, self.right, self.patch_indices, self.patch_values)

    def detach(self):
        return SequenceMemory(self.codec_port, self.posterior.detach(), self.left.detach(), self.right.detach(),
                              self.patch_indices.detach(), self.patch_values.detach(), self.dtype, self.checksum)

    @property
    def count(self):return self.posterior.count

    @property
    def bytes(self):return sum(x.numel() * x.element_size() for x in self.tensors())


def validate_sequence(model, segment):
    if not isinstance(segment, SequenceMemory):raise ValueError('invalid sequence memory')
    codec = getattr(model, 'dream_memory', None)
    if codec is None or segment.codec_port not in codec.feature_vae.heads:
        raise ValueError('sequence codec is absent from this checkpoint')
    n, dim = segment.count, model.config.text_config.hidden_size
    latent = model.config.dream_memory['feature_vae']['latent_size']
    if codec.feature_vae.heads[segment.codec_port].decoder[-1].out_features != dim:
        raise ValueError('sequence codec has incompatible native width')
    p = segment.posterior
    if n < 1 or p.mu.shape != (n, latent) or p.logvar.shape != p.mu.shape:
        raise ValueError('invalid sequence posterior')
    if p.mean.shape != (n, 1) or p.scale.shape != (n, 1) or (p.scale <= 0).any():
        raise ValueError('invalid sequence normalization')
    if segment.left.ndim != 2 or segment.right.ndim != 2:
        raise ValueError('invalid sequence correction factors')
    rank = segment.left.shape[1]
    if segment.left.shape != (n, rank) or segment.right.shape != (rank, dim) or rank > min(n, dim):
        raise ValueError('invalid sequence correction shapes')
    idx = segment.patch_indices
    if idx.ndim != 1 or idx.dtype != torch.int64 or segment.patch_values.shape != idx.shape:
        raise ValueError('invalid sequence numerical patches')
    if idx.numel() and ((idx < 0).any() or (idx >= n * dim).any() or (idx[1:] <= idx[:-1]).any()):
        raise ValueError('sequence patches must have unique ascending in-range indices')
    if segment.dtype not in DTYPES or not re.fullmatch('[a-f0-9]{64}', segment.checksum):
        raise ValueError('invalid sequence reconstruction contract')
    if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in (*p.tensors(), segment.left, segment.right, segment.patch_values)):
        raise ValueError('nonfinite or incompatible sequence values')


def _base(head, posterior):
    device = next(head.parameters()).device
    with torch.autocast(device_type=device.type, enabled=False):
        return head.decoder(posterior.mu.to(device)) * posterior.scale.to(device) + posterior.mean.to(device)


@torch.no_grad()
def encode_sequence(model, features, *, chunk_size=128):
    """Fit on observed native embeddings only, with no recall questions.

    Choose the smallest measured factor+patch encoding among candidate ranks.
    Chunks bound SVD scratch, not source length: every row is serialized.
    All output tensors live on CPU until explicitly decoded for a query.
    """
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('sequence writes require a frozen checkpoint')
    if type(chunk_size) is not int or chunk_size < 1:raise ValueError('positive codec chunk size required')
    if features.ndim != 2 or features.shape[1] != model.config.text_config.hidden_size or not len(features):
        raise ValueError('nonempty ordered native input features required')
    dtype = str(features.dtype).removeprefix('torch.')
    if dtype not in DTYPES or not torch.isfinite(features).all():raise ValueError('invalid native input features')
    port = getattr(model.config,'sequence_codec_port',None) or model.config.language_ports[min(model.config.language_ports, key=int)]
    head = model.dream_memory.feature_vae.heads[port]
    device = next(head.parameters()).device
    segments = []
    for start in range(0, len(features), chunk_size):
        source = features[start:start + chunk_size].detach().to(device)
        x = source.float()
        with torch.autocast(device_type=device.type, enabled=False):
            mean = x.mean(-1, keepdim=True)
            scale = (x.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
            mu, logvar = head.posterior(head.encoder((x - mean) / scale)).chunk(2, -1)
            posterior = LatentPortMemory(mu, logvar.clamp(-12, 8), mean, scale)
            base = _base(head, posterior).cpu()
        # CPU SVD avoids an expensive GPU solver workspace. Reconstruction
        # correction always uses CPU FP32, so cold load follows the same path.
        original = source.cpu().contiguous()
        residual = original.float() - base
        u, s, v = torch.linalg.svd(residual, full_matrices=False)
        ranks = sorted({0, len(s), *(min(r, len(s)) for r in (4, 8, 16, 32, 64))})
        best = None
        for rank in ranks:
            left = (u[:, :rank] * s[:rank]).contiguous()
            right = v[:rank].contiguous()
            rebuilt = (base + left @ right).to(original.dtype)
            bits = torch.int32 if original.dtype == torch.float32 else torch.int16
            indices = (rebuilt.view(bits).reshape(-1) != original.view(bits).reshape(-1)).nonzero().flatten()
            values = original.reshape(-1)[indices].float()
            cost = (left.numel() + right.numel() + values.numel()) * 4 + indices.numel() * 8
            if best is None or cost < best[0]:best = (cost, left, right, indices, values)
        _, left, right, indices, values = best
        posterior = LatentPortMemory(*(t.detach().cpu().contiguous() for t in posterior.tensors()))
        segment = SequenceMemory(port, posterior, left, right, indices, values, dtype, feature_checksum(original))
        # A numerical reconstruction guarantee is checked before publication,
        # and again on every cold decode. Failures never become silent recall.
        validate_sequence(model, segment)
        decode_segment(model, segment)
        segments.append(segment)
    return tuple(segments)


@torch.no_grad()
def decode_segment(model, segment):
    validate_sequence(model, segment)
    head = model.dream_memory.feature_vae.heads[segment.codec_port]
    x = (_base(head, segment.posterior).cpu() + segment.left.cpu() @ segment.right.cpu()).to(DTYPES[segment.dtype])
    x.reshape(-1)[segment.patch_indices.cpu()] = segment.patch_values.cpu().to(x.dtype)
    if feature_checksum(x) != segment.checksum:
        raise ValueError('sequence reconstruction checksum failed; checkpoint or numerical backend is incompatible')
    return x


def install_sequence_capture(model):
    """A caller-local capture scope, safe across independent read sessions."""
    model._sequence_capture_context = ContextVar(f'sequence_capture_{id(model)}', default=None)
    def capture(module, args, kwargs):
        scope = model._sequence_capture_context.get()
        if scope is None:return
        captured, attention_mask = scope
        value = kwargs.get('inputs_embeds')
        if value is None:raise ValueError('native language input embeddings were not supplied')
        if value.shape[0] != 1:raise ValueError('write one ordered session per call')
        value = value[0]
        if attention_mask is not None:
            if attention_mask.shape != (1, len(value)):raise ValueError('sequence validity mask mismatch')
            value = value[attention_mask[0].bool()]
        captured.append(value.detach().clone())
    model.model.language_model.register_forward_pre_hook(capture, with_kwargs=True)


@contextmanager
def capture_native_inputs(model, attention_mask=None):
    """Capture native text/vision embeddings without changing their forward."""
    if model._sequence_capture_context.get() is not None:raise RuntimeError('nested sequence capture')
    captured = []
    token = model._sequence_capture_context.set((captured, attention_mask))
    try:yield captured
    finally:model._sequence_capture_context.reset(token)


@torch.no_grad()
def prepare_ordered_inputs(model, sequences, inputs, *, max_new_tokens=None, max_length=None):
    """Restore selected sequence features in order through ALL native layers.

    No prefix token IDs are recovered. Original input_ids remain the returned
    generation prefix, so existing callers keep their normal output slicing.
    Recurrent and attention caches are rebuilt normally and remain ephemeral.
    """
    if not sequences:raise ValueError('this unit has no ordered memory; re-observe its source to create it')
    forbidden = {'inputs_embeds', 'past_key_values', 'position_ids', 'pixel_values', 'pixel_attention_mask', 'spatial_shapes'}
    if any(inputs.get(k) is not None for k in forbidden):
        raise ValueError('ordered recall currently takes a fresh text query; stored memories may include vision')
    ids = inputs.get('input_ids')
    if ids is None or ids.ndim != 2 or ids.shape[0] != 1:raise ValueError('ordered recall requires one tokenized query')
    mask = inputs.get('attention_mask')
    if mask is None:mask = torch.ones_like(ids)
    if mask.shape != ids.shape or not mask.bool().all():raise ValueError('ordered recall requires an unpadded query')
    length = sum(s.count for s in sequences)
    generation_config = inputs.get('generation_config') or model.generation_config
    reserve = max_new_tokens if max_new_tokens is not None else generation_config.max_new_tokens
    if reserve is None:
        effective_max = max_length if max_length is not None else generation_config.max_length
        reserve = max(0, effective_max - ids.shape[1])
    limit = model.config.text_config.max_position_embeddings
    if length + ids.shape[1] + reserve > limit:
        raise ValueError(f'selected memory + query + output ({length + ids.shape[1] + reserve}) exceeds native context {limit}; select smaller units, no truncation performed')
    embeddings = model.get_input_embeddings()(ids)
    prefix = torch.cat([decode_segment(model, segment) for segment in sequences]).to(embeddings)
    bos=getattr(model.config,'bos_token_id',None)
    if bos is None:bos=getattr(model.config.text_config,'bos_token_id',None)
    combined,_=assemble_memory_query(prefix,embeddings[0],ids[0],bos)
    return dict(inputs, inputs_embeds=combined[None],
                attention_mask=torch.ones((1, length + ids.shape[1]), device=ids.device, dtype=mask.dtype))


def assemble_memory_query(prefix,query,query_ids,bos_token_id):
    """Keep the native start token before recalled observations and the query.

    No feature or query token is omitted. Inserting a new conversation start
    after the recalled observations can make the model treat them as outside
    the current conversation. The same ordering is used by training and recall.
    """
    leading=int(bos_token_id is not None and len(query_ids)>0 and int(query_ids[0])==bos_token_id)
    return torch.cat((query[:leading],prefix,query[leading:]),dim=0),leading


def sequence_tensors(segments):
    return {f'sequence.{i}.{label}':value.detach().cpu().contiguous()
            for i, segment in enumerate(segments) for label, value in zip(LABELS, segment.tensors())}


def sequence_metadata(segments):
    return [dict(codec_port=s.codec_port, count=s.count, rank=s.left.shape[1], patches=len(s.patch_indices),
                 dtype=s.dtype, checksum=s.checksum) for s in segments]


def sequence_shapes(model, metadata):
    if not isinstance(metadata, list):raise ValueError('invalid sequence metadata')
    result = {}
    width = model.config.dream_memory['feature_vae']['latent_size'] if metadata else 0
    dim = model.config.text_config.hidden_size
    for i, row in enumerate(metadata):
        n, rank, patches = (row.get(k) for k in ('count', 'rank', 'patches'))
        if (type(n) is not int or n < 1 or type(rank) is not int or not 0 <= rank <= min(n, dim)
                or type(patches) is not int or not 0 <= patches <= n * dim):raise ValueError('invalid sequence shape metadata')
        if row.get('codec_port') not in model.dream_memory.feature_vae.heads or row.get('dtype') not in DTYPES:
            raise ValueError('invalid sequence codec metadata')
        shapes = ([n, width], [n, width], [n, 1], [n, 1], [n, rank], [rank, dim], [patches], [patches])
        result.update({f'sequence.{i}.{label}':shape for label, shape in zip(LABELS, shapes)})
    return result


def load_sequences(model, metadata, tensors):
    segments = []
    for i, row in enumerate(metadata):
        values = [tensors[f'sequence.{i}.{label}'] for label in LABELS]
        segment = SequenceMemory(row['codec_port'], LatentPortMemory(*values[:4]), *values[4:], row['dtype'], row['checksum'])
        validate_sequence(model, segment);segments.append(segment)
    return tuple(segments)
