"""Caller-local observation of native contextual features, without text labels."""
from contextlib import contextmanager
from contextvars import ContextVar

def install_concept_capture(model):
    model._native_concept_context=ContextVar(f'native_concepts_{id(model)}',default=None)
    for depth,name in model.config.language_ports.items():
        def capture(module,inputs,output,name=name):
            scope=model._native_concept_context.get()
            if scope is None:return
            values,mask=scope
            if output.shape[0]!=1:raise ValueError('one independent concept stream per scope')
            value=output[0]
            if mask is not None:
                if mask.shape!=(1,len(value)):raise ValueError('concept validity mask mismatch')
                value=value[mask[0].bool()]
            if name in values:raise ValueError('concept extraction expects one complete forward')
            values[name]=value.detach().float().cpu().clone()
        model.model.language_model.layers[int(depth)].register_forward_hook(capture)

@contextmanager
def capture_native_features(model,mask=None):
    if model._native_concept_context.get() is not None:raise RuntimeError('nested native concept extraction')
    values={};token=model._native_concept_context.set((values,mask))
    try:yield values
    finally:model._native_concept_context.reset(token)
