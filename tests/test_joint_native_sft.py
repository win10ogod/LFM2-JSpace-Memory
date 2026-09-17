from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from current_memory_fixture import dream_model
from lfm2_titans.native_sft_memory import capture_supervised_gradients
from lfm2_titans.sft_lora import WeightLoRALinear,shared_effective_weights


def test_removed_writer_and_reader_branches_cannot_be_selected():
    model=dream_model().eval().requires_grad_(False)
    ids=torch.tensor([[1,3,5]])
    for options in ({'memory_write_gradient_scope':'local'}, {'memory_read_mode':'latent'},
                    {'memory_record':0,'memory_start':0}):
        with pytest.raises(ValueError):model(input_ids=ids,**options)
    assert not hasattr(model.memory.begin(model.memory.initial_state()),'independent_write')
    assert not hasattr(model.dream_memory.weight_vae,'decode_fast')


def test_shared_initial_graph_read_matches_independent_rows_and_larger_tiles():
    from lfm2_titans.batched_memory import BatchedMemoryBlock
    model=dream_model().train();model.memory.residual_recall=False
    units=[model.initial_physical_memory(create_graph=True) for _ in range(3)]
    x=torch.randn(3,7,16,requires_grad=True);name='language_0'
    reference=torch.cat([model.memory.begin(u.graph).residual(name,x[i:i+1]) for i,u in enumerate(units)])
    expected=torch.autograd.grad(reference.square().mean(),(x,model.memory.slow_weights))
    model.memory.feature_chunk_size=256
    block=BatchedMemoryBlock(model,units,torch.ones(3,7,dtype=torch.long),fresh_rows=[True]*3)
    actual=block.residual(name,x)
    gradients=torch.autograd.grad(actual.square().mean(),(x,model.memory.slow_weights))
    torch.testing.assert_close(actual,reference,atol=1e-6,rtol=1e-5)
    for a,b in zip(gradients,expected):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)


def test_supervised_capture_preserves_parameter_gradients_and_normalizes_rows():
    from lfm2_titans.episodic_adapters import AdapterMemory
    torch.manual_seed(42);batch=4;width=5;vocab=7
    factors=[torch.randn(width,vocab,requires_grad=True) for _ in range(batch)]
    x=torch.randn(batch,6,width);logits=torch.stack([x[i]@factors[i] for i in range(batch)])
    labels=torch.randint(vocab,(batch,6));labels[:,:2]=-100;labels[-1,-2:]=-100
    target=labels[:,1:];normalizer=(target!=-100).sum()
    loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,vocab),target.reshape(-1),ignore_index=-100)
    expected=torch.autograd.grad(loss,factors,retain_graph=True)
    units=[SimpleNamespace(adapters=AdapterMemory({'weight':f},{'weight':torch.zeros_like(f)},
        {'weight':torch.zeros_like(f)})) for f in factors]
    pending=capture_supervised_gradients(units,labels,normalizer)
    loss.backward()
    for i,(factor,reference,item) in enumerate(zip(factors,expected,pending)):
        torch.testing.assert_close(factor.grad,reference)
        torch.testing.assert_close(item['gradients']['weight'],reference*normalizer/(target[i]!=-100).sum())


def test_effective_weight_cache_keeps_values_gradients_and_grad_modes():
    torch.manual_seed(4);layer=WeightLoRALinear(nn.Linear(5,6),2,4)
    with torch.no_grad():layer.adapter_B.normal_()
    x=torch.randn(3,5)
    reference=torch.nn.functional.linear(x,layer.weight)+torch.nn.functional.linear(x,layer.weight)
    grads=torch.autograd.grad(reference.square().sum(),(layer.adapter_A,layer.adapter_B))
    with shared_effective_weights():
        with torch.no_grad():cold=layer.weight
        first=layer.weight;second=layer.weight
        assert first is second and first.requires_grad and not cold.requires_grad
        result=torch.nn.functional.linear(x,first)+torch.nn.functional.linear(x,second)
        actual=torch.autograd.grad(result.square().sum(),(layer.adapter_A,layer.adapter_B))
    torch.testing.assert_close(result,reference)
    for a,b in zip(actual,grads):torch.testing.assert_close(a,b)
    with shared_effective_weights():assert layer.weight is not first


def test_joint_native_batch_trains_vision_and_memory_without_windows():
    torch.set_num_threads(2);model=dream_model().train()
    model.config.native_joint_sft=True;model.config.native_sft_aux_features=3
    model.config.sequence_codec_port='native_input'
    model.dream_memory.feature_vae.heads['native_input']=deepcopy(model.dream_memory.feature_vae.heads['language_0'])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    ids=torch.tensor([[31,1,3,5],[1,4,7,9]])
    labels=ids.clone();labels[:,:2]=-100
    calls=[];hook=model.lm_head.register_forward_hook(lambda *args:calls.append(1))
    output=model(input_ids=ids,labels=labels,attention_mask=torch.ones_like(ids),use_cache=False,
        pixel_values=torch.randn(1,4,12),pixel_attention_mask=torch.ones(1,4,dtype=torch.long),
        spatial_shapes=torch.tensor([[2,2]]))
    output.loss.backward();hook.remove()
    assert calls==[1] and output.logits.shape==(2,4,32)
    assert model._native_sft_memory.last['source_autograd_traversals']==0
    assert model._native_sft_memory.last['training_chunks']==0
    assert not hasattr(model._native_sft_memory,'sessions')
    model._native_sft_memory.finish_backward()
    assert model._native_sft_memory.ffn_replay and not model._native_sft_memory.pending
    for prefix in ('model.language_model.','model.vision_tower.','model.multi_modal_projector.',
                   'memory.','physical_memory.','dream_memory.feature_vae.heads.native_input.',
                   'dream_memory.weight_vae.','dream_memory.controller.'):
        gradients=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients),prefix
        assert sum(float(g.abs().sum()) for g in gradients)>0,prefix


def test_joint_lora_gradient_checkpointing_can_recompute_both_backwards(tmp_path):
    import os,sys
    os.environ['DISABLE_VERSION_CHECK']='1'
    if os.environ.get('LLAMAFACTORY_SRC'):
        sys.path.insert(0,os.environ['LLAMAFACTORY_SRC'])
    from llamafactory.hparams import FinetuningArguments,ModelArguments
    from llamafactory.model.adapter import init_adapter
    from lfm2_titans.sft_lora import apply_sft_lora
    factory=SimpleNamespace(FinetuningArguments=FinetuningArguments,ModelArguments=ModelArguments,
        init_adapter=init_adapter,WORK=tmp_path)
    native=dream_model().train();native.config.native_joint_sft=True;native.config.native_sft_aux_features=3
    from lfm2_titans.multiport_connectome import MultiportConnectome
    old=native.memory
    options=dict(native.config.memory_parameters,compiled_execution=True,checkpoint_write_gradients=True,
        separate_write_key=True,learned_value=True,write_association='next')
    native.memory=MultiportConnectome(old.nodes,old.indices,old.specs,**options)
    native.memory.load_state_dict(old.state_dict(),strict=False)
    native.config.memory_parameters=options
    model,_,_,_,_=apply_sft_lora(native,factory,rank=2)
    native.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    ids=torch.tensor([[1,3,5,7],[1,4,6,8]])
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        result=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids.clone(),use_cache=False)
        result.loss.backward()
        native._native_sft_memory.finish_backward()
        native._native_sft_memory.save_replay(tmp_path)
        prior=[v.clone() for v in native._native_sft_memory.ffn_replay]
        native._native_sft_memory.ffn_replay=[]
        native._native_sft_memory.load_replay(tmp_path)
        for a,b in zip(prior,native._native_sft_memory.ffn_replay):torch.testing.assert_close(a,b,rtol=0,atol=0)
        grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
