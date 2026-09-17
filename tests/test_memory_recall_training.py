from copy import deepcopy
import torch
import pytest
from current_memory_fixture import dream_model
from lfm2_titans.memory_recall_training import split_episodes,wrong_memory_rows
from lfm2_titans.sequence_memory import assemble_memory_query


def test_memory_framing_preserves_all_vectors_and_leading_bos():
    memory=torch.tensor([[10.,11.],[12.,13.]])
    query=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
    combined,leading=assemble_memory_query(memory,query,torch.tensor([1,7,8]),1)
    assert leading==1
    torch.testing.assert_close(combined,torch.cat((query[:1],memory,query[1:])),rtol=0,atol=0)
    plain,leading=assemble_memory_query(memory,query,torch.tensor([9,7,8]),1)
    assert leading==0
    torch.testing.assert_close(plain,torch.cat((memory,query)),rtol=0,atol=0)


def inputs():
    ids=torch.tensor([[1,29,1,5,6,7,8,30,29,2,4,30,29,1,9,10,30,29,2,12,13,30],
                      [1,29,1,17,18,19,20,30,29,2,4,30,29,1,9,11,30,29,2,21,22,30]])
    labels=torch.full_like(ids,-100);labels[:,-3:]=ids[:,-3:]
    return dict(input_ids=ids,labels=labels,attention_mask=torch.ones_like(ids))


def recall_model():
    model=dream_model().train();model.config.native_joint_sft=True
    model.config.native_memory_recall=dict(user_header_ids=[29,1],turn_end_id=30)
    model.config.sequence_codec_port='native_input'
    model.dream_memory.feature_vae.heads['native_input']=deepcopy(model.dream_memory.feature_vae.heads['language_0'])
    return model


def test_episode_boundary_excludes_query_and_answer_from_write():
    batch=inputs();spec=dict(user_header_ids=[29,1],turn_end_id=30)
    episodes=split_episodes(batch['input_ids'],batch['labels'],batch['attention_mask'],spec)
    assert episodes[0]['source'].tolist()==[5,6,7,8]
    assert episodes[1]['source'].tolist()==[17,18,19,20]
    assert episodes[0]['query'].tolist()==[1,29,1,9,10,30,29,2,12,13,30]
    assert wrong_memory_rows(episodes)==[1,0]
    contaminated=batch['labels'].clone();contaminated[0,4]=6
    with pytest.raises(ValueError,match='mask_history'):split_episodes(batch['input_ids'],contaminated,batch['attention_mask'],spec)


def test_native_selected_logit_positions_preserve_loss_and_gradients():
    model=dream_model().train()
    ids=torch.tensor([[1,5,6,7,8,9],[1,11,12,13,14,15]])
    labels=torch.full_like(ids,-100);labels[0,-2:]=ids[0,-2:];labels[1,-3:]=ids[1,-3:]
    reference=model(input_ids=ids,labels=labels,use_memory=False,use_cache=False)
    parameters=tuple(p for p in model.model.language_model.parameters() if p.requires_grad)
    expected=torch.autograd.grad(reference.loss,parameters)
    row,col=(labels[:,1:]!=-100).nonzero(as_tuple=True);positions=torch.unique(col,sorted=True)
    subset=model(input_ids=ids,use_memory=False,use_cache=False,logits_to_keep=positions)
    logits=subset.logits[row,torch.searchsorted(positions,col)]
    loss=model.loss_function(logits=logits,labels=None,shift_labels=labels[row,col+1],vocab_size=32)
    actual=torch.autograd.grad(loss,parameters)
    torch.testing.assert_close(loss,reference.loss)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)


def test_actual_write_then_read_reaches_both_memories_without_source_cache():
    torch.set_num_threads(2);model=recall_model()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    observed=[]
    def capture(module,args,kwargs):
        observed.append(dict(ids=kwargs.get('input_ids'),embedding_shape=tuple(kwargs['inputs_embeds'].shape),
                             cache=kwargs.get('past_key_values'),use_cache=kwargs.get('use_cache')))
    handle=model.model.language_model.register_forward_pre_hook(capture,with_kwargs=True)
    batch=inputs();output=model(**batch)
    output.loss.backward();handle.remove()
    report=model._native_sft_memory.last
    assert len(observed)==5  # source; dual recall; fast-only; empty; wrong unit.
    assert all(row['cache'] is None and row['use_cache'] is False for row in observed)
    assert report['source_observation_tokens']==8 and report['stored_posterior_positions']==8
    assert report['query_or_answer_given_to_writer'] is False
    assert report['raw_source_given_to_read'] is False
    assert report['sequence_residual_patches_used_in_training'] is False
    for prefix in ['memory.','physical_memory.','dream_memory.feature_vae.heads.native_input.encoder',
                   'dream_memory.feature_vae.heads.native_input.decoder']:
        grads=[p.grad for name,p in model.named_parameters() if name.startswith(prefix) and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads),prefix
        assert sum(float(g.abs().sum()) for g in grads)>0,prefix


def test_changing_future_queries_and_answers_cannot_change_written_memory(monkeypatch):
    from lfm2_titans.multiport_connectome import MemoryBlock
    import lfm2_titans.memory_recall_training as objective
    model=recall_model();written=[];ffn=[];posteriors=[];original=MemoryBlock.commit
    def capture(block,**kwargs):
        state,receipt=original(block,**kwargs);written.append(state.fast.detach().clone())
        return state,receipt
    monkeypatch.setattr(MemoryBlock,'commit',capture)
    apply=model._episodic_adapter_bank.apply_gradients
    def capture_ffn(*args,**kwargs):
        state,receipt=apply(*args,**kwargs)
        ffn.append({k:v.detach().clone() for k,v in state.factors.items()})
        return state,receipt
    monkeypatch.setattr(model._episodic_adapter_bank,'apply_gradients',capture_ffn)
    encode=objective.encode_observations
    def capture_codes(*args,**kwargs):
        codes=encode(*args,**kwargs)
        posteriors.append([v.detach().clone() for c in codes.values() for v in c.tensors()])
        return codes
    monkeypatch.setattr(objective,'encode_observations',capture_codes)
    batch=inputs();model(**batch)
    changed={k:v.clone() for k,v in batch.items()}
    changed['input_ids'][:,-3:-1]=torch.tensor([[23,24],[25,26]])
    changed['labels'][:,-3:-1]=changed['input_ids'][:,-3:-1]
    changed['input_ids'][:,15]=torch.tensor([14,15])
    model(**changed)
    assert len(written)==4
    for a,b in zip(written[:2],written[2:]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(ffn[:2],ffn[2:]):
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    for a,b in zip(posteriors[:2],posteriors[2:]):
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_lora_compiled_writer_supports_two_phase_checkpointed_backward(tmp_path):
    import os,sys
    from types import SimpleNamespace
    os.environ['DISABLE_VERSION_CHECK']='1'
    if os.environ.get('LLAMAFACTORY_SRC'):sys.path.insert(0,os.environ['LLAMAFACTORY_SRC'])
    from llamafactory.hparams import FinetuningArguments,ModelArguments
    from llamafactory.model.adapter import init_adapter
    from lfm2_titans.sft_lora import apply_sft_lora,apply_training_scope
    from lfm2_titans.multiport_connectome import MultiportConnectome
    factory=SimpleNamespace(FinetuningArguments=FinetuningArguments,ModelArguments=ModelArguments,
        init_adapter=init_adapter,WORK=tmp_path)
    native=recall_model();old=native.memory
    options=dict(native.config.memory_parameters,compiled_execution=True,checkpoint_write_gradients=True,
        separate_write_key=True,learned_value=True,write_association='next')
    native.memory=MultiportConnectome(old.nodes,old.indices,old.specs,**options)
    native.memory.load_state_dict(old.state_dict(),strict=False);native.config.memory_parameters=options
    model,_,_,_,_=apply_sft_lora(native,factory,rank=2)
    native.config.native_memory_recall['train_scope']='memory'
    apply_training_scope(native)
    frozen={name:p.detach().clone() for name,p in native.model.named_parameters()}
    assert not any(p.requires_grad for p in native.model.parameters())
    native.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    for _ in range(2):
        model.zero_grad(set_to_none=True);result=model(**inputs());result.loss.backward()
        gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    for name,p in native.model.named_parameters():torch.testing.assert_close(p,frozen[name],rtol=0,atol=0)
