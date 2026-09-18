from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.nn import functional as F
from current_memory_fixture import dream_model
from test_memory_recall_training import recall_model,inputs
from lfm2_titans.physical_memory import PhysicalMemorySession
from lfm2_titans.autonomous_memory import (comparison_signals,policy_objective,
    prediction_divergence,source_prediction,autonomous_consolidate)


def frozen():
    m=dream_model().eval().requires_grad_(False)
    m.config.sequence_codec_port='native_input'
    m.dream_memory.feature_vae.heads['native_input']=deepcopy(m.dream_memory.feature_vae.heads['language_0'])
    m.config.memory_checkpoint_id='autonomy-test'
    m._episodic_adapter_bank.checkpoint_id=m.config.memory_checkpoint_id
    return m


def test_every_source_position_kl_matches_unchunked_native_head():
    m=frozen();a=torch.randn(137,16);b=torch.randn_like(a)
    with torch.no_grad():
        expected=F.kl_div(m.lm_head(a).float().log_softmax(-1),m.lm_head(b).float().log_softmax(-1),
            log_target=True,reduction='batchmean')
        actual=prediction_divergence(m,a,b,chunk_size=13)
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-5)


def test_normal_save_and_generate_automatically_consolidate_and_compare(tmp_path):
    m=frozen();before={n:p._version for n,p in m.named_parameters()}
    s=PhysicalMemorySession(m,rank=2);ids=torch.tensor([[1,3,5,7]])
    with patch('lfm2_titans.autonomous_memory.autonomous_consolidate',wraps=autonomous_consolidate) as call:
        s.observe(input_ids=ids,use_cache=False,logits_to_keep=1)
        s.learn(input_ids=ids,labels=ids,use_cache=False)
        s.seal(tmp_path/'first.safetensors')
        assert call.call_count==1 and s.last_consolidation['positions']==4
        assert s.last_consolidation['query_or_answer_seen'] is False
        out=s.generate(input_ids=torch.tensor([[1,9,10]]),max_new_tokens=2,do_sample=False)
        assert call.call_count==1  # The same generation is not replayed repeatedly.
    assert out.shape[0]==1 and out.shape[1]<=5
    decision=s.last_memory_action
    assert decision['action'] in ('complete','compressed','physical')
    assert decision['query_only'] and decision['comparison_prefills']==3
    assert decision['generation_calls']==1
    assert all(p._version==before[n] for n,p in m.named_parameters())
    s.close()


def test_plain_model_generate_routes_through_context_owned_archive_and_keeps_options():
    m=frozen();received=[]
    def generate(query,**kwargs):
        received.append((query,kwargs));return {'tokens':query['input_ids']}
    archive=SimpleNamespace(model=m,generate=generate)
    ids=torch.tensor([[1,7]])
    with m.use_memory_archive(archive,top_k=3,index_options={'device':'cpu'}):
        out=m.generate(ids,max_new_tokens=4096,top_k=57,temperature=.8,do_sample=True)
    assert torch.equal(out,ids) and m._archive_context.get() is None
    _,r=received[0]
    assert r['top_k']==3 and r['generation_inputs']['top_k']==57
    assert r['generation_inputs']['max_new_tokens']==4096
    assert r['generation_inputs']['temperature']==.8


def test_concurrent_callers_keep_their_archive_bindings_separate():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    m=frozen();barrier=Barrier(2)
    def caller(marker):
        archive=SimpleNamespace(model=m,generate=lambda *a,**k:{'tokens':torch.tensor([[marker]])})
        with m.use_memory_archive(archive):
            barrier.wait(timeout=10)
            result=m.generate(torch.tensor([[1,3]]),max_new_tokens=4)
        assert m._archive_context.get() is None
        return int(result[0,0])
    with ThreadPoolExecutor(2) as workers:assert list(workers.map(caller,[11,17]))==[11,17]


def test_future_answers_cannot_change_read_or_consolidation_policy_inputs(monkeypatch):
    import lfm2_titans.memory_recall_training as objective
    m=recall_model().eval();seen=[];original=objective.rank_reads
    def capture(*args,**kwargs):
        result=original(*args,**kwargs);seen.append(result[2].clone());return result
    monkeypatch.setattr(objective,'rank_reads',capture)
    batch=inputs()
    with torch.no_grad():m(**batch)
    n=len(seen);changed={k:v.clone() for k,v in batch.items()}
    changed['input_ids'][:,-3:-1]=torch.tensor([[23,24],[25,26]])
    changed['labels'][:,-3:-1]=changed['input_ids'][:,-3:-1]
    with torch.no_grad():m(**changed)
    assert n==4 and len(seen)==2*n
    for a,b in zip(seen[:n],seen[n:]):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)


def test_policy_target_is_detached_but_existing_controller_learns():
    m=dream_model();c=m.dream_memory.controller
    signals=comparison_signals(torch.tensor([[1.,2.,3.],[1.,1.,1.],[10.,0.,0.]]))
    optimizer=torch.optim.Adam(c.parameters(),lr=.02)
    losses=torch.tensor([3.,.1,5.],requires_grad=True)
    for _ in range(80):
        optimizer.zero_grad();score,_=c(signals,c.initial_state())
        loss=policy_objective(score,losses);loss.backward();optimizer.step()
    assert int(c(signals,c.initial_state())[0].argmax())==1
    assert losses.grad is None


def test_soft_policy_cross_entropy_floor_is_not_failure_to_converge():
    import math
    from lfm2_titans.autonomous_memory import policy_statistics
    tied=torch.tensor([[6.,6.],[3.,3.]])
    scores=torch.zeros_like(tied,requires_grad=True)
    loss=policy_objective(scores,tied);loss.backward()
    stats=policy_statistics(scores,tied)
    assert abs(float(loss.detach())-math.log(2))<1e-6
    assert stats['excess_kl']==0 and torch.equal(scores.grad,torch.zeros_like(scores))
    decisive=torch.tensor([[.1,6.,7.]])
    untrained=policy_statistics(torch.zeros_like(decisive),decisive)
    assert untrained['target_entropy']<1e-6 and untrained['excess_kl']>1.


def test_policy_temperature_preserves_choices_and_exposes_raw_loss():
    from lfm2_titans.autonomous_memory import policy_statistics
    scores=torch.tensor([[.008,.001,0.],[-.002,.004,.001]],requires_grad=True)
    quality=torch.tensor([[.2,6.,6.],[6.,.2,6.]])
    scaled=policy_objective(scores,quality,score_temperature=.01)
    stats=policy_statistics(scores,quality,score_temperature=.01)
    torch.testing.assert_close(scaled,torch.tensor(stats['cross_entropy']))
    assert stats['raw_cross_entropy']>stats['cross_entropy']
    assert torch.equal(scores.argmax(-1),(scores/.01).argmax(-1))
    scaled.backward();assert torch.isfinite(scores.grad).all()
    import pytest
    with pytest.raises(ValueError):policy_objective(scores,quality,score_temperature=0.)


def test_batched_dream_matches_independent_replay_and_retention(tmp_path):
    from lfm2_titans.autonomous_memory import replay_candidate,replay_candidates,consolidation_observations,consolidation_observations_batch
    m=frozen();units=[]
    for ids in (torch.tensor([[1,3,5,7]]),torch.tensor([[1,11,13,9,8,4]])):
        s=PhysicalMemorySession(m,rank=2)
        s.observe(input_ids=ids,use_cache=False,logits_to_keep=1)
        s.learn(input_ids=ids,labels=ids,use_cache=False)
        with s.pin() as (u,_):units.append(u)
        s.dirty=False;s.close()  # Isolated in-memory fixture; no publication.
    reference=[replay_candidate(m,u)[0] for u in units]
    actual=replay_candidates(m,units)
    for a,b in zip(actual,reference):
        torch.testing.assert_close(a.graph.fast,b.graph.fast,atol=1e-6,rtol=1e-5)
        for key in a.adapters.factors:torch.testing.assert_close(a.adapters.factors[key],b.adapters.factors[key])
    expected=[consolidation_observations(m,u,c) for u,c in zip(units,reference)]
    batched=consolidation_observations_batch(m,units,actual)
    for a,b in zip(batched,expected):
        for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)


def test_accepted_automatic_dream_preserves_prior_physical_unit_on_disk(tmp_path,monkeypatch):
    from dataclasses import replace
    import json
    from safetensors.torch import save_file
    from lfm2_titans.native_concepts import NativeConceptLens
    m=frozen();lens_dir=tmp_path/'lens';lens_dir.mkdir()
    save_file({str(k):torch.eye(16) for k in m.config.language_ports},str(lens_dir/'jacobians.safetensors'))
    (lens_dir/'result.json').write_text(json.dumps(dict(protocol=dict(checkpoint_id=m.config.memory_checkpoint_id),
        body_fixed=True,native_outputs_identical=True)))
    lens=NativeConceptLens(m,lens_dir,sparsity=4)
    monkeypatch.setattr(m,'configured_concept_lens',lambda:lens)
    archive=m.open_physical_archive(tmp_path/'archive');s=archive.session
    ids=torch.tensor([[1,3,5,7]]);s.observe(input_ids=ids,use_cache=False,logits_to_keep=1)
    with s.pin() as (original,_):expected=original.graph.fast.clone()
    def accept(model,unit):
        candidate=replace(unit,graph=replace(unit.graph,fast=unit.graph.fast+1)).detach()
        return candidate,dict(accepted=True,source_only=True)
    # Exercise the successful atomic-publication path; retention quality is
    # covered separately, not inferred from this deliberately forced decision.
    monkeypatch.setattr('lfm2_titans.autonomous_memory.autonomous_consolidate',accept)
    receipt=archive.append(start_new=False)
    previous=receipt['autonomous_consolidation']['preserved_previous_unit']['unit_path']
    restored=s.bank.load_unit(previous)
    torch.testing.assert_close(restored.graph.fast,expected,rtol=0,atol=0)
    torch.testing.assert_close(s.bank.load_unit(receipt['unit_path']).graph.fast,expected+1,rtol=0,atol=0)
    from pathlib import Path
    assert Path(previous).stem<receipt['unit_id']
    assert archive.query(dict(input_ids=ids),top_k=1)['matches'][0]['unit_id']==receipt['unit_id']
    archive.close();lens.close()
