import torch
from current_memory_fixture import dream_model
from lfm2_titans.physical_memory import PhysicalMemorySession


def test_accepted_dream_keeps_the_session_writable(tmp_path):
    torch.set_num_threads(2)
    model=dream_model().eval().requires_grad_(False)
    model.config.memory_checkpoint_id='dream-lifecycle-test'
    model._episodic_adapter_bank.checkpoint_id=model.config.memory_checkpoint_id
    from copy import deepcopy
    model.config.sequence_codec_port='native_input'
    model.dream_memory.feature_vae.heads['native_input']=deepcopy(model.dream_memory.feature_vae.heads['language_0'])
    before={n:p._version for n,p in model.named_parameters()}
    ids=torch.tensor([[1,3,5,7]])
    # Isolate the write/consolidate lifecycle. The public model opener also
    # requires a calibrated retrieval lens, covered by the address tests.
    session=PhysicalMemorySession(model,rank=2)
    session.observe(input_ids=ids,use_cache=False,logits_to_keep=1)
    session.learn(input_ids=ids,labels=ids,use_cache=False)
    session.seal(tmp_path/'original.safetensors')
    result=session.dream_consolidate(source_id='observed-unit',validator=lambda original,candidate:True)
    assert result['published']  # Lifecycle test, not a claim of retention quality.
    with session.pin() as (state,_):
        assert all(v.is_leaf and v.requires_grad for v in state.adapters.factors.values())
        commits=state.adapters.commits
    updated=session.learn(input_ids=ids,labels=ids,use_cache=False)
    assert updated['commits']==commits+1 and torch.isfinite(torch.tensor(updated['loss']))
    assert all(p._version==before[n] for n,p in model.named_parameters())
    session.seal(tmp_path/'after-dream.safetensors');session.close()
