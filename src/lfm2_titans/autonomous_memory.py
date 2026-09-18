"""Model-owned comparison and source-only Dream consolidation.

The existing MaleCNS controller ranks reads using query-prefix distributions.
Supervised answers supply training targets only; they never become policy
inputs or enter a memory writer. No new learned parameters are introduced.
"""
from dataclasses import replace
import math
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from .sequence_memory import prepare_ordered_inputs,decode_segment
from .batched_memory import BatchedMemoryBlock
from .episodic_adapters import AdapterBatch

READ_ACTIONS=('complete','compressed','physical')
POLICY_SIGNALS=('predictive_entropy','distribution_disagreement','peak_probability',
                'source_prediction_error','relative_read_cost','candidate_id')


def comparison_signals(logits,*,errors=None,costs=None):
    """[candidate,vocab] -> six observable features, without answer labels."""
    with torch.no_grad():
        logp=logits.detach().float().log_softmax(-1);p=logp.exp()
        average=p.mean(0).clamp_min(1e-30)
        entropy=-(p*logp).sum(-1)/math.log(p.shape[-1])
        disagreement=(p*(logp-average.log())).sum(-1)
        peak=p.amax(-1)
        errors=torch.zeros_like(peak) if errors is None else torch.as_tensor(errors,device=p.device,dtype=p.dtype)
        costs=torch.zeros_like(peak) if costs is None else torch.as_tensor(costs,device=p.device,dtype=p.dtype)
        identity=torch.arange(len(p),device=p.device,dtype=p.dtype)
        return torch.stack((entropy,disagreement,peak,errors,costs,identity),dim=-1)


def rank_reads(model,logits,*,errors=None,costs=None,state=None):
    signals=comparison_signals(logits,errors=errors,costs=costs)
    controller=model.dream_memory.controller
    with torch.autocast(device_type=signals.device.type,enabled=False):
        scores,updated=controller(signals,state if state is not None else controller.initial_state())
    return scores,updated,signals


def policy_objective(scores,answer_losses,temperature=.25,score_temperature=1.):
    """Distill measured positive recall quality; never worsen a control path."""
    if not all(math.isfinite(t) and t>0 for t in (temperature,score_temperature)):
        raise ValueError('policy temperatures must be finite and positive')
    targets=(-answer_losses.detach()/temperature).softmax(-1)
    return -(targets*(scores.float()/score_temperature).log_softmax(-1)).sum(-1).mean()


@torch.no_grad()
def policy_statistics(scores,answer_losses,temperature=.25,score_temperature=1.):
    """Separate the moving soft-target entropy floor from optimization error."""
    target_logp=(-answer_losses.detach().float()/temperature).log_softmax(-1)
    target=target_logp.exp();predicted=(scores.detach().float()/score_temperature).log_softmax(-1)
    entropy=-(target*target_logp).sum(-1).mean()
    cross_entropy=-(target*predicted).sum(-1).mean()
    return dict(target_entropy=float(entropy),cross_entropy=float(cross_entropy),
        raw_cross_entropy=float(-(target*scores.detach().float().log_softmax(-1)).sum(-1).mean()),
        excess_kl=float(F.kl_div(predicted,target_logp,log_target=True,reduction='batchmean')
            if predicted.ndim==2 else F.kl_div(predicted,target_logp,log_target=True,reduction='sum')))


def compressed_prefix(model,sequences):
    values=[]
    for s in sequences:
        head=model.dream_memory.feature_vae.heads[s.codec_port];p=s.posterior
        device=next(head.parameters()).device
        with torch.autocast(device_type=device.type,enabled=False):
            values.append(head.decoder(p.mu.to(device))*p.scale.to(device)+p.mean.to(device))
    return torch.cat(values)


def prepare_read(model,sequences,inputs,action,**limits):
    if action not in READ_ACTIONS:raise ValueError('unknown internal memory action')
    if action=='physical':return dict(inputs)
    prepared=prepare_ordered_inputs(model,sequences,inputs,**limits)
    if action=='compressed':
        ids=inputs['input_ids'];bos=getattr(model.config.text_config,'bos_token_id',None)
        leading=int(bos is not None and int(ids[0,0])==bos)
        value=prepared['inputs_embeds'].clone();prefix=compressed_prefix(model,sequences).to(value)
        value[:,leading:leading+len(prefix)]=prefix
        prepared['inputs_embeds']=value
    return prepared


@torch.no_grad()
def autonomous_generate(model,bank,adapters,graph,sequences,inputs,*,state=None,**options):
    """Compare query-only prefills, then generate with the model's chosen read.

    Generation arguments and the output budget are passed through unchanged.
    Comparisons never generate candidate answers or examine reference answers.
    """
    inputs=dict(inputs,**options)
    forbidden=('inputs_embeds','past_key_values','position_ids','pixel_values','pixel_attention_mask','spatial_shapes')
    if any(inputs.get(k) is not None for k in forbidden):
        raise ValueError('autonomous recall takes a fresh text query; stored memories may include vision')
    prepared=[];logits=[]
    limits={k:options.get(k,inputs.get(k)) for k in ('max_new_tokens','max_length')}
    with bank.use(adapters):
        for action in READ_ACTIONS:
            values=prepare_read(model,sequences,inputs,action,**limits)
            # Generation-only options do not belong in comparison forwards.
            forward={k:v for k,v in values.items() if k in ('input_ids','inputs_embeds','attention_mask')}
            if 'inputs_embeds' in forward:forward.pop('input_ids',None)
            out=model(**forward,memory_state=graph,use_cache=False,logits_to_keep=1)
            logits.append(out.logits[0,-1]);prepared.append(values)
        scores,updated,signals=rank_reads(model,torch.stack(logits))
        choice=int(scores.argmax())
        tokens=model.generate(**prepared[choice],memory_state=graph)
    return tokens,updated.detach(),dict(action=READ_ACTIONS[choice],scores=scores.cpu().tolist(),
        signals=signals.cpu().tolist(),signal_names=POLICY_SIGNALS,query_only=True,
        comparison_prefills=len(READ_ACTIONS),generation_calls=1)


def replay_candidate(model,unit,*,create_graph=False):
    """Same source-only reconstruction/replay/graph commit in training and use."""
    if not unit.sequences:raise ValueError('Dream requires observed ordered memory')
    count=sum(s.count for s in unit.sequences)
    if count>model.config.text_config.max_position_embeddings:raise ValueError('Dream exceeds native context; no source truncated')
    with torch.set_grad_enabled(create_graph):
        candidate,terms=model.dream_memory.reconstruct_unit(model,unit,sample=False)
    # First-order replay: do not retain a backbone tape for source feature
    # collection. The reconstructed physical weights and graph commit retain
    # their outer gradients; the later fresh query trains their useful effect.
    with torch.no_grad(),model._episodic_adapter_bank.use(candidate.adapters):
        embedding=model.get_input_embeddings().weight
        prefix=compressed_prefix(model,unit.sequences).to(device=embedding.device,dtype=embedding.dtype)
        out=model(inputs_embeds=prefix[None],memory_state=candidate.graph,dream_capture=True,
                  use_cache=False,logits_to_keep=1)
    observed=out.dream_features
    visual_replay(model,unit,observed)
    block=model.memory.begin(candidate.graph)
    for name,value in observed.items():block.observe(name,name,value)
    graph,receipt=block.commit(create_graph=create_graph)
    result=replace(candidate,graph=graph,address=None)
    return (result if create_graph else result.detach()),dict(codec=terms,graph=receipt,
        source_positions=count,query_or_answer_seen=False,replay='posterior_code_native_forward')


@torch.no_grad()
def visual_replay(model,unit,observed):
    """Retain every visual position; a later functional gate decides adoption."""
    name=model.config.vision_port
    if name in (unit.latents or {}):
        port=unit.latents[name];head=model.dream_memory.feature_vae.heads[name]
        device=next(head.parameters()).device
        with torch.autocast(device_type=device.type,enabled=False):
            observed[name]=head.decoder(port.mu.to(device))*port.scale.to(device)+port.mean.to(device)


@torch.no_grad()
def source_batch(model,units,*,compressed,capture=False):
    """One native padded batch, independent units, every valid source position."""
    from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlForConditionalGeneration
    embedding=model.get_input_embeddings().weight
    prefixes=[compressed_prefix(model,u.sequences) if compressed else
              torch.cat([decode_segment(model,s) for s in u.sequences]) for u in units]
    values=pad_sequence([x.to(device=embedding.device,dtype=embedding.dtype) for x in prefixes],batch_first=True)
    lengths=torch.tensor([len(x) for x in prefixes],device=values.device)
    mask=torch.arange(values.shape[1],device=values.device)[None]<lengths[:,None]
    blocks=BatchedMemoryBlock(model,units,torch.zeros_like(mask,dtype=torch.long),observation_mask=mask)
    token=model._memory_context.set((blocks,capture,mask))
    try:
        with model._episodic_adapter_bank.use(AdapterBatch(tuple(u.adapters for u in units))):
            output=Lfm2VlForConditionalGeneration.forward(model,inputs_embeds=values,attention_mask=mask,
                logits_to_keep=1,output_hidden_states=True,use_cache=False,return_dict=True)
    finally:model._memory_context.reset(token)
    hidden=[output.hidden_states[-1][i,:len(x)] for i,x in enumerate(prefixes)]
    last=model.lm_head(torch.stack([x[-1] for x in hidden]))
    observed=[b.native_observations for b in blocks.blocks]
    return hidden,last,observed


def replay_candidates(model,units,*,create_graph=False):
    with torch.set_grad_enabled(create_graph):
        candidates=[model.dream_memory.reconstruct_unit(model,u,sample=False)[0] for u in units]
    _,_,observed=source_batch(model,candidates,compressed=True,capture=True)
    results=[]
    for unit,candidate,features in zip(units,candidates,observed):
        visual_replay(model,unit,features)
        block=model.memory.begin(candidate.graph)
        for name,value in features.items():block.observe(name,name,value)
        graph,_=block.commit(create_graph=create_graph)
        result=replace(candidate,graph=graph,address=None)
        results.append(result if create_graph else result.detach())
    return results


@torch.no_grad()
def consolidation_observations_batch(model,units,candidates):
    teacher,_,_=source_batch(model,units,compressed=False)
    before,before_logits,_=source_batch(model,units,compressed=True)
    after,after_logits,_=source_batch(model,candidates,compressed=True)
    retained,_,_=source_batch(model,candidates,compressed=False)
    results=[]
    for i in range(len(units)):
        kl=lambda hidden:prediction_divergence(model,hidden,teacher[i])
        results.append((torch.stack((before_logits[i],after_logits[i])),
            torch.stack((kl(before[i]),kl(after[i]))),kl(retained[i])))
    return results


@torch.no_grad()
def source_prediction(model,unit,*,compressed):
    """All observed positions; no token recovery or generated training labels."""
    prefix=compressed_prefix(model,unit.sequences) if compressed else torch.cat([decode_segment(model,s) for s in unit.sequences])
    with model._episodic_adapter_bank.use(unit.adapters):
        embedding=model.get_input_embeddings().weight
        out=model(inputs_embeds=prefix[None].to(device=embedding.device,dtype=embedding.dtype),
                  memory_state=unit.graph,use_cache=False,logits_to_keep=1,output_hidden_states=True)
    return out.hidden_states[-1][0],out.logits[0,-1].detach()


def prediction_divergence(model,student,teacher,chunk_size=64):
    """Exact all-position KL with a bounded output-head buffer, no sampling."""
    total=student.new_zeros((),dtype=torch.float32)
    for start in range(0,len(student),chunk_size):
        logp=model.lm_head(student[start:start+chunk_size]).float().log_softmax(-1)
        reference=model.lm_head(teacher[start:start+chunk_size]).float().log_softmax(-1)
        total+=F.kl_div(logp,reference,log_target=True,reduction='sum')
    return (total/len(student)).clamp_min(0)


@torch.no_grad()
def consolidation_observations(model,unit,candidate):
    teacher,_=source_prediction(model,unit,compressed=False)
    before,before_logits=source_prediction(model,unit,compressed=True)
    after,after_logits=source_prediction(model,candidate,compressed=True)
    retained,_=source_prediction(model,candidate,compressed=False)
    kl=lambda hidden:prediction_divergence(model,hidden,teacher)
    original_error,new_error,retention=kl(before),kl(after),kl(retained)
    return torch.stack((before_logits,after_logits)),torch.stack((original_error,new_error)),retention


@torch.no_grad()
def autonomous_consolidate(model,unit):
    """Model ranking plus source-retention verification, with immutable history.

    The check measures source prediction agreement, not a promise about every
    future question. Explicit runtime receipts preserve that distinction.
    """
    candidate,receipt=replay_candidate(model,unit,create_graph=False)
    logits,errors,retention=consolidation_observations(model,unit,candidate)
    original_error,new_error=errors
    scores,_,signals=rank_reads(model,logits,errors=errors)
    tolerance=getattr(model.config,'dream_retention_kl',.01)
    verified=bool(torch.isfinite(new_error) & torch.isfinite(retention) &
                  (new_error<original_error) & (retention<=tolerance))
    selected=int(scores.argmax())==1;accept=selected and verified
    return (candidate if accept else unit),dict(accepted=accept,model_selected_replay=selected,
        source_kl_before=float(original_error),source_kl_after=float(new_error),retention_kl=float(retention),
        retention_kl_tolerance=tolerance,positions=receipt['source_positions'],
        query_or_answer_seen=False,criterion='source predictive improvement and retention',
        scores=scores.cpu().tolist(),signals=signals.cpu().tolist())
