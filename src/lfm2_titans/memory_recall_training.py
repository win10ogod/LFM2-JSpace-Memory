"""Source-only writes followed by fresh-context, memory-dependent query loss.

This objective uses existing graph/FFN weights and VAE heads; it adds no model
parameters. LlamaFactory still owns examples, batches, backward and optimization.
Native multi-turn boundaries separate observations from queries. No SFT windows.
"""
from dataclasses import replace
from contextlib import contextmanager
import time
import math
import hashlib
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from .batched_memory import BatchedMemoryBlock
from .episodic_adapters import AdapterBatch
from .latent_memory import encode_observations,LatentPortMemory
from .sft_lora import warm_effective_weights
from .sequence_memory import assemble_memory_query,SequenceMemory,feature_checksum,decode_segment
from .autonomous_memory import rank_reads,policy_objective,policy_statistics,replay_candidates,consolidation_observations_batch


def split_episodes(input_ids,labels,attention_mask,spec):
    """Use native role boundaries; no answer-dependent source selection."""
    header=spec['user_header_ids'];end=spec['turn_end_id'];episodes=[]
    for ids,target,valid in zip(input_ids,labels,attention_mask):
        ids=ids[valid.bool()];target=target[valid.bool()];values=ids.tolist()
        starts=[i for i in range(len(values)-len(header)+1) if values[i:i+len(header)]==header]
        if len(starts)!=2:raise ValueError('Memory recall SFT requires exactly an observation turn and a query turn')
        first,last=starts;source_start=first+len(header)
        try:source_end=values.index(end,source_start)
        except ValueError:raise ValueError('Observation turn is incomplete') from None
        if source_end>=last or source_end-source_start<2:raise ValueError('Invalid observation/query boundaries')
        if (target[:last]!=-100).any():raise ValueError('Use native mask_history=True; observation/acknowledgment labels must be masked')
        if not (target[last:]!=-100).any():raise ValueError('Query has no supervised answer')
        prefix=ids[:first]
        query=torch.cat((prefix,ids[last:]))
        query_labels=torch.cat((torch.full_like(prefix,-100),target[last:]))
        episodes.append(dict(source=ids[source_start:source_end],query=query,labels=query_labels))
    return episodes


def per_row_nll(output,labels,positions=None):
    mask=labels[:,1:]!=-100;row,col=mask.nonzero(as_tuple=True)
    index=col if positions is None else torch.searchsorted(positions,col)
    values=F.cross_entropy(output.logits[row,index].float(),labels[row,col+1],reduction='none')
    return values.new_zeros(len(labels)).scatter_add(0,row,values)/mask.sum(1).clamp_min(1)


def wrong_memory_rows(episodes):
    n=len(episodes)
    for offset in range(1,n):
        chosen=[(i+offset)%n for i in range(n)]
        if all(not torch.equal(episodes[i]['source'],episodes[j]['source']) for i,j in enumerate(chosen)):
            return chosen
    return [next((j for j in range(n) if j!=i and not torch.equal(e['source'],episodes[j]['source'])),None)
            for i,e in enumerate(episodes)]


def store_training_sequence(model,features,posterior):
    """Production unit format, with rank-zero corrections instead of an SVD search.

    Every differing element is stored as a numerical patch. The production
    decoder verifies all features. This read is detached; the separate
    compressed-read loss supplies the VAE's task gradients, with no fake
    straight-through gradient through exact reconstruction.
    """
    with torch.no_grad():
        port=model.config.sequence_codec_port;head=model.dream_memory.feature_vae.heads[port]
        code=posterior.detach()
        with torch.autocast(device_type=features.device.type,enabled=False):
            base=(head.decoder(code.mu)*code.scale+code.mean).to(features.dtype)
        bits=torch.int32 if features.dtype==torch.float32 else torch.int16
        patch=(base.contiguous().view(bits).reshape(-1)!=features.contiguous().view(bits).reshape(-1)).nonzero().flatten()
        segment=SequenceMemory(port,code,features.new_empty((len(features),0),dtype=torch.float32),
            features.new_empty((0,features.shape[-1]),dtype=torch.float32),patch,
            features.detach().reshape(-1)[patch].float(),str(features.dtype).removeprefix('torch.'),feature_checksum(features))
        decode_segment(model,segment)
        return segment


def positive_recall_objective(complete,compressed,reconstruction,spec):
    """Only positive supervised paths enter optimization; controls are metrics."""
    return complete+spec.get('compressed_recall_weight',.25)*compressed+spec.get('reconstruction_weight',.05)*reconstruction


def joint_recall_objective(complete,compressed,physical,consolidated,read_policy,consolidation_policy,reconstruction,spec,*,source_codec=None):
    weights={k:spec.get(k,v) for k,v in dict(compressed_recall_weight=.25,physical_recall_weight=.25,
        consolidated_recall_weight=.25,memory_policy_weight=.1,reconstruction_weight=.05,source_codec_weight=0.).items()}
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in weights.values()):
        raise ValueError('memory loss weights must be finite and nonnegative')
    parts=dict(complete=complete,compressed=weights['compressed_recall_weight']*compressed,
        physical=weights['physical_recall_weight']*physical,consolidated=weights['consolidated_recall_weight']*consolidated,
        read_policy=weights['memory_policy_weight']*read_policy,
        consolidation_policy=weights['memory_policy_weight']*consolidation_policy,
        reconstruction=weights['reconstruction_weight']*reconstruction)
    if weights['source_codec_weight']>0 and source_codec is None:raise ValueError('source codec target is missing')
    parts['source_codec']=weights['source_codec_weight']*source_codec if source_codec is not None else complete.new_zeros(())
    # Preserve the existing sum/grouping; metrics do not alter the objective.
    loss=(positive_recall_objective(complete,compressed,reconstruction,weights)+parts['physical']+parts['consolidated']
          +weights['memory_policy_weight']*(read_policy+consolidation_policy)+parts['source_codec'])
    return loss,parts


@contextmanager
def evaluation_parameters(model):
    """Do not build model-parameter tapes during temporary evaluation writes."""
    parameters=[] if model.training else [p for p in model.parameters() if p.requires_grad]
    try:
        for p in parameters:p.requires_grad_(False)
        yield
    finally:
        for p in parameters:p.requires_grad_(True)


@contextmanager
def evaluation_write_context(model):
    """Only episodic factors need gradients during an evaluation write.

    Keep dropout and every child in evaluation mode. LFM decoder layers use
    their own `training` flag only to activate native checkpoint recomputation;
    setting that flag locally avoids retaining the full source activation tape.
    All parameter flags and layer flags are restored, including on exceptions.
    """
    if model.training:
        yield 0
        return
    from transformers.models.lfm2.modeling_lfm2 import Lfm2DecoderLayer
    parameters=[p for p in model.parameters() if p.requires_grad]
    layers=[m for m in model.model.language_model.modules()
            if isinstance(m,Lfm2DecoderLayer) and m.gradient_checkpointing]
    flags=[m.training for m in layers]
    try:
        for p in parameters:p.requires_grad_(False)
        for m in layers:m.training=True  # Do not call train(): children stay in eval.
        yield len(layers)
    finally:
        for m,flag in zip(layers,flags):m.training=flag
        for p in parameters:p.requires_grad_(True)


def forward(owner,native_forward,inputs):
    model=owner.model;spec=model.config.native_memory_recall
    if any(k in spec for k in ["contrast_weight","fast_recall_weight","margin"]):
        raise ValueError("Convert the obsolete recall objective configuration before continuation")
    if inputs.get('past_key_values') is not None or inputs.get('inputs_embeds') is not None:
        raise ValueError('Memory recall training requires a fresh native tokenized episode')
    if inputs.get('pixel_values') is not None:
        raise ValueError('This HashHop objective expects text observations; do not silently drop images')
    start=time.monotonic();warm_effective_weights(model)
    ids=inputs['input_ids'];mask=inputs.get('attention_mask',torch.ones_like(ids));labels=inputs['labels']
    episodes=split_episodes(ids,labels,mask,spec);wrong=wrong_memory_rows(episodes)
    padding=getattr(model.config.text_config,'pad_token_id',None)
    source_ids=pad_sequence([e['source'] for e in episodes],batch_first=True,padding_value=0 if padding is None else padding)
    source_mask=torch.arange(source_ids.shape[1],device=ids.device)[None]<torch.tensor(
        [len(e['source']) for e in episodes],device=ids.device)[:,None]
    source_labels=source_ids.masked_fill(~source_mask,-100)
    units=[model.initial_physical_memory(create_graph=model.training) for _ in episodes]
    blocks=BatchedMemoryBlock(model,units,source_ids,observation_mask=source_mask,fresh_rows=[True]*len(units))
    context=model._memory_context.set((blocks,True,source_mask));capture=[];capture_scope=owner.inputs.set(capture)
    try:
        with torch.no_grad(),model._episodic_adapter_bank.use(AdapterBatch(tuple(u.adapters for u in units))):
            source_output=native_forward(input_ids=source_ids,attention_mask=source_mask,logits_to_keep=1,
                use_cache=False,return_dict=True)
    finally:model._memory_context.reset(context);owner.inputs.reset(capture_scope)
    if getattr(source_output,'past_key_values',None) is not None:raise RuntimeError('Source KV cache must not survive the writer')
    observed=[{name:value.detach() for name,value in block.native_observations.items()} for block in blocks.blocks]
    native_inputs=capture[0].detach()
    del source_output,capture,blocks
    graph_written=[];codes=[];sequences=[];auxiliary=[];ffn_deltas=[];observed_count=0;codec_reports=[]
    codec_port=model.config.sequence_codec_port
    for i,(unit,features) in enumerate(zip(units,observed)):
        writer=model.memory.begin(unit.graph)
        for name,value in features.items():
            writer.observe(name,name,value);observed_count+=len(value)
        graph,_=writer.commit(create_graph=model.training)
        # Every observation feature is written. Auxiliary reconstruction may
        # be sampled, but source coverage and posterior storage are complete.
        latent=encode_observations(model.dream_memory.feature_vae,
            {codec_port:native_inputs[i,source_mask[i]]},create_graph=model.training)[codec_port]
        codes.append(latent)
        sequences.append(store_training_sequence(model,native_inputs[i,source_mask[i]],latent))
        graph_written.append(replace(unit,graph=graph))
        training_features={}
        for name,value in dict(features,**{codec_port:native_inputs[i,source_mask[i]]}).items():
            choose=torch.linspace(0,len(value)-1,min(32,len(value)),device=value.device).long()
            training_features[name]=value[choose]
        with torch.autocast(device_type=ids.device.type,enabled=False):
            dream,terms=model.dream_memory.training_loss(graph.fast-unit.graph.fast,training_features)
        codec_reports.append(terms['feature_terms'][codec_port])
        auxiliary.append(dream)
    del observed,native_inputs
    # Deployment order: observe -> graph commit -> source-only FFN learn.
    # Detach the graph only for the first-order inner FFN gradient; the original
    # differentiable graph write is retained for the supervised query.
    learning_units=[replace(u,graph=u.graph.detach()) for u in graph_written]
    learning_blocks=BatchedMemoryBlock(model,learning_units,source_ids,observation_mask=source_mask)
    token=model._memory_context.set((learning_blocks,False,source_mask))
    factors=[v for unit in units for v in unit.adapters.factors.values()]
    try:
        with evaluation_write_context(model) as checkpointed_layers,torch.enable_grad(),model._episodic_adapter_bank.use(AdapterBatch(tuple(u.adapters for u in learning_units))):
            learned=native_forward(input_ids=source_ids,attention_mask=source_mask,labels=source_labels,use_cache=False,return_dict=True)
            source_loss=float(learned.loss.detach())
            gradients=torch.autograd.grad(learned.loss,factors,create_graph=False)
    finally:model._memory_context.reset(token)
    counts=(source_labels[:,1:]!=-100).sum(1).clamp_min(1)
    cursor=0;written=[]
    for i,unit in enumerate(graph_written):
        count=len(unit.adapters.factors)
        row_gradients=[g*counts.sum()/counts[i] for g in gradients[cursor:cursor+count]];cursor+=count
        adapters,_=model._episodic_adapter_bank.apply_gradients(unit.adapters,row_gradients,
            learning_rate=spec.get('write_learning_rate',.001),first_order_graph=model.training)
        written.append(replace(unit,adapters=adapters,sequences=(sequences[i],)))
        ffn_deltas.extend((adapters.factors[n]-unit.adapters.factors[n]).detach() for n in adapters.factors)
    del gradients,learned,learning_blocks

    def read(states,latent_codes,stored_sequences=None):
        embeddings=[];targets=[];lengths=[]
        head=model.dream_memory.feature_vae.heads[codec_port]
        for i,(episode,code) in enumerate(zip(episodes,latent_codes)):
            query=model.get_input_embeddings()(episode['query'])
            if stored_sequences is not None:
                prefix=decode_segment(model,stored_sequences[i]).to(query)
            elif code is None:
                prefix=query.new_empty((0,query.shape[-1]))
            else:
                with torch.autocast(device_type=ids.device.type,enabled=False):
                    prefix=(head.decoder(code.mu)*code.scale+code.mean).to(query.dtype)
            # Complete recall uses the production decoder; compressed recall
            # is explicitly an auxiliary objective without numerical patches.
            bos=getattr(model.config,'bos_token_id',None)
            if bos is None:bos=getattr(model.config.text_config,'bos_token_id',None)
            value,leading=assemble_memory_query(prefix,query,episode['query'],bos)
            embeddings.append(value);lengths.append(len(value))
            targets.append(torch.cat((episode['labels'][:leading],
                torch.full((len(prefix),),-100,device=ids.device,dtype=torch.long),episode['labels'][leading:])))
        values=pad_sequence(embeddings,batch_first=True);target=pad_sequence(targets,batch_first=True,padding_value=-100)
        valid=torch.arange(values.shape[1],device=ids.device)[None]<torch.tensor(lengths,device=ids.device)[:,None]
        read_blocks=BatchedMemoryBlock(model,states,torch.zeros_like(target),observation_mask=valid)
        row,col=(target[:,1:]!=-100).nonzero(as_tuple=True)
        positions=torch.unique(col,sorted=True)
        token=model._memory_context.set((read_blocks,False,valid))
        try:
            with model._episodic_adapter_bank.use(AdapterBatch(tuple(u.adapters for u in states))):
                # All context positions pass through the native stack. Its
                # supported logits_to_keep only skips unsupervised output-head
                # rows, preserving every answer target and its gradients.
                output=native_forward(inputs_embeds=values,attention_mask=valid,
                    logits_to_keep=positions,use_cache=False,return_dict=True)
        finally:model._memory_context.reset(token)
        chosen=output.logits[row,torch.searchsorted(positions,col)]
        output.loss=model.loss_function(logits=chosen,labels=None,shift_labels=target[row,col+1],
            vocab_size=model.config.text_config.vocab_size,num_items_in_batch=inputs.get('num_items_in_batch'))
        first=(target[:,1:]!=-100).long().argmax(-1)
        output.query_prefix_logits=output.logits[torch.arange(len(target),device=target.device),torch.searchsorted(positions,first)]
        return output,target,positions

    correct,query_labels,positions=read(written,codes,sequences);good=per_row_nll(correct,query_labels,positions)
    compressed,code_labels,code_positions=read(written,codes);compressed_loss=compressed.loss
    compressed_nll=per_row_nll(compressed,code_labels,code_positions)
    fast,fast_labels,fast_positions=read(written,[None]*len(written));fast_loss=fast.loss
    fast_nll=per_row_nll(fast,fast_labels,fast_positions)
    read_scores=[]
    for i in range(len(written)):
        score,_,_=rank_reads(model,torch.stack((correct.query_prefix_logits[i],compressed.query_prefix_logits[i],fast.query_prefix_logits[i])))
        read_scores.append(score)
    read_losses=torch.stack((good,compressed_nll,fast_nll),-1)
    selected_actions=torch.stack(read_scores).detach().argmax(-1)
    policy_options=dict(temperature=spec.get('policy_target_temperature',.25),
        score_temperature=spec.get('policy_score_temperature',1.))
    routing_loss=policy_objective(torch.stack(read_scores),read_losses,**policy_options)
    dreamed=replay_candidates(model,written,create_graph=model.training)
    dream_scores=[];dream_retentions=[];dream_source_improved=[]
    for signals,errors,retention in consolidation_observations_batch(model,written,dreamed):
        score,_,_=rank_reads(model,signals,errors=errors)
        dream_scores.append(score);dream_retentions.append(retention)
        dream_source_improved.append(bool(errors[1]<errors[0]))
    consolidated,dream_labels,dream_positions=read(dreamed,codes)
    consolidated_nll=per_row_nll(consolidated,dream_labels,dream_positions)
    # Future queries teach whether source-only replay was useful. They never
    # enter candidate construction, retention checks, or policy inputs.
    consolidation_policy=policy_objective(torch.stack(dream_scores),torch.stack((compressed_nll,consolidated_nll),-1),**policy_options)
    # Controls cannot improve the optimized loss by becoming worse.
    with torch.no_grad():
        empty,empty_labels,empty_positions=read(units,[None]*len(units));empty_nll=per_row_nll(empty,empty_labels,empty_positions)
        available=[j is not None for j in wrong];indices=[i if j is None else j for i,j in enumerate(wrong)]
        incorrect,wrong_labels,wrong_positions=read([written[j] for j in indices],[codes[j] for j in indices],
            [sequences[j] for j in indices])
        wrong_nll=per_row_nll(incorrect,wrong_labels,wrong_positions)
        valid_wrong=torch.tensor(available,device=wrong_nll.device,dtype=torch.bool)
    del empty,incorrect
    with torch.autocast(device_type=ids.device.type,enabled=False):
        dream_loss=torch.stack(auxiliary).mean()+model.dream_memory.weight_replay_loss(ffn_deltas)
    query_loss=correct.loss
    source_codec=None
    if spec.get('source_codec_weight',0.)>0:
        posterior=LatentPortMemory(*(torch.cat([getattr(c,k) for c in codes]) for k in ('mu','logvar','mean','scale')))
        source_codec=model.dream_memory.feature_vae.token_reconstruction_loss(codec_port,posterior,
            torch.cat([e['source'] for e in episodes]),model.get_input_embeddings().weight,
            temperature=spec.get('source_codec_temperature',.02),chunk_size=spec.get('source_codec_chunk_size',64))
    correct.loss,loss_parts=joint_recall_objective(query_loss,compressed_loss,fast_loss,consolidated.loss,
        routing_loss,consolidation_policy,dream_loss,spec,source_codec=source_codec)
    memory_loss=correct.loss-query_loss
    if not torch.isfinite(correct.loss):raise FloatingPointError('Nonfinite memory-dependent training loss')
    owner.batches+=1;owner.pending=[];owner.ffn_replay=ffn_deltas
    owner.last=dict(batch=owner.batches,physical_batch=len(units),padded_tokens=ids.shape[1],vision_tiles=0,
        input_tokens=int(mask.sum()),target_tokens=int((labels[:,1:]!=-100).sum()),
        supervised_loss=float(query_loss.detach()),memory_loss=float(memory_loss.detach()),
        source_loss=source_loss,fast_only_loss=float(fast_loss.detach()),compressed_recall_loss=float(compressed_loss.detach()),
        consolidated_recall_loss=float(consolidated.loss.detach()),read_policy_loss=float(routing_loss.detach()),
        consolidation_policy_loss=float(consolidation_policy.detach()),
        read_policy_actions=selected_actions.cpu().tolist(),
        read_policy_oracle_actions=read_losses.detach().argmin(-1).cpu().tolist(),
        read_policy_regret=float((read_losses.detach().gather(1,selected_actions[:,None]).squeeze(1)-read_losses.detach().amin(-1)).mean()),
        consolidated_query_gain=float((compressed_nll-consolidated_nll).detach().mean()),
        consolidation_improved_rows=int((consolidated_nll.detach()<compressed_nll.detach()).sum()),
        consolidation_model_actions=torch.stack(dream_scores).detach().argmax(-1).cpu().tolist(),
        consolidation_source_improved=dream_source_improved,
        read_policy_statistics=policy_statistics(torch.stack(read_scores),read_losses,**policy_options),
        consolidation_policy_statistics=policy_statistics(torch.stack(dream_scores),torch.stack((compressed_nll,consolidated_nll),-1),**policy_options),
        policy_temperatures=policy_options,
        consolidation_gain_sum=float((compressed_nll-consolidated_nll).detach().sum()),
        consolidation_gain_square_sum=float((compressed_nll-consolidated_nll).detach().square().sum()),
        dream_retention_kl=float(torch.stack(dream_retentions).mean()),
        autonomous_policy_answer_inputs=False,dream_candidates_built=len(dreamed),
        dream_auxiliary_loss=float(dream_loss.detach()),
        loss_contributions={k:float(v.detach()) for k,v in loss_parts.items()},
        source_codec_loss=float(source_codec.detach()) if source_codec is not None else None,
        source_codec_positions=sum(c.count for c in codes) if source_codec is not None else 0,
        compressed_memory_nll=float(compressed_nll.detach().mean()),
        physical_memory_nll=float(fast_nll.detach().mean()),
        consolidated_memory_nll=float(consolidated_nll.detach().mean()),
        codec_metrics={k:float(torch.stack([r[k] for r in codec_reports]).mean()) for k in codec_reports[0]},
        correct_memory_nll=float(good.detach().mean()),empty_memory_nll=float(empty_nll.detach().mean()),wrong_memory_nll=float(wrong_nll[valid_wrong].mean()) if valid_wrong.any() else None,
        negative_control_gradients=False,wrong_control_rows=int(valid_wrong.sum()),
        outer_meta_gradients=model.training,evaluation_checkpointed_layers=checkpointed_layers,
        source_observation_tokens=int(source_mask.sum()),stored_posterior_positions=sum(c.count for c in codes),
        graph_observed_features=observed_count,source_autograd_traversals=1,
        initial_source_forward_passes=2,consolidation_source_forward_passes=5,
        source_forward_passes=7,write_order="observe_graph_then_learn_ffn",
        training_chunks=0,persistent_training_sessions=0,source_kv_reused=False,raw_source_given_to_read=False,
        sequence_residual_patches_used_in_training=True,query_or_answer_given_to_writer=False,
        ffn_gradient_source='source-only batched first-order VJP',ffn_replay_tensors=len(ffn_deltas),
        training_objective='positive complete/code/physical/consolidated recall + query-only read policy + source-only consolidation policy + VAE reconstruction',forward_seconds=time.monotonic()-start)
    if not model.training:
        # Evaluation-only provenance, never passed to a writer or controller.
        # Shared source/query hashes let analysis account for correlated
        # counterfactual questions instead of pretending every token is iid.
        rows=[]
        for i,e in enumerate(episodes):
            end=int((e['labels']!=-100).nonzero()[0])
            digest=lambda x:hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            rows.append(dict(source_hash=digest(e['source']),query_hash=digest(e['query'][:end]),
                complete_nll=float(good[i]),compressed_nll=float(compressed_nll[i]),
                physical_nll=float(fast_nll[i]),consolidated_nll=float(consolidated_nll[i]),
                empty_nll=float(empty_nll[i]),wrong_nll=float(wrong_nll[i]) if available[i] else None,
                selected_action=int(selected_actions[i]),
                selected_nll=float(read_losses[i,selected_actions[i]]),
                consolidation_action=int(dream_scores[i].argmax()),
                source_improved=dream_source_improved[i],retention_kl=float(dream_retentions[i])))
        owner.last['per_example']=rows
    return correct
