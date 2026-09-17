"""Source-only writes followed by fresh-context, memory-dependent query loss.

This objective uses existing graph/FFN weights and VAE heads; it adds no model
parameters. LlamaFactory still owns examples, batches, backward and optimization.
Native multi-turn boundaries separate observations from queries. No SFT windows.
"""
from dataclasses import replace
import time
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from .batched_memory import BatchedMemoryBlock
from .episodic_adapters import AdapterBatch
from .latent_memory import encode_observations
from .sft_lora import warm_effective_weights
from .sequence_memory import assemble_memory_query


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
    chosen=[]
    for i,episode in enumerate(episodes):
        other=next((j for j in range(len(episodes)) if j!=i and
                    not torch.equal(episode['source'],episodes[j]['source'])),None)
        if other is None:raise ValueError('Wrong-memory contrast requires distinct observations in the native batch')
        chosen.append(other)
    return chosen


def forward(owner,native_forward,inputs):
    model=owner.model;spec=model.config.native_memory_recall
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
    units=[model.initial_physical_memory(create_graph=True) for _ in episodes]
    blocks=BatchedMemoryBlock(model,units,source_ids,observation_mask=source_mask,fresh_rows=[True]*len(units))
    context=model._memory_context.set((blocks,True,source_mask));capture=[];capture_scope=owner.inputs.set(capture)
    try:
        with model._episodic_adapter_bank.use(AdapterBatch(tuple(u.adapters for u in units))):
            source_output=native_forward(input_ids=source_ids,attention_mask=source_mask,labels=source_labels,
                use_cache=False,return_dict=True)
    finally:model._memory_context.reset(context);owner.inputs.reset(capture_scope)
    if getattr(source_output,'past_key_values',None) is not None:raise RuntimeError('Source KV cache must not survive the writer')
    source_loss=float(source_output.loss.detach())
    factors=[v for unit in units for v in unit.adapters.factors.values()]
    # One source-only VJP for the entire native batch. Its gradients are first
    # order; no FlashAttention second derivative or answer enters this write.
    gradients=torch.autograd.grad(source_output.loss,factors,create_graph=False)
    source_counts=(source_labels[:,1:]!=-100).sum(1).clamp_min(1)
    observed=[{name:value.detach() for name,value in block.native_observations.items()} for block in blocks.blocks]
    native_inputs=capture[0].detach()
    del source_output,capture,blocks
    cursor=0;written=[];codes=[];auxiliary=[];ffn_deltas=[];observed_count=0
    codec_port=model.config.sequence_codec_port
    for i,(unit,features) in enumerate(zip(units,observed)):
        count=len(unit.adapters.factors)
        row_gradients=[g*source_counts.sum()/source_counts[i] for g in gradients[cursor:cursor+count]];cursor+=count
        adapters,_=model._episodic_adapter_bank.apply_gradients(unit.adapters,row_gradients,
            learning_rate=spec.get('write_learning_rate',.001),first_order_graph=True)
        writer=model.memory.begin(unit.graph)
        for name,value in features.items():
            writer.observe(name,name,value);observed_count+=len(value)
        graph,_=writer.commit(create_graph=True)
        # Every observation feature is written. Auxiliary reconstruction may
        # be sampled, but source coverage and posterior storage are complete.
        latent=encode_observations(model.dream_memory.feature_vae,
            {codec_port:native_inputs[i,source_mask[i]]},create_graph=True)[codec_port]
        codes.append(latent);written.append(replace(unit,graph=graph,adapters=adapters))
        training_features={}
        for name,value in dict(features,**{codec_port:native_inputs[i,source_mask[i]]}).items():
            choose=torch.linspace(0,len(value)-1,min(32,len(value)),device=value.device).long()
            training_features[name]=value[choose]
        with torch.autocast(device_type=ids.device.type,enabled=False):
            dream,_=model.dream_memory.training_loss(graph.fast-unit.graph.fast,training_features)
        auxiliary.append(dream)
        ffn_deltas.extend((adapters.factors[n]-unit.adapters.factors[n]).detach() for n in adapters.factors)
    del gradients,observed,native_inputs

    def read(states,latent_codes):
        embeddings=[];targets=[];lengths=[]
        head=model.dream_memory.feature_vae.heads[codec_port]
        for episode,code in zip(episodes,latent_codes):
            query=model.get_input_embeddings()(episode['query'])
            if code is None:
                prefix=query.new_empty((0,query.shape[-1]))
            else:
                with torch.autocast(device_type=ids.device.type,enabled=False):
                    prefix=(head.decoder(code.mu)*code.scale+code.mean).to(query.dtype)
            # The read gets only stored posterior codes and the query. Exact
            # sequence residual patches/raw source embeddings are not supplied
            # to this compression-learning objective. Inference keeps them.
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
        return output,target,positions

    correct,query_labels,positions=read(written,codes);good=per_row_nll(correct,query_labels,positions)
    # A small auxiliary fast-weight-only read teaches physical storage to carry
    # information even when decoded latent observations cannot supply it.
    fast,fast_labels,_=read(written,[None]*len(written));fast_loss=fast.loss
    empty,empty_labels,empty_positions=read(units,[None]*len(units));empty_nll=per_row_nll(empty,empty_labels,empty_positions)
    incorrect,wrong_labels,wrong_positions=read([written[j] for j in wrong],[codes[j] for j in wrong])
    wrong_nll=per_row_nll(incorrect,wrong_labels,wrong_positions)
    del empty,incorrect,fast
    # Both sides differentiate: a source-independent shortcut that improves
    # correct and empty/wrong contexts equally cannot satisfy this margin.
    contrast=(F.relu(spec.get('margin',.25)+good-empty_nll)+
              F.relu(spec.get('margin',.25)+good-wrong_nll)).mean()
    with torch.autocast(device_type=ids.device.type,enabled=False):
        dream_loss=torch.stack(auxiliary).mean()+model.dream_memory.weight_replay_loss(ffn_deltas)
    query_loss=correct.loss
    memory_loss=(spec.get('fast_recall_weight',.25)*fast_loss+spec.get('contrast_weight',.1)*contrast+
        spec.get('reconstruction_weight',.05)*dream_loss)
    correct.loss=query_loss+memory_loss
    if not torch.isfinite(correct.loss):raise FloatingPointError('Nonfinite memory-dependent training loss')
    owner.batches+=1;owner.pending=[];owner.ffn_replay=ffn_deltas
    owner.last=dict(batch=owner.batches,physical_batch=len(units),padded_tokens=ids.shape[1],vision_tiles=0,
        input_tokens=int(mask.sum()),target_tokens=int((labels[:,1:]!=-100).sum()),
        supervised_loss=float(query_loss.detach()),memory_loss=float(memory_loss.detach()),
        source_loss=source_loss,fast_only_loss=float(fast_loss.detach()),
        correct_memory_nll=float(good.detach().mean()),empty_memory_nll=float(empty_nll.detach().mean()),wrong_memory_nll=float(wrong_nll.detach().mean()),
        source_observation_tokens=int(source_mask.sum()),stored_posterior_positions=sum(c.count for c in codes),
        graph_observed_features=observed_count,source_autograd_traversals=1,
        training_chunks=0,persistent_training_sessions=0,source_kv_reused=False,raw_source_given_to_read=False,
        sequence_residual_patches_used_in_training=False,query_or_answer_given_to_writer=False,
        ffn_gradient_source='source-only batched first-order VJP',ffn_replay_tensors=len(ffn_deltas),
        training_objective='source write then fresh-context dual-memory recall',forward_seconds=time.monotonic()-start)
    return correct
