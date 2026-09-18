"""Whole-example joint memory loss; native LlamaFactory owns training.

There is no SFT window/session scheduler. FFN updates reuse the native
supervised backward; the model returns native logits and CE plus memory loss.
"""
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
import json,time,os
import torch
from torch.nn import functional as F
from transformers import TrainerCallback
from .batched_memory import BatchedMemoryBlock
from .episodic_adapters import AdapterBatch
from .sft_lora import save_custom_lora,shared_effective_weights,warm_effective_weights


def sampled_pairs(features,count,temporal):
    x=features.detach().float().reshape(-1,features.shape[-1])
    available=len(x)-1 if temporal else len(x)
    if available<1:return None
    indices=torch.linspace(0,available-1,min(count,available),device=x.device).long()
    return x[indices],x[indices+1] if temporal else x[indices]


def capture_supervised_gradients(units,labels,normalizer,gradient_scale=1.):
    """Observe native backward without changing its gradients or objective."""
    pending=[]
    for i,unit in enumerate(units):
        recorded={};count=(labels[i,1:]!=-100).sum().clamp_min(1)
        scale=normalizer.detach().float()/count if torch.is_tensor(normalizer) else normalizer/count
        scale=scale*gradient_scale
        for name,value in unit.adapters.factors.items():
            def capture(gradient,name=name,target=recorded,scale=scale):
                target[name]=gradient.detach()*scale
                return gradient
            value.register_hook(capture)
        pending.append(dict(adapters=unit.adapters.detach(),gradients=recorded))
    return pending


class NativeSFTMemory:
    def __init__(self,model):
        self.model=model;self.batches=0;self.last={};self.pending=[];self.ffn_replay=[];self.gradient_scale=1.;self.eval_reports=[]
        self.inputs=ContextVar(f'joint_memory_inputs_{id(model)}',default=None)
        def capture(module,args,kwargs):
            target=self.inputs.get()
            if target is not None:target.append(kwargs['inputs_embeds'].detach())
        model.model.language_model.register_forward_pre_hook(capture,with_kwargs=True)

    def forward(self,native_forward,inputs):
        with shared_effective_weights():
            if getattr(self.model.config,'native_memory_recall',None):
                from .memory_recall_training import forward,evaluation_parameters
                if not self.model.training and not self.eval_reports and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                # Evaluation needs only the source-only inner write gradient.
                # Query scoring, codecs and meta-gradients must remain disabled.
                with evaluation_parameters(self.model),torch.set_grad_enabled(self.model.training):
                    output=forward(self,native_forward,inputs)
                if not self.model.training:
                    if torch.cuda.is_available():self.last['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
                    self.eval_reports.append(dict(self.last))
                return output
            return self._forward(native_forward,inputs)

    def _forward(self,native_forward,inputs):
        if self.pending:self.finish_backward()
        model=self.model;start=time.monotonic();ids=inputs['input_ids'];attention=inputs.get('attention_mask')
        events=[torch.cuda.Event(enable_timing=True) for _ in range(5)] if ids.is_cuda else None
        if events:events[0].record()
        warm_effective_weights(model)
        if events:events[1].record()
        if attention is None:attention=torch.ones_like(ids)
        units=[model.initial_physical_memory(create_graph=True) for _ in ids]
        blocks=BatchedMemoryBlock(model,units,ids,inputs.get('spatial_shapes'),attention.bool(),fresh_rows=[True]*len(units))
        token=model._memory_context.set((blocks,True,attention));captured=[];scope=self.inputs.set(captured)
        labels=inputs['labels'];denominator=inputs.get('num_items_in_batch')
        if denominator is None:denominator=(labels[:,1:]!=-100).sum()
        denominator=denominator.clamp_min(1) if torch.is_tensor(denominator) else max(1,denominator)
        self.pending=capture_supervised_gradients(units,labels,denominator,self.gradient_scale)
        try:
            with model._episodic_adapter_bank.use(AdapterBatch(tuple(u.adapters for u in units))):
                output=native_forward(**dict(inputs,use_cache=False,return_dict=True,num_items_in_batch=denominator))
        finally:model._memory_context.reset(token);self.inputs.reset(scope)
        if events:events[2].record()
        forward_seconds=time.monotonic()-start;tick=time.monotonic()
        # FFN update gradients are collected by the native Trainer backward.
        if events:events[3].record()
        gradient_seconds=time.monotonic()-tick;tick=time.monotonic()
        count=getattr(model.config,'native_sft_aux_features',32)
        if type(count) is not int or count<2:raise ValueError('at least two auxiliary observations required')
        losses=[];read_before=[];read_after=[];feature_count=0;codec_reports=[]
        for row,(unit,block) in enumerate(zip(units,blocks.blocks)):
            writer=model.memory.begin(unit.graph);pairs={};vae_features={}
            for name,features in block.native_observations.items():
                pair=sampled_pairs(features,count,name!=model.config.vision_port)
                if pair is None:continue
                key,value=pair;pairs[name]=pair;vae_features[name]=value
                writer.observe(name,name,torch.stack((key,value),dim=1) if name!=model.config.vision_port else key)
                feature_count+=len(key)
            written_graph,_=writer.commit(create_graph=True)
            written=replace(unit,graph=written_graph);recalls=[]
            for name,(key,value) in pairs.items():
                target=F.layer_norm(value,(value.shape[-1],));gate=model.memory.ports[name].residual_gate.sigmoid()
                prediction=model.memory.read(name,key,written.graph.fast)*gate
                recalls.append(F.mse_loss(prediction,target))
                with torch.no_grad():
                    prior=model.memory.read(name,key,unit.graph.fast.detach())*gate.detach()
                    read_before.append(float(F.mse_loss(prior,target)));read_after.append(float(recalls[-1].detach()))
            codec_port=getattr(model.config,'sequence_codec_port',None)
            if codec_port is not None:
                source=captured[0][row][attention[row].bool()]
                pair=sampled_pairs(source,count,False)
                if pair is not None:vae_features[codec_port]=pair[0]
            with torch.autocast(device_type=ids.device.type,enabled=False):
                dream,terms=model.dream_memory.training_loss(written_graph.fast-unit.graph.fast,vae_features)
            if codec_port in terms['feature_terms']:codec_reports.append(terms['feature_terms'][codec_port])
            if self.ffn_replay:
                dream=dream-.5*(terms['distortion']+model.dream_memory.spec['kl_weight']*terms['kl'])
            losses.append(torch.stack(recalls).mean()+dream)
        with torch.autocast(device_type=ids.device.type,enabled=False):
            replay_loss=model.dream_memory.weight_replay_loss(self.ffn_replay)
        auxiliary=getattr(model.config,'native_sft_memory_loss_weight',.05)*(torch.stack(losses).mean()+.5*replay_loss)
        supervised=output.loss;output.loss=supervised+auxiliary
        if not torch.isfinite(output.loss):raise FloatingPointError('nonfinite joint SFT loss')
        if events:events[4].record();events[4].synchronize()
        self.batches+=1
        self.last=dict(batch=self.batches,physical_batch=len(ids),padded_tokens=ids.shape[1],
            vision_tiles=0 if inputs.get('spatial_shapes') is None else len(inputs['spatial_shapes']),
            input_tokens=int(attention.sum()),target_tokens=int((labels[:,1:]!=-100).sum()),
            supervised_loss=float(supervised.detach()),memory_loss=float(auxiliary.detach()),
            native_forward_seconds=forward_seconds,source_gradient_seconds=gradient_seconds,
            memory_aux_seconds=time.monotonic()-tick,auxiliary_feature_pairs=feature_count,
            source_autograd_traversals=0,training_chunks=0,persistent_training_sessions=0,
            ffn_gradient_source='native supervised backward',ffn_replay_tensors=len(self.ffn_replay),
            ffn_replay_loss=float(replay_loss.detach()),
            codec_metrics={k:float(torch.stack([r[k] for r in codec_reports]).mean()) for k in codec_reports[0]} if codec_reports else {},
            memory_read_before=sum(read_before)/len(read_before),memory_read_after=sum(read_after)/len(read_after),
            native_forward=f'{native_forward.__module__}.{native_forward.__qualname__}')
        if events:
            self.last.update(weight_materialization_seconds=events[0].elapsed_time(events[1])/1000,
                native_forward_seconds=events[1].elapsed_time(events[2])/1000,
                source_gradient_seconds=events[2].elapsed_time(events[3])/1000,
                memory_aux_seconds=events[3].elapsed_time(events[4])/1000,timing_basis='CUDA events')
        return output

    @torch.no_grad()
    def finish_backward(self):
        if not self.pending:return
        replay=[]
        for item in self.pending:
            state=item['adapters'];gradients=item['gradients']
            if set(gradients)!=set(state.factors):
                raise RuntimeError('native backward did not supply every physical FFN gradient')
            updated,_=self.model._episodic_adapter_bank.apply_gradients(state,
                tuple(gradients[n] for n in state.factors),first_order_graph=False)
            replay.extend((updated.factors[n]-state.factors[n]).detach() for n in state.factors)
        self.ffn_replay=replay;self.pending=[]

    def save_replay(self,directory):
        from safetensors.torch import save_file
        if self.pending:raise RuntimeError('checkpoint before the native backward completed')
        if self.ffn_replay:
            save_file({str(i):v.detach().cpu().contiguous() for i,v in enumerate(self.ffn_replay)},
                str(Path(directory)/'dream_replay.safetensors'))

    def load_replay(self,directory):
        from safetensors.torch import load_file
        path=Path(directory)/'dream_replay.safetensors'
        if path.exists():
            values=load_file(str(path),device=str(self.model.device))
            self.ffn_replay=[values[str(i)] for i in range(len(values))]


class NativeMemoryCheckpoint(TrainerCallback):
    def __init__(self,model,resume=None):self.model=model;self.resume=resume;self.pending_rest=0.;self.gradient_logged=False
    def on_train_begin(self,args,state,control,**kwargs):
        self.model._native_sft_memory.gradient_scale=args.gradient_accumulation_steps
        if self.resume:self.model._native_sft_memory.load_replay(self.resume)
    def on_substep_end(self,args,state,control,**kwargs):
        self.model._native_sft_memory.finish_backward()
    def on_step_begin(self,args,state,control,**kwargs):
        if self.pending_rest:time.sleep(self.pending_rest);self.pending_rest=0.
        self.tick=time.monotonic()
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    def on_pre_optimizer_step(self,args,state,control,**kwargs):
        self.model._native_sft_memory.finish_backward()
        if self.gradient_logged:return
        groups={}
        for name,p in self.model.named_parameters():
            if not p.requires_grad:continue
            group=('vision' if '.vision_tower.' in name else 'projector' if '.multi_modal_projector.' in name
                else 'language' if '.language_model.' in name else 'titans' if name.startswith('memory.')
                else 'physical_ffn' if name.startswith('physical_memory.') else 'dream')
            value=groups.setdefault(group,dict(parameters=0,with_gradient=0,squares=[]))
            value['parameters']+=p.numel()
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():raise FloatingPointError('nonfinite LoRA gradient: '+name)
                value['with_gradient']+=p.numel();value['squares'].append(p.grad.detach().float().square().sum())
        for value in groups.values():
            terms=value.pop('squares');value['gradient_norm']=float(torch.stack(terms).sum().sqrt()) if terms else 0.
        (Path(args.output_dir)/f'gradient-audit-step-{state.global_step+1}.json').write_text(json.dumps(groups,indent=2)+'\n')
        components={}
        for prefix in ('dream_memory.controller.','dream_memory.feature_vae.','dream_memory.weight_vae.'):
            selected=[p for n,p in self.model.named_parameters() if n.startswith(prefix) and p.requires_grad]
            squares=[p.grad.detach().float().square().sum() for p in selected if p.grad is not None]
            components[prefix]=dict(parameters=sum(p.numel() for p in selected),
                with_gradient=sum(p.numel() for p in selected if p.grad is not None),
                gradient_norm=float(torch.stack(squares).sum().sqrt()) if squares else 0.)
        (Path(args.output_dir)/f'component-gradient-audit-step-{state.global_step+1}.json').write_text(json.dumps(components,indent=2)+'\n')
        self.gradient_logged=True
    def on_step_end(self,args,state,control,**kwargs):
        elapsed=time.monotonic()-self.tick
        value=dict(status='running',step=state.global_step,total_steps=state.max_steps,
            step_compute_seconds=elapsed,**self.model._native_sft_memory.last)
        if torch.cuda.is_available():value.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            allocated_gib=torch.cuda.memory_allocated()/2**30,reserved_gib=torch.cuda.memory_reserved()/2**30)
        path=Path(args.output_dir)/'progress.json';temporary=path.with_suffix('.pending')
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)
        print(json.dumps(value,ensure_ascii=False),flush=True)
        self.pending_rest=elapsed*getattr(self.model.config,'native_sft_rest_ratio',1.)
        stop=os.environ.get('LFM2_SFT_PROBE_STOP')
        if stop is not None and state.global_step>=int(stop):
            control.should_training_stop=True;control.should_save=True
    def on_save(self,args,state,control,**kwargs):
        directory=Path(args.output_dir)/f'checkpoint-{state.global_step}'
        save_custom_lora(self.model,directory);self.model._native_sft_memory.save_replay(directory)
    def on_prediction_step(self,args,state,control,**kwargs):
        if getattr(self.model.config,'native_memory_recall',None):
            seconds=self.model._native_sft_memory.last.get('forward_seconds',0.)
            time.sleep(seconds*getattr(self.model.config,'native_sft_rest_ratio',1.))
    def on_evaluate(self,args,state,control,metrics=None,**kwargs):
        reports=self.model._native_sft_memory.eval_reports
        if not reports:return
        count=sum(r['physical_batch'] for r in reports)
        keys=['supervised_loss','compressed_recall_loss','fast_only_loss','consolidated_recall_loss',
              'read_policy_loss','consolidation_policy_loss','dream_retention_kl','memory_loss',
              'read_policy_regret','consolidated_query_gain',
              'compressed_memory_nll','physical_memory_nll','consolidated_memory_nll',
              'source_codec_loss',
              'dream_auxiliary_loss','correct_memory_nll','empty_memory_nll','wrong_memory_nll']
        values={k:sum(r[k]*r['physical_batch'] for r in reports)/count
                for k in keys if all(r.get(k) is not None for r in reports)}
        value=dict(step=state.global_step,examples=count,**values,
            policy_temperatures=reports[0].get('policy_temperatures'),
            read_action_counts={str(i):sum(r.get('read_policy_actions',[]).count(i) for r in reports) for i in range(3)},
            consolidation_action_counts={str(i):sum(r.get('consolidation_model_actions',[]).count(i) for r in reports) for i in range(2)},
            codec_metrics={k:sum(r['codec_metrics'][k]*r['physical_batch'] for r in reports)/count
                for k in reports[0].get('codec_metrics',{})},
            query_or_answer_given_to_writer=any(r['query_or_answer_given_to_writer'] for r in reports),
            outer_meta_gradients=any(r.get('outer_meta_gradients',False) for r in reports),
            evaluation_checkpointed_layers=max(r.get('evaluation_checkpointed_layers',0) for r in reports),
            peak_allocated_gib=max(r.get('peak_allocated_gib',0.) for r in reports),
            native_evaluation_loss=(metrics or {}).get('eval_loss'))
        for name in ('read_policy_statistics','consolidation_policy_statistics'):
            if all(name in r for r in reports):
                value[name]={k:sum(r[name][k]*r['physical_batch'] for r in reports)/count for k in reports[0][name]}
        if all('per_example' in r for r in reports):value['per_example']=[x for r in reports for x in r['per_example']]
        if all('consolidation_gain_sum' in r for r in reports):
            total=sum(r['consolidation_gain_sum'] for r in reports)
            squares=sum(r['consolidation_gain_square_sum'] for r in reports)
            mean=total/count;variance=max(0.,(squares-count*mean*mean)/max(1,count-1))
            value['consolidation_gain_normal_95_lower']=mean-1.96*(variance/count)**.5
            value['consolidation_gain_ci_scope']='approximate per-question interval; paired source correlation not modeled'
        (Path(args.output_dir)/f'memory-eval-step-{state.global_step}.json').write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps(dict(stage='heldout_memory_evaluation',**value)),flush=True)
        reports.clear()
    def on_train_end(self,args,state,control,**kwargs):
        save_custom_lora(self.model,args.output_dir);self.model._native_sft_memory.save_replay(args.output_dir)
