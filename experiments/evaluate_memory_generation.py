"""Cold physical recall, multi-hop hash reasoning and full coherent outputs.

All oracle/source-visible controls are explicitly separate from native routing.
The writer cannot open test questions. Coherence heuristics are diagnostics;
the unabridged generations must also be reviewed before claiming coherence.
"""
import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import re
import time
from unittest.mock import patch

from research_common import load,dump,log,paced,digest,address_inputs


def hash_score(answer,expected):
    actual=answer.strip().split();target=expected.split()
    length=len(target[0])
    valid=all(re.fullmatch(r'[A-Za-z0-9]{'+str(length)+'}',v) for v in actual) and bool(actual)
    prefix=0
    for a,b in zip(actual,target):
        if a!=b:break
        prefix+=1
    return dict(exact=actual==target,valid_hash_format=valid,
        correct_prefix_values=prefix if valid else 0,expected_values=len(target),
        final_value_correct=valid and actual[-1]==target[-1])


def coherence_diagnostics(answer,token_ids,facts):
    grams=[tuple(token_ids[i:i+4]) for i in range(max(0,len(token_ids)-3))]
    counts=Counter(grams)
    numerals={2:'[二兩]',3:'三',4:'四',5:'五'}
    retries=rf"(?:{facts['retries']}|{numerals[facts['retries']]})\s*次"
    attempts=rf"(?:{facts['total_attempts']}|{numerals[facts['total_attempts']]})\s*次"
    return dict(exact_code_preserved=facts['code'] in answer,exact_filename_preserved=facts['file'] in answer,
        timeout_mentioned=bool(re.search(rf"(?<![0-9A-Za-z]){facts['timeout']}\s*秒",answer)),
        retry_count_mentioned=bool(re.search(retries,answer)),total_attempts_mentioned=bool(re.search(attempts,answer)),
        paragraphs=len([p for p in answer.split('\n\n') if p.strip()]),characters=len(answer),
        repeated_4gram_fraction=1-len(counts)/len(grams) if grams else 0.,
        largest_4gram_repeat=max(counts.values(),default=0),manual_semantic_review_required=True)


def inputs_for(processor,messages):
    return processor.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        return_tensors='pt',return_dict=True).to('cuda')


def write(args):
    import torch
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'pages.json').exists():raise FileExistsError('Writer artifacts already exist')
    protocol=json.loads((args.data/'protocol.json').read_text())
    if digest(args.data/'sources.json')!=protocol['source_sha256']:raise ValueError('Evaluation sources changed')
    model,processor=load(args.model);archive=model.open_physical_archive(args.out/'archive')
    versions={n:p._version for n,p in model.named_parameters()};rows=[]
    original=model.generate
    def forbid(*a,**kw):raise RuntimeError('The observation writer must not generate test labels')
    model.generate=forbid
    try:
        for source in json.loads((args.data/'sources.json').read_text()):
            inputs=processor(text=source['text'],return_tensors='pt',truncation=False).to('cuda')
            receipt=paced(archive.session.observe,**inputs,use_cache=False,logits_to_keep=1)
            learning=paced(archive.session.learn,**inputs,labels=inputs['input_ids'].clone(),learning_rate=.001)
            saved=paced(archive.append)
            rows.append(dict(source_id=source['id'],kind=source['kind'],source_tokens=inputs['input_ids'].numel(),
                observed=receipt,write=learning,**saved))
            dump(args.out/'pages.json',rows);log('memory_generation_source_written',source=source['id'],tokens=rows[-1]['source_tokens'])
    finally:model.generate=original
    fixed=all(p._version==versions[n] for n,p in model.named_parameters())
    if not fixed:raise RuntimeError('Frozen body changed during source writes')
    dump(args.out/'write-result.json',dict(body_fixed=fixed,questions_opened=False,generated_labels=0,
        checkpoint_id=model.config.memory_checkpoint_id,units=len(rows),source_sha256=protocol['source_sha256'],
        peak_vram_gib=torch.cuda.max_memory_allocated()/2**30))
    archive.close()


def read(args):
    import torch
    protocol=json.loads((args.data/'protocol.json').read_text())
    if digest(args.data/'questions.json')!=protocol['question_sha256']:raise ValueError('Evaluation questions changed')
    written=json.loads((args.out/'write-result.json').read_text())
    model,processor=load(args.model);archive=model.open_physical_archive(args.out/'archive')
    if written['checkpoint_id']!=model.config.memory_checkpoint_id:raise ValueError('Wrong memory checkpoint')
    versions={n:p._version for n,p in model.named_parameters()}
    pages={p['source_id']:p for p in json.loads((args.out/'pages.json').read_text())}
    queries=json.loads((args.data/'questions.json').read_text());rows=[];routes=[]
    conditions=['native-routed','oracle-units','wrong-unit','empty']
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])

    def component_control(q,inputs,generation,condition):
        """Evaluation-only interventions; the model retains one public reader."""
        namespace=type(model).__module__.rsplit('.',1)[0]
        adapters_module=importlib.import_module(namespace+'.episodic_adapters')
        graph_module=importlib.import_module(namespace+'.multiport_connectome')
        sequence_module=importlib.import_module(namespace+'.sequence_memory')
        ordered=sorted((pages[name]['unit_id'] for name in q['required_sources']))
        units=[archive.session.bank.load_unit(archive._path(name)) for name in ordered]
        selected=dict(inputs)
        with torch.no_grad():
            if condition=='oracle-latent-only':
                selected=sequence_module.prepare_ordered_inputs(model,
                    tuple(s for u in units for s in u.sequences),inputs,max_new_tokens=generation['max_new_tokens'])
                return paced(model.generate,**selected,use_memory=False,**generation),ordered
            if condition=='oracle-codes-and-weights':
                decoded=[]
                for unit in units:
                    for segment in unit.sequences:
                        decoded.append(sequence_module._base(model.dream_memory.feature_vae.heads[segment.codec_port],segment.posterior))
                query=model.get_input_embeddings()(inputs['input_ids'])
                prefix=torch.cat(decoded).to(query)[None]
                if len(prefix[0])+query.shape[1]+generation['max_new_tokens']>model.config.text_config.max_position_embeddings:
                    raise ValueError('Compressed-code diagnostic exceeds native context; no truncation')
                bos=getattr(model.config,'bos_token_id',None)
                if bos is None:bos=getattr(model.config.text_config,'bos_token_id',None)
                combined,_=sequence_module.assemble_memory_query(prefix[0],query[0],inputs['input_ids'][0],bos)
                selected=dict(inputs,inputs_embeds=combined[None],
                    attention_mask=torch.ones((1,prefix.shape[1]+query.shape[1]),device=query.device,dtype=torch.long))
            mixture=adapters_module.AdapterMixture(tuple((u.adapters,1.) for u in units))
            graph=graph_module.MountedState(model.memory.initial_state(),
                tuple(graph_module.MemoryPage(name,u.graph,1.) for name,u in zip(ordered,units)),active_weight=0.)
            with archive.session.bank.use(mixture):
                return paced(model.generate,**selected,memory_state=graph,**generation),ordered

    def run_query(q,condition,sources=None):
        history=[]
        prompts=q['questions'] if q['kind']=='coherence' else [q['question']]
        for turn,prompt in enumerate(prompts):
            history.append(dict(role='user',content=prompt))
            messages=list(history)
            if condition=='source-visible':
                # These source strings are opened only after every memory-only run.
                context='\n\n'.join(sources[name]['text'] for name in q['required_sources'])
                messages=[dict(history[0],content=context+'\n\n'+history[0]['content']),*history[1:]]
            inputs=inputs_for(processor,messages)
            cap=protocol['coherence_max_new_tokens'] if q['kind']=='coherence' else protocol['hash_max_new_tokens']
            generation=dict(max_new_tokens=cap,do_sample=False)
            required={pages[name]['unit_id'] for name in q['required_sources']}
            selected=[];decision=None
            if condition in ['native-routed','oracle-units']:
                query_inputs=address_inputs(processor,prompt)
                keys=paced(archive.encode_query,query_inputs)
                if condition=='native-routed':
                    result=paced(archive.generate,{},encoded_query=keys,generation_inputs=dict(inputs),
                        top_k=q['top_k'],index_options=dict(device='cuda'),**generation)
                else:
                    matches=[dict(unit_id=pages[name]['unit_id'],memory_hash=pages[name]['memory_hash'],score=1.)
                             for name in q['required_sources']]
                    # Only the selector is replaced in this named oracle control.
                    with patch.object(archive,'query',return_value=dict(matches=matches,oracle_control=True)):
                        result=paced(archive.generate,{},encoded_query=keys,generation_inputs=dict(inputs),
                            top_k=len(matches),**generation)
                output=result['tokens'];selected=result['loaded_units'];decision=result.get('memory_decision')
                routes.append(dict(id=q['id'],turn=turn,condition=condition,selected=selected,
                    required=sorted(required),complete_source_coverage=required.issubset(selected)))
            elif condition in ('oracle-physical-only','oracle-latent-only','oracle-codes-and-weights'):
                output,selected=component_control(q,inputs,generation,condition)
            elif condition=='wrong-unit':
                wrong=next(p for p in pages.values() if p['unit_id'] not in required)
                archive.mount_async(wrong['unit_id']).result()
                output=paced(archive.session.generate,**inputs,**generation);selected=[wrong['unit_id']]
                decision=archive.session.last_memory_action
            else:
                with torch.no_grad():output=paced(model.generate,**inputs,use_memory=False,**generation)
            tokens=output[0,inputs['input_ids'].shape[1]:].tolist()
            answer=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
            row=dict(id=q['id'],kind=q['kind'],condition=condition,turn=turn,question=prompt,answer=answer,
                generated_tokens=len(tokens),hit_generation_limit=len(tokens)>=cap,ended_with_eos=bool(tokens and tokens[-1] in eos),
                input_tokens=inputs['input_ids'].numel(),selected_units=selected,memory_decision=decision)
            if q['kind']=='hash':row.update(hops=q['hops'],task=q['task'],hash_length=q['hash_length'],expected=q['answer'],**hash_score(answer,q['answer']))
            else:row.update(facts=q['facts'],diagnostics=coherence_diagnostics(answer,tokens,q['facts']))
            rows.append(row);history.append(dict(role='assistant',content=answer))
            dump(args.out/'answers.json',rows);dump(args.out/'routes.json',routes)
            log('memory_generation_answer',**row)

    for q in queries:
        for condition in conditions:run_query(q,condition)
    diagnostic_queries=[q for q in queries if q['kind']=='coherence' or
        (q['id'].startswith(('hash-0-','hash-1-')) and q['hops'] in (1,4,8)) or q['id'].startswith('cross-4-')]
    for q in diagnostic_queries:
        for condition in ['oracle-physical-only','oracle-latent-only','oracle-codes-and-weights']:
            run_query(q,condition)
    sources={s['id']:s for s in json.loads((args.data/'sources.json').read_text())}
    for q in queries:run_query(q,'source-visible',sources)
    scores={}
    for condition in conditions+['source-visible','oracle-physical-only','oracle-latent-only','oracle-codes-and-weights']:
        subset=[r for r in rows if r['condition']==condition and r['kind']=='hash']
        scores[condition]=dict(n=len(subset),exact=sum(r['exact'] for r in subset),
            by_hash_length={str(n):dict(n=sum(r['hash_length']==n for r in subset),exact=sum(r['exact'] for r in subset if r['hash_length']==n))
                            for n in sorted({r['hash_length'] for r in subset})},
            by_hops={str(h):dict(n=sum(r['hops']==h for r in subset),exact=sum(r['exact'] for r in subset if r['hops']==h))
                     for h in sorted({r['hops'] for r in subset})},
            cross_unit=dict(n=sum(r['id'].startswith('cross') for r in subset),
                exact=sum(r['exact'] for r in subset if r['id'].startswith('cross'))))
    fixed=all(p._version==versions[n] for n,p in model.named_parameters())
    if not fixed:raise RuntimeError('Frozen body changed during recall')
    by_id={(r['id'],r['condition']):r for r in rows if r['kind']=='hash'}
    failure_counts=dict(native_decoding=sum(not by_id[q['id'],'source-visible']['exact'] for q in queries if q['kind']=='hash'),
        memory_read=sum(by_id[q['id'],'source-visible']['exact'] and not by_id[q['id'],'oracle-units']['exact'] for q in queries if q['kind']=='hash'),
        routing_or_composition=sum(by_id[q['id'],'oracle-units']['exact'] and not by_id[q['id'],'native-routed']['exact'] for q in queries if q['kind']=='hash'))
    result=dict(checkpoint_id=model.config.memory_checkpoint_id,body_fixed=fixed,hash_scores=scores,
        diagnostic_failure_counts=failure_counts,coherence_outputs=sum(r['kind']=='coherence' for r in rows),
        generation_limit_hits=sum(r['hit_generation_limit'] for r in rows),
        coherence_claim='Pending manual review of full multi-turn outputs; heuristics do not establish coherence',
        component_ablation_scope='hash tables 0/1 at hops 1,4,8; cross-unit hop 4; both coherence dialogues; correct-unit oracle selection',
        native_routes_with_all_required_sources=sum(r['complete_source_coverage'] for r in routes if r['condition']=='native-routed'),
        peak_vram_gib=torch.cuda.max_memory_allocated()/2**30,training_inputs_never_opened=True,
        protocol_sha256=digest(args.data/'protocol.json'))
    dump(args.out/'result.json',result);archive.close();log('memory_generation_complete',**result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['write','read'])
    p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    gate=os.environ.get('LFM2_START_GATE')
    while gate and not Path(gate).exists():time.sleep(.1)
    globals()[a.mode](a)
