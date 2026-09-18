"""Free generation on identical queries with different caller-owned memories.

Units are explicitly selected to isolate memory-conditioned recall. This is
not a global index-routing benchmark; the matched HashHop suite measures that.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
from evaluate_memory_generation import write as write_memories,hash_score
from research_common import load,dump,digest,paced,log,question_inputs


def prepare(source,out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    sources={};queries=[];pairs=defaultdict(list)
    for line in (source/'validation.jsonl').read_text().splitlines():
        row=json.loads(line);sid=row['id'].rsplit('-h',1)[0]
        text=row['messages'][0]['content'];question=row['messages'][2]['content'];answer=row['messages'][3]['content']
        if sid in sources and sources[sid]['text']!=text:raise ValueError('Inconsistent source identity')
        sources[sid]=dict(id=sid,kind='hash',text=text)
        qid=hashlib.sha256(question.encode()).hexdigest()
        query=dict(id=row['id'],source_id=sid,question=question,answer=answer,query_pair=qid)
        queries.append(query);pairs[qid].append(query)
    if len(queries)!=128 or len(sources)!=16 or len(pairs)!=64:raise ValueError('Unexpected fixed held-out set')
    for pair in pairs.values():
        if len(pair)!=2 or pair[0]['answer']==pair[1]['answer'] or pair[0]['source_id']==pair[1]['source_id']:
            raise ValueError('Counterfactual pair does not require a different memory answer')
    dump(out/'sources.json',list(sources.values()));dump(out/'questions.json',queries)
    dump(out/'protocol.json',dict(questions=128,counterfactual_pairs=64,units=16,max_new_tokens=256,
        source_sha256=digest(out/'sources.json'),question_sha256=digest(out/'questions.json'),
        scope='Caller supplies the current memory unit; no index-selection claim. Independent cold reader process.',
        same_question_different_memory_different_answer=True,conditions=['provided-memory','memory-disabled'],
        heldout_source_sha256=digest(source/'validation.jsonl')))


def read(args):
    import torch
    protocol=json.loads((args.data/'protocol.json').read_text())
    if digest(args.data/'questions.json')!=protocol['question_sha256']:raise ValueError('Questions changed')
    written=json.loads((args.out/'write-result.json').read_text())
    model,processor=load(args.model);archive=model.open_physical_archive(args.out/'archive')
    if written['checkpoint_id']!=model.config.memory_checkpoint_id:raise ValueError('Checkpoint mismatch')
    versions={n:p._version for n,p in model.named_parameters()}
    pages={p['source_id']:p for p in json.loads((args.out/'pages.json').read_text())}
    questions=json.loads((args.data/'questions.json').read_text());rows=[]
    for condition in protocol['conditions']:
        for query in questions:
            inputs=question_inputs(processor,query['question'])
            if condition=='provided-memory':
                archive.mount_async(pages[query['source_id']]['unit_id']).result()
                output=paced(archive.session.generate,**inputs,max_new_tokens=protocol['max_new_tokens'],do_sample=False)
            else:
                with torch.no_grad():
                    output=paced(model.generate,**inputs,use_memory=False,max_new_tokens=protocol['max_new_tokens'],do_sample=False)
            tokens=output[0,inputs['input_ids'].shape[1]:]
            answer=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
            row=dict(query,condition=condition,expected=query['answer'],answer=answer,
                generated_tokens=len(tokens),hit_generation_limit=len(tokens)==protocol['max_new_tokens'],
                **hash_score(answer,query['answer']))
            rows.append(row);dump(args.out/'answers.json',rows);log('counterfactual_memory_answer',**row)
    scores={}
    for condition in protocol['conditions']:
        subset=[r for r in rows if r['condition']==condition];pairs=defaultdict(list)
        for row in subset:pairs[row['query_pair']].append(row)
        if len(subset)!=128 or len(pairs)!=64 or any(len(p)!=2 for p in pairs.values()):raise ValueError('Incomplete pair coverage')
        scores[condition]=dict(exact=sum(r['exact'] for r in subset),n=len(subset),
            both_memories_correct=sum(all(r['exact'] for r in pair) for pair in pairs.values()),pairs=len(pairs),
            output_limit_hits=sum(r['hit_generation_limit'] for r in subset))
    fixed=all(p._version==versions[n] for n,p in model.named_parameters())
    if not fixed:raise RuntimeError('Evaluation changed the body')
    dump(args.out/'result.json',dict(checkpoint_id=model.config.memory_checkpoint_id,body_fixed=fixed,
        scores=scores,scope=protocol['scope'],protocol_sha256=digest(args.data/'protocol.json'),
        peak_vram_gib=torch.cuda.max_memory_allocated()/2**30))
    archive.close();log('counterfactual_memory_complete',scores=scores)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','write','read'])
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',type=Path);a=p.parse_args()
    gate=os.environ.get('LFM2_START_GATE')
    while gate and not Path(gate).exists():time.sleep(.1)
    if a.mode=='prepare':prepare(a.data,a.out)
    elif a.mode=='write':write_memories(a)
    else:read(a)
