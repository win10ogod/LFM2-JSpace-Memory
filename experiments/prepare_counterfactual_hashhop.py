"""Same-query counterfactual memories with independent graph depth and hop count."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
from itertools import product
from hashhop_curriculum import random_symbol,graph,question,source_text,validate_case,training_token_lengths
from research_common import dump,digest


def examples(seed,groups,forbidden=()):
    rng=random.Random(seed);used=set(forbidden);initial=set(used);rows=[];graphs=[]
    settings=list(product([8,10,12,16],[16,32,64,96],[8,16]))
    for _ in range(100):
        rng.shuffle(settings)
        subset=settings[:min(groups,len(settings))]
        if groups<8 or all(len({s[i] for s in subset})==n for i,n in [(0,4),(1,4),(2,2)]):break
    else:raise RuntimeError('Could not balance the small validation settings')
    for group in range(groups):
        depth,pairs,length=settings[group%len(settings)]
        start_at=0 if depth==8 or group%4==0 else rng.randint(1,depth-8)
        anchor=random_symbol(rng,used,length)
        for view in range(2):
            edges,chain=graph(rng,hops=depth,pairs=pairs,used=used,hash_length=length)
            replaced=chain[start_at];chain[start_at]=anchor
            edges=[(anchor if a==replaced else a,anchor if b==replaced else b) for a,b in edges]
            rng.shuffle(edges);graph_id=f'g{group}-v{view}'
            graphs.append(dict(id=graph_id,pair_id=f'g{group}',edges=edges,chain=chain,start_index=start_at))
            for hops in [1,2,4,8]:
                for task in ['trace','jump']:
                    values=chain[start_at+1:start_at+hops+1]
                    answer=' '.join(values) if task=='trace' else values[-1]
                    validate_case(edges,anchor,hops,task,answer)
                    rows.append(dict(id=f'{graph_id}-h{hops}-{task}',graph_id=graph_id,pair_id=f'g{group}',
                        hops=hops,task=task,start=anchor,start_index=start_at,hash_length=length,
                        messages=[dict(role='user',content=source_text(edges)),dict(role='assistant',content='Observed.'),
                                  dict(role='user',content=question(anchor,hops,task,True)),dict(role='assistant',content=answer)],images=[]))
    # Keep counterfactual views adjacent in validation's native sequential
    # batches. Training still uses the native sampler and no special batching.
    rows.sort(key=lambda r:(int(r['pair_id'][1:]),r['hops'],r['task'],r['graph_id']))
    return rows,graphs,used-initial


def audit(rows,graphs):
    lookup={g['id']:g for g in graphs};questions=defaultdict(list);combinations=defaultdict(set)
    terminals=correct=interior=0
    for row in rows:
        graph=lookup[row['graph_id']];table=dict(graph['edges'])
        validate_case(graph['edges'],row['start'],row['hops'],row['task'],row['messages'][-1]['content'])
        path=[];value=row['start']
        while value in table:value=table[value];path.append(value)
        shortcut=' '.join(path) if row['task']=='trace' else path[-1]
        correct+=shortcut==row['messages'][-1]['content'];terminals+=len(path)==row['hops']
        interior+=row['start'] in set(table.values())
        questions[row['messages'][2]['content']].append(row)
        combinations[row['graph_id']].add((row['start'],row['hops']))
    paired=sum(len(q)==2 and len({r['messages'][-1]['content'] for r in q})==2 for q in questions.values())
    if paired!=len(rows)//2 or correct>=len(rows)//4 or not interior:
        raise RuntimeError('Counterfactual or fixed-hop coverage audit failed')
    if not all(len(q)==4 for q in combinations.values()):raise RuntimeError('Each graph needs four hop counts')
    return dict(records=len(rows),graphs=len(graphs),counterfactual_query_pairs=paired,
        interior_start_records=interior,terminal_target_records=terminals,
        ignore_hop_count_solver_correct=correct,all_targets_verified=True,
        hop_counts=sorted({r['hops'] for r in rows}),hash_lengths=sorted({r['hash_length'] for r in rows}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--prior-data',type=Path,required=True);a=p.parse_args()
    if a.directory.exists():raise FileExistsError(a.directory)
    forbidden=set()
    for name in ['training-graphs.json','evaluation-graphs.json']:
        for g in json.loads((a.prior_data/name).read_text()):
            for edge in g['edges']:forbidden.update(edge)
    train,train_graphs,train_symbols=examples(917203,64,forbidden)
    valid,valid_graphs,valid_symbols=examples(617307,8,forbidden|train_symbols)
    if train_symbols&valid_symbols:raise RuntimeError('Train/validation symbol overlap')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,trust_remote_code=True)
    lengths=training_token_lengths(tok,train);vl=training_token_lengths(tok,valid)
    a.directory.mkdir(parents=True)
    for filename,rows in [('train.jsonl',train),('validation.jsonl',valid)]:
        with (a.directory/filename).open('w') as f:
            for row in rows:f.write(json.dumps({k:row[k] for k in ['id','messages','images']},ensure_ascii=False)+'\n')
    for name,value in [('training-graphs.json',train_graphs),('validation-graphs.json',valid_graphs)]:dump(a.directory/name,value)
    template=dict(formatting='sharegpt',columns=dict(messages='messages',images='images'),
        tags=dict(role_tag='role',content_tag='content',user_tag='user',assistant_tag='assistant'))
    dump(a.directory/'dataset_info.json',dict(hashhop_curriculum=dict(template,file_name='train.jsonl'),
         hashhop_validation=dict(template,file_name='validation.jsonl')))
    report=dict(status='passed',training=audit(train,train_graphs),validation=audit(valid,valid_graphs),
        train_validation_symbols_disjoint=True,prior_training_and_evaluation_symbols_excluded=True,
        train_tokens=sum(lengths),validation_tokens=sum(vl),max_input_tokens=max(lengths+vl),
        sft_chunking=False,source_only_writer=True,prior_evaluation_unchanged=True,
        files_sha256={name:digest(a.directory/name) for name in ['train.jsonl','validation.jsonl']})
    dump(a.directory/'preparation.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
