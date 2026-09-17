"""Small, auditable HashHop datasets with disjoint train/evaluation graphs.

All requested edges are present and shuffled. The answer solver is used only
while constructing/scoring datasets; model evaluation never calls it to answer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import string
import math


def random_symbol(rng, used, length=8):
    while True:
        value=''.join(rng.choices(string.ascii_letters,k=length))
        if value not in used:
            used.add(value)
            return value


def graph(rng, *, hops, pairs, used, hash_length=8):
    if hops<1 or pairs<hops:raise ValueError('Every requested edge must fit the source')
    # Every chain has the same depth. The queried chain is not a unique long
    # chain surrounded by isolated pairs that could reveal which data to keep.
    chains=[[random_symbol(rng,used,hash_length) for _ in range(hops+1)]
            for _ in range(math.ceil(pairs/hops))]
    edges=[edge for chain in chains for edge in zip(chain,chain[1:])]
    rng.shuffle(edges)
    return edges,rng.choice(chains)


def question(start,hops,task,prior_observation=False):
    where='the previously observed hash pairs' if prior_observation else 'the hash pairs above'
    instruction=('Return every visited value in order, separated by single spaces. Do not repeat the starting key.'
                 if task=='trace' else 'Return only the final value, with no intermediate values.')
    return f'Using {where}, start at {start} and follow exactly {hops} arrows. {instruction} Preserve every character and its case.'


def validate_case(edges,start,hops,task,answer):
    table=dict(edges)
    if len(table)!=len(edges):raise ValueError('Ambiguous source keys')
    value=start;visited=[]
    for _ in range(hops):
        if value not in table:raise ValueError('Missing required edge')
        value=table[value];visited.append(value)
    expected=' '.join(visited) if task=='trace' else value
    if answer!=expected:raise ValueError('Incorrect HashHop target')
    return visited


def source_text(edges):
    return '\n'.join(f'{a} → {b}' for a,b in edges)


def training_examples(seed=913741):
    rng=random.Random(seed);used=set();rows=[];graphs=[]
    for hops in [1,2,4,8]:
        count=64 if hops==1 else 32
        for index in range(count):
            pairs=[16,32,64,96][index%4]
            length=[8,16][(index//4)%2]
            edges,chain=graph(rng,hops=hops,pairs=pairs,used=used,hash_length=length)
            tasks=['jump'] if hops==1 else ['trace','jump']
            graph_id=f'h{hops}-g{index}'
            graphs.append(dict(id=graph_id,edges=edges,chain=chain,hash_length=length))
            for task in tasks:
                answer=' '.join(chain[1:]) if task=='trace' else chain[-1]
                validate_case(edges,chain[0],hops,task,answer)
                rows.append(dict(id=graph_id+'-'+task,messages=[dict(role='user',content=source_text(edges)),
                    dict(role='assistant',content='Observed.'),
                    dict(role='user',content=question(chain[0],hops,task,True)),
                    dict(role='assistant',content=answer)],images=[],hops=hops,task=task,graph_id=graph_id,hash_length=length))
    assert len(rows)==256
    return rows,graphs,used


def evaluation_cases(seed=782349,forbidden=()):
    rng=random.Random(seed);used=set(forbidden);initial=set(used);sources=[];queries=[];graphs=[]
    for index in range(4):
        pairs=[24,48,80,96][index]
        length=[8,16,8,16][index]
        edges,chain=graph(rng,hops=10,pairs=pairs,used=used,hash_length=length)
        source_id=f'hash-{index}'
        sources.append(dict(id=source_id,kind='hash',text=source_text(edges)))
        graphs.append(dict(id=source_id,edges=edges,chain=chain,hash_length=length))
        for hops in [1,2,4,8,10]:
            for task in (['jump'] if hops==1 else ['trace','jump']):
                answer=' '.join(chain[1:hops+1]) if task=='trace' else chain[hops]
                validate_case(edges,chain[0],hops,task,answer)
                queries.append(dict(id=f'{source_id}-{hops}-{task}',kind='hash',hops=hops,task=task,
                    question=question(chain[0],hops,task,True),answer=answer,hash_length=length,
                    required_sources=[source_id],top_k=1,source_pair_positions=[edges.index((a,b)) for a,b in zip(chain[:hops],chain[1:hops+1])]))
    # A distinct graph spans four physical units. Each chain hop changes unit.
    edges,chain=graph(rng,hops=8,pairs=64,used=used,hash_length=16);pages=[[] for _ in range(4)]
    # Every chain, including distractors, uses the same per-hop page pattern.
    table=dict(edges);roots=[k for k in table if k not in set(table.values())]
    owner={}
    for start in roots:
        value=start
        for hop in range(8):
            edge=(value,table[value]);pages[hop%4].append(edge);owner[value]=hop%4;value=table[value]
    for i,page in enumerate(pages):
        rng.shuffle(page);sources.append(dict(id=f'cross-{i}',kind='hash',text=source_text(page)))
    graphs.append(dict(id='cross',edges=edges,chain=chain,hash_length=16))
    for hops in [2,4,8]:
        for task in ['trace','jump']:
            answer=' '.join(chain[1:hops+1]) if task=='trace' else chain[hops]
            validate_case(edges,chain[0],hops,task,answer)
            queries.append(dict(id=f'cross-{hops}-{task}',kind='hash',hops=hops,task=task,
                question=question(chain[0],hops,task,True),answer=answer,hash_length=16,
                required_sources=[f'cross-{i}' for i in sorted({owner[k] for k in chain[:hops]})],top_k=4))
    for i,name in enumerate(['映杉服務','靜澄服務']):
        code=random_symbol(rng,used);file=random_symbol(rng,used)+'.bin';timeout=[17,23][i];retry=[2,4][i]
        text=(f'{name}本次發布的操作手冊。部署識別碼為 {code}。備份檔名為 {file}。'
              f'每次請求逾時門檻是 {timeout} 秒，首次嘗試失敗後最多重試 {retry} 次。'
              '重新啟動之前，必須先備份，再檢查備份校驗值，兩步成功後才能重新啟動。'
              '遇到校驗失敗時保留原服務，禁止刪除原備份，交由值班人員處理。')
        source_id=f'coherence-{i}'
        sources.append(dict(id=source_id,kind='coherence',text=text))
        queries.append(dict(id=source_id,kind='coherence',required_sources=[source_id],top_k=1,
            questions=[f'請依照你記住的{name}手冊，以三段連貫文字寫約三百字的交接說明。包含部署識別碼、備份檔名、逾時秒數、最多重試次數、重啟順序及校驗失敗處理。不要列點。',
                f'接續上面的說明，解釋{name}在第一次請求失敗後應如何處理，以及最多總共嘗試幾次；請保留部署識別碼與備份檔名，寫成連貫段落。'],
            facts=dict(code=code,file=file,timeout=timeout,retries=retry,total_attempts=retry+1),
            manual_review_required=True))
    return sources,queries,graphs,used-initial


def training_token_lengths(tokenizer,rows):
    lengths=[]
    for row in rows:
        encoded=tokenizer.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=False,return_dict=True)
        ids=encoded['input_ids']
        if ids and isinstance(ids[0],list):
            if len(ids)!=1:raise ValueError('Expected exactly one training conversation')
            ids=ids[0]
        lengths.append(len(ids))
        if len(ids)>4096:raise RuntimeError('Generated SFT record exceeds native cutoff; no truncation permitted here')
    return lengths


def prepare(directory,tokenizer_path):
    from transformers import AutoTokenizer
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir(parents=True)
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True,trust_remote_code=True)
    train,train_graphs,train_symbols=training_examples()
    sources,queries,eval_graphs,eval_symbols=evaluation_cases(forbidden=train_symbols)
    if train_symbols&eval_symbols:raise RuntimeError('Train/evaluation symbol leakage')
    lengths=training_token_lengths(tokenizer,train)
    with (directory/'train.jsonl').open('x') as f:
        for row in train:f.write(json.dumps({k:v for k,v in row.items() if k not in ['hops','task','graph_id','hash_length']},ensure_ascii=False)+'\n')
    for name,data in [('training-graphs.json',train_graphs),('sources.json',sources),('questions.json',queries),('evaluation-graphs.json',eval_graphs)]:
        (directory/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    info={'hashhop_curriculum':dict(file_name='train.jsonl',formatting='sharegpt',
        columns=dict(messages='messages',images='images'),tags=dict(role_tag='role',content_tag='content',user_tag='user',assistant_tag='assistant'))}
    (directory/'dataset_info.json').write_text(json.dumps(info,indent=2)+'\n')
    protocol=dict(training_examples=len(train),training_graphs=len(train_graphs),training_hops=[1,2,4,8],
        evaluation_hash_questions=sum(q['kind']=='hash' for q in queries),coherence_sessions=2,coherence_turns=2,
        heldout_hop_length=10,train_seed=913741,evaluation_seed=782349,hash_lengths=[8,16],alphabet_size=52,
        disjoint_graphs_and_symbols=True,shuffled_edges=True,every_required_edge_present=True,
        no_table_name_or_source_location_in_hash_queries=True,all_distractor_chains_same_depth=True,
        reference_article='https://magic.dev/blog/100m-token-context-windows',
        reference_implementation='https://github.com/magicproduct/hash-hop/blob/main/hashhop/generate.py',
        variant='Magic-style task, arrow notation from the article, whitespace-separated trace answers; native chat wrapper and physical cold-recall controls are local extensions',
        zero_shot_baseline_is_not_a_capacity_limit=True,
        native_sft_cutoff=4096,largest_complete_training_record=max(lengths),training_tokens=sum(lengths),
        sft_chunking=False,external_answer_lookup_during_generation=False,native_mask_history=True,
        training_layout='native observation/acknowledgment/query/answer turns; model loss writes only observation then reads with fresh cache',
        objective='query CE through physical weights and decoded VAE posteriors, physical-weight-only auxiliary recall and counterfactual memory contrasts',
        evaluation_controls=['native-routed','oracle-units','wrong-unit','empty','source-visible-last'],
        cross_unit_routing='four candidates; report source coverage separately from answer accuracy',
        hash_max_new_tokens=256,coherence_max_new_tokens=1024,
        coherence_metrics='fact preservation, contradiction evidence, repetition, completion and full manual output review; no keyword-only claim of coherence',
        training_policy='Run only after baseline evaluation; native LlamaFactory, small separate continuation; do not alter the ongoing SFT dataset',
        source_sha256=hashlib.sha256((directory/'sources.json').read_bytes()).hexdigest(),
        question_sha256=hashlib.sha256((directory/'questions.json').read_bytes()).hexdigest())
    (directory/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    return protocol


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.directory,a.tokenizer),ensure_ascii=False))
