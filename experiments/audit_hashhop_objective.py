"""Audit the completed pilot's labels, task shortcuts, and contrast gradients."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
import sys

from research_common import dump,digest


def parse(row):
    source,_,query,target=row['messages']
    table=dict(line.split(' → ') for line in source['content'].splitlines())
    match=re.search(r'start at ([A-Za-z]+) and follow exactly (\d+) arrows',query['content'])
    start,hops=match.group(1),int(match.group(2))
    task='trace' if 'every visited value' in query['content'] else 'jump'
    return table,start,hops,task,target['content'],query['content']


def walk(table,start,count=None):
    visited=[];seen=set();value=start
    while value in table and (count is None or len(visited)<count):
        if value in seen:raise ValueError('Cycle in pilot chain')
        seen.add(value);value=table[value];visited.append(value)
    if count is not None and len(visited)!=count:raise ValueError('Missing required edge')
    return visited


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rows=[json.loads(line) for line in (a.data/'train.jsonl').read_text().splitlines()]
    counts=Counter();bad=[];questions=defaultdict(set);per_graph=defaultdict(set)
    for row in rows:
        table,start,hops,task,answer,query=parse(row)
        visited=walk(table,start,hops);terminal_path=walk(table,start)
        expected=' '.join(visited) if task=='trace' else visited[-1]
        shortcut=' '.join(terminal_path) if task=='trace' else terminal_path[-1]
        counts['rows']+=1;counts['starts_at_root']+=start not in set(table.values())
        counts['requested_target_is_terminal']+=visited[-1] not in table
        counts['ignore_hop_count_solver_correct']+=shortcut==answer
        if answer!=expected:bad.append(row['id'])
        questions[query].add(answer)
        per_graph[row['id'].rsplit('-',1)[0]].add((start,hops))
    graphs={g['id']:g for g in json.loads((a.data/'evaluation-graphs.json').read_text())}
    evaluation=Counter()
    for query in json.loads((a.data/'questions.json').read_text()):
        if query['kind']!='hash':continue
        name=query['required_sources'][0];graph=graphs['cross' if name.startswith('cross') else name]
        table=dict(graph['edges']);start=re.search(r'start at ([A-Za-z]+)',query['question']).group(1)
        full=walk(table,start);shortcut=' '.join(full) if query['task']=='trace' else full[-1]
        evaluation['queries']+=1;evaluation['ignore_hop_count_solver_correct']+=shortcut==query['answer']
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from lfm2_titans.memory_recall_training import wrong_memory_rows
    positive=torch.tensor([8.],requires_grad=True)
    empty=torch.tensor([6.],requires_grad=True);wrong=torch.tensor([7.9],requires_grad=True)
    contrast=.1*(torch.relu(.25+positive-empty)+torch.relu(.25+positive-wrong)).mean()
    contrast.backward()
    root=Path(__file__).resolve().parents[1]
    result=dict(status='complete',training=dict(counts),incorrect_target_ids=bad,
        independent_training_graphs=len(per_graph),
        distinct_start_hop_combinations_per_graph=dict(Counter(len(q) for q in per_graph.values())),
        same_query_with_different_memory_answers=sum(len(v)>1 for v in questions.values()),
        evaluation=dict(evaluation),
        contrast_gradient_probe=dict(loss=float(contrast.detach()),correct_nll_gradient=float(positive.grad),
            empty_nll_gradient=float(empty.grad),wrong_nll_gradient=float(wrong.grad),
            meaning='This contrast term admits loss reduction by worsening negative contexts. It is an objective loophole, not proof of the learned model taking that shortcut.'),
        wrong_memory_selection_for_eight_distinct_sources=wrong_memory_rows([dict(source=torch.tensor([i])) for i in range(8)]),
        inspection_findings=[
            'Training always supplies one written unit directly. Archive index retrieval and cross-unit composition receive no query-selection supervision.',
            'Native backbone is frozen in this pilot, and native-concept addressing bypasses the trainable memory branch.',
            'Training query CE uses decoded posterior means without exact reconstruction corrections; production recall uses the complete corrected observation representation.',
            'Generation empty controls disable the memory branch; training empty controls use fresh pre-write states.'],
        causal_attribution='The shortcuts and objective gradients are verified. Their causal contribution to the final score requires a controlled continuation with corrected training design and unchanged architecture.',
        summary=(f"Checked {len(rows)} training labels; {len(bad)} are arithmetically incorrect. "
            f"{counts['starts_at_root']} queries start at chain roots and {counts['requested_target_is_terminal']} end at terminal nodes. "
            f"A solver that ignores the requested hop count solves {counts['ignore_hop_count_solver_correct']}/{len(rows)} training rows "
            f"but only {evaluation['ignore_hop_count_solver_correct']}/{evaluation['queries']} evaluation queries. "
            f"There are {sum(len(v)>1 for v in questions.values())} same-query/different-memory target pairs. "
            'The implemented contrast loss can decrease by raising empty/wrong-context NLL. This pilot also omits retrieval/cross-unit supervision and trains a compressed-only read that differs from production corrected recall. These are training-design defects; the results do not isolate an architectural capacity failure.'),
        files_sha256={str(path.relative_to(root)) if path.is_relative_to(root) else path.name:digest(path)
                      for path in [a.data/'train.jsonl',a.data/'questions.json',root/'src/lfm2_titans/memory_recall_training.py',root/'experiments/hashhop_curriculum.py']})
    dump(a.out,result);print(json.dumps({k:v for k,v in result.items() if k not in ['files_sha256','summary','inspection_findings']},indent=2))


if __name__=='__main__':main()
