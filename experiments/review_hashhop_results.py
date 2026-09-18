"""Recompute strict scores and pair the completed SFT/HashHop generations."""
import argparse
from collections import Counter
import json
from pathlib import Path
from research_common import dump,digest


def read(path):
    return json.loads(Path(path).read_text())


def keyed(rows):
    result={(r['id'],r['condition'],r['turn']):r for r in rows}
    if len(result)!=len(rows):raise RuntimeError('Duplicate generation conditions')
    return result


def review(work):
    trial=work/'hashhop-memory-trial'
    old=read(work/'framing-fixed-generation/result.json');new=read(trial/'generation/result.json')
    if old['protocol_sha256']!=new['protocol_sha256'] or not old['body_fixed'] or not new['body_fixed']:
        raise RuntimeError('Protocol mismatch or evaluation mutated the body')
    before=keyed(read(work/'framing-fixed-generation/answers.json'))
    after=keyed(read(trial/'generation/answers.json'))
    if before.keys()!=after.keys() or len(after)!=278:raise RuntimeError('Incomplete matched generations')
    conditions={}
    for condition,expected in new['hash_scores'].items():
        rows=[r for r in after.values() if r['kind']=='hash' and r['condition']==condition]
        strict=sum(r['answer'].strip().split()==r['expected'].split() for r in rows)
        if len(rows)!=expected['n'] or strict!=expected['exact']:
            raise RuntimeError('Recorded summary differs from strict raw-answer grading')
        pairs=[(before[key],r) for key,r in after.items() if r['kind']=='hash' and r['condition']==condition]
        conditions[condition]=dict(n=len(rows),before_exact=sum(a['exact'] for a,b in pairs),
            after_exact=strict,gained=[b['id'] for a,b in pairs if b['exact'] and not a['exact']],
            lost=[b['id'] for a,b in pairs if a['exact'] and not b['exact']],
            same_text=sum(a['answer']==b['answer'] for a,b in pairs))
    coherence=[]
    for key,row in after.items():
        if row['kind']!='coherence':continue
        old_row=before[key]
        coherence.append(dict(id=row['id'],condition=row['condition'],turn=row['turn'],
            before_tokens=old_row['generated_tokens'],after_tokens=row['generated_tokens'],
            before_hit_limit=old_row['hit_generation_limit'],after_hit_limit=row['hit_generation_limit'],
            before_diagnostics=old_row['diagnostics'],after_diagnostics=row['diagnostics']))
    short=read(trial/'memory-check/result.json')
    hot={r['id']:r for r in read(trial/'memory-check/hot.json')}
    short_rows=read(trial/'memory-check/answers.json')
    mismatches=[dict(id=r['id'],hot=hot[r['id']]['answer'],cold=r['answer'])
                for r in short_rows if r['condition']=='oracle-cold' and r['answer']!=hot[r['id']]['answer']]
    groups={}
    for kind in ['text','hash','chain','visual']:
        rows=[r for r in short_rows if r['kind']==kind and r['condition']=='native-default']
        groups[kind]=dict(n=len(rows),exact=sum(r['exact'] for r in rows),
                          recorded_content=sum(r['content_correct'] for r in rows))
    result=dict(status='complete',baseline_checkpoint=old['checkpoint_id'],candidate_checkpoint=new['checkpoint_id'],
        protocol_sha256=new['protocol_sha256'],generation_conditions=len(after),
        hash_exact_match_rule='answer.strip().split() == expected.split(); hash characters and case must match; whitespace separators may differ',
        generation_empty_control='use_memory=False: memory-disabled native model, not a fresh active graph/FFN state. Training empty_memory_nll uses fresh pre-write states and is a different control.',
        matched_hash_scores=conditions,short_memory=short,short_native_by_kind=groups,
        hot_cold_text_mismatches=mismatches,coherence=coherence,
        coherence_semantic_verdict='See separately authored manual review; diagnostic substring/repetition scores are not semantic pass criteria.',
        artifacts_sha256={name:digest(trial/name) for name in ['generation/result.json','generation/answers.json',
            'memory-check/result.json','memory-check/answers.json','merged-weight-audit.json','training-review.json']})
    dump(trial/'evaluation-review.json',result)
    text='# HashHop continuation: actual generation results\n\n'
    text+=f"Parent SFT: `{old['checkpoint_id']}`. HashHop: `{new['checkpoint_id']}`. All 278 generation conditions completed under the same protocol.\n\n"
    text+='## Short fresh-memory recall\n\n'
    text+=f"Native routing selected the expected unit in {short['top1_native']}/{short['questions']} cases.\n\n"
    text+='| Condition | Exact answers | Recorded content score |\n|---|---:|---:|\n'
    for name,values in short['scores'].items():
        text+=f"| {name} | {values['exact']}/{values['n']} | {values['content']}/{values['n']} |\n"
    text+='\nContent scores use the original substring/visual rules and can reward an expected string embedded in a wrong chain. Strict exact scores and raw answers remain primary. The short protocol is not a matched before/after improvement claim because its old SFT reader had different framing.\n\n'
    text+='The generation condition named `empty` calls `use_memory=False`: it disables the memory branch. It differs from the training objective’s empty condition, which uses fresh pre-write graph/FFN states. Those NLL and generation controls should not be conflated.\n\n'
    text+='## Matched unseen HashHop queries\n\n'
    text+='| Condition | Parent SFT exact | HashHop exact | Gained / lost cases |\n|---|---:|---:|---:|\n'
    for name,values in conditions.items():
        text+=f"| {name} | {values['before_exact']}/{values['n']} | {values['after_exact']}/{values['n']} | {len(values['gained'])} / {len(values['lost'])} |\n"
    text+='\nHash exact match compares whitespace-separated hash sequences: every hash character and its case must match; whitespace separators may differ. These development queries include 1/2/4/8/10 hops and cross-unit chains. Training graphs and symbols are disjoint. Oracle conditions supply the correct unit selector; source-visible controls supply the original text. They are diagnostics, not replacements for native routing. This is not an extreme-context capacity benchmark.\n\n'
    text+='## Coherent generation\n\n'
    for name in dict.fromkeys(r['condition'] for r in coherence):
        rows=[r for r in coherence if r['condition']==name]
        text+=f"- {name}: {sum(r['before_hit_limit'] for r in rows)}/{len(rows)} parent outputs and {sum(r['after_hit_limit'] for r in rows)}/{len(rows)} candidate outputs hit the generation limit.\n"
    text+='\nUnabridged answers are in `generation/answers.json`; the separately authored manual review assesses factual consistency, repeated text and instruction following. Generation length alone is not a quality score.\n'
    (trial/'evaluation-review.md').write_text(text)
    print(json.dumps(dict(status='complete',matched_hash_scores=conditions),ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();review(a.work)
