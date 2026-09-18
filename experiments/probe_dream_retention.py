"""Measure real Dream replay on a frozen checkpoint without publishing a unit.

The existing replay API constructs candidates from source memory only. Its
validator merely captures the candidate and always declines publication;
questions/answers are opened only after consolidation has returned.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time
from research_common import load,dump,paced,generated,normalized,log,digest


def main(a):
    import torch
    model,processor=load(a.model);torch.manual_seed(17)
    archive=model.open_physical_archive(a.memory/'archive')
    versions={n:p._version for n,p in model.named_parameters()}
    pages={r['page']:r for r in json.loads((a.memory/'pages.json').read_text())}
    result=dict(checkpoint_id=model.config.memory_checkpoint_id,seed=17,scope='Three specified units: text-0, chain, visual-0; six existing probes, cap 64. No model or archive updates.',units=[],answers=[])
    for name in ['text-0','chain','visual-0']:
        archive.mount_async(pages[name]['unit_id']).result()
        with archive.session.pin() as (original,_):pass
        captured=[]
        def capture(before,candidate):
            captured.append(candidate)
            return False
        receipt=paced(archive.session.dream_consolidate,source_id=pages[name]['unit_id'],validator=capture)
        if receipt['published'] or len(captured)!=1:raise RuntimeError('Diagnostic must not publish any state')
        candidate=captured[0]
        unit=dict(page=name,source_unit=pages[name]['unit_id'],vae=receipt['vae'],feature_replay=receipt['feature_replay'],
            graph_relative_change=float((candidate.graph.fast-original.graph.fast).norm()/original.graph.fast.norm().clamp_min(1e-12)),
            adapter_change_norm=float(sum((candidate.adapters.factors[k]-v).square().sum() for k,v in original.adapters.factors.items()).sqrt()),
            all_sequence_checksums_preserved=[s.checksum for s in candidate.sequences]==[s.checksum for s in original.sequences],
            candidate_factors_writable=all(v.is_leaf and v.requires_grad for v in candidate.adapters.factors.values()),
            published=False)
        result['units'].append(unit);log('dream_candidate',**unit)
        queries=[q for q in json.loads((a.memory/'questions.json').read_text()) if q['page']==name]
        for condition,state in [('before-complete',original),('after-complete',candidate),
                                ('before-physical-only',replace(original,sequences=())),
                                ('after-physical-only',replace(candidate,sequences=()))]:
            session=type(archive.session)(model,rank=model.config.physical_memory['rank'],_initial_unit=state)
            try:
                for q in queries:
                    if condition.endswith('physical-only'):
                        # Explicit experimental component ablation, matching
                        # the existing matched-generation protocol. The public
                        # session reader always requires ordered observations.
                        with session.pin() as (unit,_),session.bank.use(unit.adapters),torch.no_grad():
                            answer=generated(processor,q['question'],model.generate,memory_state=unit.graph)
                    else:answer=generated(processor,q['question'],session.generate)
                    row=dict(id=q['id'],page=name,condition=condition,expected=q['answer'],**answer)
                    row['exact']=normalized(answer['answer'])==normalized(q['answer'])
                    result['answers'].append(row);dump(a.out,result);log('dream_retention_answer',**row)
            finally:session.close()
        del captured,candidate,original
    result['scores']={c:dict(n=sum(r['condition']==c for r in result['answers']),exact=sum(r['exact'] for r in result['answers'] if r['condition']==c))
        for c in ['before-complete','after-complete','before-physical-only','after-physical-only']}
    result['body_fixed']=all(p._version==versions[n] for n,p in model.named_parameters())
    result['questions_sha256']=digest(a.memory/'questions.json')
    result['completed']=True;dump(a.out,result);archive.close()
    log('dream_retention_complete',scores=result['scores'],body_fixed=result['body_fixed'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['model','memory','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();gate=os.environ.get('LFM2_START_GATE')
    while gate and not Path(gate).exists():time.sleep(.1)
    main(a)
