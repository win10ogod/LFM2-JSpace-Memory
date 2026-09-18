"""Read-only convergence assessment; stable high loss is not useful convergence."""
import argparse
import json
import math
from pathlib import Path
from research_common import dump,digest


def training_rows(path):
    decoder=json.JSONDecoder();rows={}
    for line in path.read_text().splitlines():
        at=line.find('{"status": "running"')
        if at<0:continue
        try:row=decoder.raw_decode(line[at:])[0]
        except ValueError:continue
        if 'step' in row:rows[row['step']]=row
    return [rows[k] for k in sorted(rows)]


def main(a):
    history=training_rows(a.run/'training.log')
    points=[]
    for p in (a.run/'training').glob('memory-eval-step-*.json'):
        value=json.loads(p.read_text());value['artifact_sha256']=digest(p);points.append(value)
    points.sort(key=lambda x:x['step']);latest=points[-1] if points else {}
    keys=['supervised_loss','compressed_recall_loss','fast_only_loss','consolidated_recall_loss']
    stable={}
    for k in keys:
        values=[p[k] for p in points[-3:] if k in p]
        stable[k]=(len(values)==3 and (max(values)-min(values))<=a.relative_tolerance*max(abs(values[-1]),1e-6))
    metric=lambda name:latest.get(name,float('inf'))
    codec=latest.get('codec_metrics',{})
    controls=min(metric('empty_memory_nll'),metric('wrong_memory_nll'))
    useful=dict(
        reconstruction_beats_zero=codec.get('mean_distortion',float('inf'))<codec.get('zero_decoder_distortion',0.),
        compressed_beats_empty_and_wrong=(metric('compressed_memory_nll')<controls if 'compressed_memory_nll' in latest else None),
        physical_beats_empty_and_wrong=(metric('physical_memory_nll')<controls if 'physical_memory_nll' in latest else None),
        post_dream_better=metric('consolidated_recall_loss')<metric('compressed_recall_loss'),
        policy_regret_small=metric('read_policy_regret')<=a.maximum_regret)
    # Compare row means with row means. Older reports without corresponding
    # positive row means remain unknown, never substituted with token means.
    generated=a.run/'counterfactual-generation/result.json'
    generation_finished=generated.exists()
    final_audit=a.run/'objective-audit/result.json';paired=None
    if final_audit.exists():
        detailed=json.loads(final_audit.read_text());paired=detailed['paired_cluster_bootstrap']
        useful.update(compressed_beats_empty_and_wrong=all(paired[k]['lower95']>0 for k in ['compressed_vs_empty','compressed_vs_wrong']),
            physical_beats_empty_and_wrong=all(paired[k]['lower95']>0 for k in ['physical_vs_empty','physical_vs_wrong']),
            post_dream_better=paired['dream_gain']['lower95']>0,
            policy_regret_small=detailed['read_policy_regret']<=a.maximum_regret)
    assessment='not_established'
    if all(stable.values()) and not all(useful.values()):assessment='stable_but_not_useful'
    elif all(stable.values()) and all(useful.values()):assessment='candidate_for_functional_review'
    result=dict(assessment=assessment,claim_of_convergence=False,
        latest_training_step=history[-1]['step'] if history else None,
        fixed_validation_steps=[p['step'] for p in points],stable=stable,useful_screening=useful,
        criteria=dict(fixed_validation_points=3,relative_range_tolerance=a.relative_tolerance,
                      maximum_selected_read_regret_nats_per_token=a.maximum_regret),
        teacher_forced_screening_is_not_generation_accuracy=True,free_generation_finished=generation_finished,
        paired_cluster_confidence_intervals=paired,
        limitations=['Different training batches cannot establish convergence.',
            'Soft policy CE has a moving target-entropy floor; compare excess KL and decision regret.',
            'Some older reports lack entropy-floor and paired-gain statistics.',
            'A stable but ineffective memory substrate does not pass.',
            'Final free-generation, retention, coherence and checkpoint-to-checkpoint review are required.'],
        latest_validation=latest,
        training_tail=[{k:r.get(k) for k in ['step',*keys,'read_policy_regret','consolidated_query_gain','codec_metrics']}
                       for r in history[-8:]])
    dump(a.out,result);print(json.dumps({k:result[k] for k in ['assessment','latest_training_step','fixed_validation_steps','stable','useful_screening']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--relative-tolerance',type=float,default=.01)
    p.add_argument('--maximum-regret',type=float,default=.05)
    main(p.parse_args())
