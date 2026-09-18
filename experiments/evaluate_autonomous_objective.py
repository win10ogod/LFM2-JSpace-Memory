"""Fixed held-out loss/decision audit; no optimizer and no backbone updates."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from research_common import load,dump,paced,log


def main(a):
    import torch
    from datasets import load_from_disk
    os.environ['DISABLE_VERSION_CHECK']='1'
    sys.path.insert(0,'/mnt/f/稠密轉MOE試驗/LlamaFactory/src')
    from llamafactory.hparams import DataArguments
    from llamafactory.data import get_template_and_fix_tokenizer,SFTDataCollatorWith4DAttentionMask
    model,processor=load(a.model)
    model.config.native_joint_sft=True
    model.config.native_memory_recall=json.loads(a.preprocessing.read_text())['episode_spec']
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    data=load_from_disk(a.tokenized)['validation']
    if len(data)!=128:raise ValueError('Unexpected fixed validation size')
    template=get_template_and_fix_tokenizer(processor.tokenizer,DataArguments(template='lfm2_vl',cutoff_len=4096))
    collator=SFTDataCollatorWith4DAttentionMask(template=template,model=model,tokenizer=processor.tokenizer,
        processor=processor,pad_to_multiple_of=8,label_pad_token_id=-100,block_diag_attn=False,
        neat_packing=False,attn_implementation=model.config._attn_implementation,compute_dtype=torch.bfloat16)
    versions={n:p._version for n,p in model.named_parameters()};reports=[]
    def evaluate(batch):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):return model(**batch)
    for offset in range(0,len(data),8):
        batch=collator([data[i] for i in range(offset,min(offset+8,len(data)))])
        batch={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in batch.items()}
        output=paced(evaluate,batch)
        report=dict(model._native_sft_memory.last,total_loss=float(output.loss),offset=offset)
        reports.append(report);dump(a.out/'batches.json',reports)
        log('fixed_objective_audit',offset=offset,loss=report['total_loss'],
            read_policy=report['read_policy_statistics'],consolidation_policy=report['consolidation_policy_statistics'])
    if any(p._version!=versions[n] for n,p in model.named_parameters()):raise RuntimeError('Audit changed model weights')
    rows=[x for r in reports for x in r['per_example']]
    # Sources connected by the same query form one counterfactual cluster.
    import numpy as np
    parent={r['source_hash']:r['source_hash'] for r in rows};queries={}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for r in rows:
        key=r['query_hash'];source=r['source_hash']
        if key in queries:parent[find(source)]=find(queries[key])
        else:queries[key]=source
    clusters={}
    for r in rows:clusters.setdefault(find(r['source_hash']),[]).append(r)
    if len(parent)!=16 or len(queries)!=64 or len(clusters)!=8:
        raise ValueError('Fixed counterfactual pairing/grouping changed')
    rng=np.random.default_rng(731);groups=list(clusters.values());intervals={}
    for name,left,right in [('compressed_vs_empty','empty_nll','compressed_nll'),
                            ('compressed_vs_wrong','wrong_nll','compressed_nll'),
                            ('physical_vs_empty','empty_nll','physical_nll'),
                            ('physical_vs_wrong','wrong_nll','physical_nll'),
                            ('dream_gain','compressed_nll','consolidated_nll')]:
        if any(r[left] is None or r[right] is None for r in rows):raise ValueError('A paired control is unavailable')
        sums=np.array([sum(r[left]-r[right] for r in g) for g in groups]);counts=np.array([len(g) for g in groups])
        draws=rng.integers(len(groups),size=(5000,len(groups)))
        estimates=sums[draws].sum(1)/counts[draws].sum(1)
        intervals[name]=dict(mean=float(sums.sum()/counts.sum()),
            lower95=float(np.quantile(estimates,.025)),upper95=float(np.quantile(estimates,.975)))
    result=dict(checkpoint_id=model.config.memory_checkpoint_id,examples=len(rows),body_fixed=True,
        independent_counterfactual_clusters=len(groups),paired_cluster_bootstrap=intervals,
        read_policy_regret=sum(r['selected_nll']-min(r['complete_nll'],r['compressed_nll'],r['physical_nll']) for r in rows)/len(rows),
        policy_statistics={n:{k:sum(r[n][k]*r['physical_batch'] for r in reports)/len(rows)
            for k in reports[0][n]} for n in ['read_policy_statistics','consolidation_policy_statistics']},
        per_example=rows,scope='Fixed 128-question teacher-forced audit, batch 8; correlated questions grouped by shared source/query. Not free-generation accuracy.')
    dump(a.out/'result.json',result);log('fixed_objective_complete',**{k:v for k,v in result.items() if k!='per_example'})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['model','tokenized','preprocessing','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);gate=os.environ.get('LFM2_START_GATE')
    while gate and not Path(gate).exists():time.sleep(.1)
    main(a)
