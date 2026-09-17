"""Test whether sparse J-components mediate more than their matched remainder.

Probe construction is independent of downstream question templates. A
nonnegative-fit remainder is NOT assumed orthogonal to the entire J-space.
"""
from pathlib import Path
import argparse,importlib,json,os,time
import torch
from torch.nn import functional as F
from research_common import load,paced,dump,log
from jacobian_diagnostic import record_layers
from test_jspace_functions import suite,evaluate


BASELINE_WORDS='''garden river mountain ocean forest desert bridge train bicycle airplane
cat dog horse sheep cow bird fish whale dolphin shark
apple orange lemon banana grape pear peach cherry plum melon
table chair window door house school library hospital station museum
music painting sculpture poetry novel dance theater cinema camera photograph
rain snow wind cloud sunshine thunder lightning storm rainbow fog
algebra geometry calculus physics chemistry biology geology astronomy medicine ecology
computer keyboard screen processor memory network compiler database algorithm program
bread rice pasta potato carrot tomato onion lettuce pepper bean
gold silver copper iron steel glass wood paper plastic cotton
clock calendar compass ruler scale thermometer telescope microscope battery engine'''.split()


@torch.no_grad()
def probe(model,tokenizer,concept):
    inputs=tokenizer.apply_chat_template([dict(role='user',content='Tell me about '+concept+'.')],
        tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True).to(model.lm_head.weight.device)
    with record_layers(model.model.language_model.layers,[26,28]) as values:
        model(**inputs,use_memory=False,use_cache=False,logits_to_keep=1)
    return {l:h[0,-1].detach().float().cpu() for l,h in values.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--lens',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();gate=os.environ.get('LFM2_START_GATE')
    while gate and not Path(gate).exists():time.sleep(.1)
    args.out.mkdir(exist_ok=False)
    conditions=['clean','full','j_component','remainder','remainder_clamp_j','j_clamp_remainder']
    protocol=dict(items=suite(True)['items'],conditions=conditions,baseline_concepts=BASELINE_WORDS,
        probe_template='Tell me about {concept}.',probe_position='last prompt token, before answer',
        centering='subtract mean of baseline concepts at the same layer',sparsity=16,
        decomposition='positive pursuit with projected nonnegative least squares refit',
        normalization='match every position intervention norm to full-probe swap',
        layers=dict(write=26,clamp=28),window_selection='fixed after exploratory pilot, new concepts',
        clamp='clean coordinates in union of original/target probe supports, up to 32 directions',
        no_downstream_question_used_for_probe=True,no_strength_selection=True)
    dump(args.out/'protocol.json',protocol)
    model,processor=load(args.model);tokenizer=processor.tokenizer;lens=model.open_concept_lens(args.lens)
    module=importlib.import_module(type(lens).__module__)
    baseline=[paced(probe,model,tokenizer,c) for c in BASELINE_WORDS]
    means={l:torch.stack([r[l] for r in baseline]).mean(0) for l in (26,28)}
    originals={c:paced(probe,model,tokenizer,c) for c in ('Kenya','Norway','Canada','Vietnam')}
    components={};quality={}
    for layer in (26,28):
        dictionary=paced(lens.dictionary,layer);quality[str(layer)]={}
        for concept,values in originals.items():
            full=(values[layer]-means[layer]).to(dictionary.device)
            sparse=paced(module.sparse_nonnegative,full[None],dictionary,sparsity=16)
            directions=dictionary[sparse.indices[0].long()]
            j=(sparse.coefficients[0,:,None]*directions).sum(0)
            components[(concept,layer)]=dict(full=full,j_component=j,remainder=full-j,support=directions)
            quality[str(layer)][concept]=dict(residual_fraction=float(sparse.residual_fraction[0]),
                coefficients=sparse.coefficients[0].tolist(),indices=sparse.indices[0].tolist(),
                support_decoded_for_display_only=[tokenizer.decode([i]) for i in sparse.indices[0].tolist()])
        log('jspace_probe_decomposition',layer=layer,quality=quality[str(layer)])
    dump(args.out/'decomposition.json',quality)
    matrices={l:lens.matrices[str(l)] for l in (26,28)};versions={n:p._version for n,p in model.named_parameters()}
    rows=[]
    for item in protocol['items']:
        a,b=item['original'],item['target']
        pairs={name:{l:F.normalize(torch.stack([components[(c,l)][name] for c in (a,b)]),dim=-1)
                     for l in (26,28)} for name in ('full','j_component','remainder')}
        j_clamp={28:torch.cat([components[(c,28)]['support'] for c in (a,b)])}
        for condition in conditions:
            kind={'clean':'full','full':'full','j_component':'j_component','remainder':'remainder',
                'remainder_clamp_j':'remainder','j_clamp_remainder':'j_component'}[condition]
            operation='clean' if condition=='clean' else 'swap_26_clamp_28' if 'clamp' in condition else 'swap_26'
            clamp=j_clamp if condition=='remainder_clamp_j' else {28:pairs['remainder'][28]}
            row=paced(evaluate,model,tokenizer,item,operation,matrices,direction_override=pairs[kind],
                clamp_override=clamp,norm_reference=pairs['full'])
            row['condition']=condition;rows.append(row);dump(args.out/'trials.json',rows)
            log('jspace_privilege_trial',**row)
    summary={}
    for condition in conditions:
        group=[r for r in rows if r['condition']==condition]
        summary[condition]=dict(n=len(group),counterfactual_prefix=sum(r['counterfactual_prefix'] for r in group),
            original_prefix=sum(r['original_prefix'] for r in group),
            mean_counterfactual_log_odds=sum(r['counterfactual_log_odds'] for r in group)/len(group))
    fixed=all(p._version==versions[n] and p.grad is None for n,p in model.named_parameters())
    result=dict(summary=summary,body_fixed=fixed,scope='fixed functional pilot, matched interventions',
        global_workspace_established=False,peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
    dump(args.out/'result.json',result);log('jspace_privilege_complete',**result);lens.close()


if __name__=='__main__':main()
