"""Predeclared functional interventions, separate from memory addressing.

This pilot measures report, silent intermediates, mediation, flexible use,
and directed modulation. No success is inferred from readable lens tokens.
The complete suite and all conditions are saved before the first trial.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse,json,os,time,re
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from research_common import load,dump,log,paced
from jacobian_diagnostic import record_layers


def swap_delta(hidden,directions):
    """Swap basis coefficients, preserving the orthogonal residual exactly."""
    coefficients=hidden@torch.linalg.pinv(directions)
    return (coefficients.flip(-1)-coefficients)@directions


def reference_swap_delta(hidden,directions,reference):
    """Hold the clean pass's swapped target; later edits do not swap it back."""
    inverse=torch.linalg.pinv(directions)
    return ((reference@inverse).flip(-1)-hidden@inverse)@directions


def suite(validation=False):
    countries=[('France','Japan','home to the Eiffel Tower','Paris','French','Europe'),
        ('Japan','France','home to Mount Fuji','Tokyo','Japanese','Asia'),
        ('Italy','China','shaped like a boot','Rome','Italian','Europe'),
        ('China','Italy','home to the Great Wall','Beijing','Chinese','Asia')]
    if validation:
        countries=[('Kenya','Norway','home to Lake Turkana and the Maasai Mara','Nairobi','Swahili','Africa'),
            ('Norway','Kenya','home to Bergen and the Lofoten islands','Oslo','Norwegian','Europe'),
            ('Canada','Vietnam','whose flag has a red maple leaf','Ottawa','English','North America'),
            ('Vietnam','Canada','home to Ha Long Bay','Hanoi','Vietnamese','Asia')]
    lookup={r[0]:r for r in countries};items=[]
    templates=[('report','The country {cue} is',0),
        ('capital','The capital city of the country {cue} is',3),
        ('language','The main language spoken in the country {cue} is',4),
        ('continent','The continent containing the country {cue} is',5)]
    for original,target,cue,*_ in countries:
        for task,template,index in templates:
            a,b=lookup[original][index],lookup[target][index]
            # Continent pairs were chosen to differ. No post-result filtering.
            items.append(dict(id=original+'-'+task,task=task,prompt=template.format(cue=cue),
                original=original,target=target,answer=a,counterfactual=b))
    conditions=['clean',*[f'swap_{l}' for l in (4,8,14,20,26,28)],'swap_14_20',
        'random_0','random_1','random_2','wrong_concept','position_control','ablate_14',
        'ablate_14_rescue_20','swap_14_clamp_20','answer_swap_14']
    if validation:
        conditions=['clean','swap_20','swap_26','swap_28','swap_26_28',
            'random_26_0','random_26_1','random_26_2','wrong_concept_26','position_control_26',
            'ablate_26','ablate_26_rescue_28','swap_26_clamp_28','answer_swap_20','answer_swap_26']
    return dict(items=items,conditions=conditions,seed=19473,strength=1.,
        generation=dict(max_new_tokens=12,do_sample=False),
        primary='counterfactual full-answer log odds and unforced completion prefix; all clean failures retained',
        evidence_scope='functional pilot on merged LFM2.5-VL-3B derivative, not unmodified upstream weights',
        no_tuning_on_results=True,validation=validation,
        window_selection='layer 26 selected on disjoint France/Japan/Italy/China pilot' if validation else 'exploratory depth sweep',
        sparse_privilege_test_complete=False,
        limitations=['small fixed concept set','lens has 12 calibration prompts',
            'pair-orthogonal controls are not the full non-J-space remainder',
            'no general workspace conclusion from this pilot alone'])


@contextmanager
def interventions(model,condition,directions,wrong,answers,clean,length,seed,clamp_override=None,norm_reference=None):
    handles=[];diagnostics=[]
    def edit(layer,operation):
        def hook(module,args,output):
            if output.shape[1]<length:return # Cached answer tokens are not directly steered.
            x=output[:,:length].float();d=directions[layer].to(x.device)
            delta=swap_delta(x,d)
            if operation in ('reference_swap','reference_random'):
                delta=reference_swap_delta(x,d,clean[layer].to(x.device))
            if operation in ('random','reference_random'):
                rng=torch.Generator(device='cpu').manual_seed(seed+layer)
                r=torch.randn(x.shape[-1],generator=rng).to(x.device)
                r=r-(r@torch.linalg.pinv(d))@d
                delta=delta.norm(dim=-1,keepdim=True)*F.normalize(r,dim=0)
            elif operation=='wrong':
                candidate=swap_delta(x,wrong[layer].to(x.device))
                delta=F.normalize(candidate,dim=-1)*delta.norm(dim=-1,keepdim=True)
            elif operation=='position':
                magnitude=delta.norm();first=torch.zeros_like(delta)
                first[:,0]=F.normalize(delta[:,0],dim=-1)*magnitude;delta=first
            elif operation=='ablate':
                coefficients=x@torch.linalg.pinv(d)
                delta=-coefficients[...,:1]*d[0]
            elif operation in ('rescue','clamp'):
                if clamp_override is not None and layer in clamp_override:d=clamp_override[layer].to(x.device)
                actual=x@torch.linalg.pinv(d)
                reference=clean[layer].to(x.device)@torch.linalg.pinv(d)
                change=reference-actual
                if operation=='rescue':change[...,1]=0
                delta=change@d
            elif operation=='answer':delta=swap_delta(x,answers[layer].to(x.device))
            if norm_reference is not None and operation=='swap':
                reference_delta=swap_delta(x,norm_reference[layer].to(x.device))
                delta=F.normalize(delta,dim=-1)*reference_delta.norm(dim=-1,keepdim=True)
            diagnostics.append(dict(layer=layer,operation=operation,
                delta_norm=float(delta.norm()),relative_norm=float(delta.norm()/x.norm().clamp_min(1e-12))))
            changed=output.clone();changed[:,:length]=(x+delta).to(output.dtype)
            return changed
        return hook
    operations=[]
    if condition.startswith('reference_swap_'):
        operations=[(int(l),'reference_swap') for l in condition.split('_')[2:]]
    elif condition.startswith('reference_random_'):
        operations=[(int(l),'reference_random') for l in condition.split('_')[2:]]
    elif condition.startswith('swap_') and 'clamp' not in condition:
        operations=[(int(l),'swap') for l in condition.split('_')[1:]]
    elif condition.startswith('random_'):
        parts=condition.split('_');operations=[(int(parts[1]) if len(parts)==3 else 14,'random')];seed+=int(parts[-1])
    elif condition.startswith('wrong_concept'):
        operations=[(int(condition.split('_')[-1]) if condition[-1].isdigit() else 14,'wrong')]
    elif condition.startswith('position_control'):
        operations=[(int(condition.split('_')[-1]) if condition[-1].isdigit() else 14,'position')]
    elif condition.startswith('ablate_'):
        parts=condition.split('_');operations=[(int(parts[1]),'ablate')]
        if len(parts)==4:operations.append((int(parts[3]),'rescue'))
    elif condition.startswith('swap_') and 'clamp' in condition:
        parts=condition.split('_');operations=[(int(parts[1]),'swap'),(int(parts[3]),'clamp')]
    elif condition.startswith('answer_swap_'):operations=[(int(condition.split('_')[-1]),'answer')]
    elif condition!='clean':raise ValueError(condition)
    try:
        for layer,operation in operations:
            handles.append(model.model.language_model.layers[layer].register_forward_hook(edit(layer,operation)))
        yield diagnostics
    finally:
        for h in handles:h.remove()


@torch.no_grad()
def evaluate(model,tokenizer,item,condition,matrices,*,direction_override=None,clamp_override=None,norm_reference=None):
    device=model.lm_head.weight.device;layers=sorted(matrices)
    ids=tokenizer(item['prompt'],return_tensors='pt')['input_ids'].to(device);length=ids.shape[1]
    def dictionary_pair(a,b,*,strict=True):
        indices=[]
        for word in (a,b):
            tokens=tokenizer(' '+word,add_special_tokens=False)['input_ids']
            if strict and len(tokens)!=1:raise ValueError(f'pilot requires a single-token concept: {word!r}: {tokens}')
            indices.append(tokens[0])
        rows=model.lm_head.weight[indices].float()*model.model.language_model.embedding_norm.weight.float()
        return {l:F.normalize(rows@j.to(device),dim=-1) for l,j in matrices.items()}
    directions=dictionary_pair(item['original'],item['target'])
    if direction_override is not None:directions=direction_override
    wrong=dictionary_pair('Germany','Brazil')
    answers=dictionary_pair(item['answer'],item['counterfactual'],strict=False)
    with record_layers(model.model.language_model.layers,layers) as clean:
        baseline=model(input_ids=ids,use_memory=False,use_cache=False,logits_to_keep=1).logits[0,-1].float()
    clean={l:h.detach().float() for l,h in clean.items()}
    with interventions(model,condition,directions,wrong,answers,clean,length,19473,
                       clamp_override=clamp_override,norm_reference=norm_reference) as edits:
        output=model(input_ids=ids,use_memory=False,use_cache=False,logits_to_keep=1).logits[0,-1].float()
        logp=output.log_softmax(-1);reference=baseline.log_softmax(-1)
        scores={}
        for label,answer in (('original',item['answer']),('counterfactual',item['counterfactual'])):
            a=tokenizer(item.get('answer_prefix',' ')+answer,add_special_tokens=False)['input_ids']
            joined=torch.cat((ids,torch.tensor([a],device=device)),dim=1)
            logits=model(input_ids=joined,use_memory=False,use_cache=False).logits[0,length-1:-1].float()
            values=logits.log_softmax(-1)[torch.arange(len(a),device=device),torch.tensor(a,device=device)]
            scores[label]=float(values.sum())
        generated=model.generate(input_ids=ids,use_memory=False,do_sample=False,max_new_tokens=12)
    text=tokenizer.decode(generated[0,length:],skip_special_tokens=True)
    starts=lambda answer:bool(re.match(r'^\s*'+re.escape(answer)+r'\b',text,re.IGNORECASE))
    top=output.topk(10).indices
    return dict(id=item['id'],task=item['task'],condition=condition,answer=text,
        original_prefix=starts(item['answer']),counterfactual_prefix=starts(item['counterfactual']),
        answer_log_probs=scores,counterfactual_log_odds=scores['counterfactual']-scores['original'],
        output_kl=float((reference.exp()*(reference-logp)).sum()),
        top10=[tokenizer.decode([i]) for i in top.tolist()],edits=edits[:len(layers)],
        answer_direction_control='first token only; full answer is used for likelihood scoring',
        hit_generation_limit=generated.shape[1]-length==12)


@torch.no_grad()
def modulation(model,tokenizer,matrices,concepts=('France','Japan','Italy','China')):
    """Identical copied text, active-focus vs passive-mention control.

    Lens scores are teacher-forced on the exact same copy positions. Free
    generation is reported separately; this does not pretend forcing is proof
    that the model spontaneously retained the concept while copying.
    """
    copied='The quiet garden was covered with fresh snow.';results=[]
    for concept in concepts:
        for mode in ('focus','mention'):
            instruction=(f'While copying, silently concentrate on {concept}.' if mode=='focus' else
                f'The word {concept} is mentioned here, but focus only on copying.')
            prompt=tokenizer.apply_chat_template([dict(role='user',content=instruction+
                ' Output only this exact sentence: '+copied)],tokenize=False,add_generation_prompt=True)
            prefix=tokenizer(prompt,return_tensors='pt')['input_ids'].to(model.lm_head.weight.device)
            continuation=tokenizer(copied,add_special_tokens=False,return_tensors='pt')['input_ids'].to(prefix.device)
            ids=torch.cat((prefix,continuation),dim=1);layers=sorted(matrices)
            with record_layers(model.model.language_model.layers,layers) as values:
                model(input_ids=ids,use_memory=False,use_cache=False,logits_to_keep=1)
            token=tokenizer(' '+concept,add_special_tokens=False)['input_ids']
            if len(token)!=1:raise ValueError('modulation concept is not one token')
            row=model.lm_head.weight[token[0]].float()*model.model.language_model.embedding_norm.weight.float()
            readouts={l:float(F.cosine_similarity(values[l][0,prefix.shape[1]:].float(),
                (row@j.to(row.device))[None],dim=-1).mean()) for l,j in matrices.items()}
            output=model.generate(input_ids=prefix,use_memory=False,do_sample=False,max_new_tokens=32)
            text=tokenizer.decode(output[0,prefix.shape[1]:],skip_special_tokens=True).strip()
            results.append(dict(concept=concept,mode=mode,cosine=readouts,free_copy=text,
                copied_exactly=text==copied,measurement='teacher-forced matched surface tokens'))
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--lens',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--validation',action='store_true')
    args=p.parse_args();gate=os.environ.get('LFM2_START_GATE')
    while gate and not Path(gate).exists():time.sleep(.1)
    args.out.mkdir(exist_ok=False);protocol=suite(args.validation);dump(args.out/'protocol.json',protocol)
    model,processor=load(args.model);tokenizer=processor.tokenizer
    report=json.loads((args.lens/'result.json').read_text())
    if report['protocol']['checkpoint_id']!=model.config.memory_checkpoint_id:raise ValueError('lens identity mismatch')
    matrices={int(k):v for k,v in load_file(str(args.lens/'jacobians.safetensors')).items()}
    versions={n:v._version for n,v in model.named_parameters()};rows=[]
    for item in protocol['items']:
        for condition in protocol['conditions']:
            row=paced(evaluate,model,tokenizer,item,condition,matrices)
            rows.append(row);dump(args.out/'trials.json',rows);log('jspace_function',**row)
    controlled=paced(modulation,model,tokenizer,matrices,tuple(dict.fromkeys(i['original'] for i in protocol['items'])))
    dump(args.out/'modulation.json',controlled)
    fixed=all(v._version==versions[n] and v.grad is None for n,v in model.named_parameters())
    groups={}
    for condition in protocol['conditions']:
        group=[r for r in rows if r['condition']==condition]
        groups[condition]=dict(n=len(group),original_prefix=sum(r['original_prefix'] for r in group),
            counterfactual_prefix=sum(r['counterfactual_prefix'] for r in group),
            mean_counterfactual_log_odds=sum(r['counterfactual_log_odds'] for r in group)/len(group),
            mean_output_kl=sum(r['output_kl'] for r in group)/len(group))
    result=dict(groups=groups,body_fixed=fixed,calibration=report['protocol'],
        global_workspace_established=False,missing_evidence=protocol['limitations'],
        peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
    if not fixed:raise RuntimeError('body changed during interventions')
    dump(args.out/'result.json',result);log('jspace_functions_complete',**result)


if __name__=='__main__':main()
