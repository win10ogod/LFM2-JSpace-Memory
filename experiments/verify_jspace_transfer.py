"""Native chat-format transfer and fixed-reference multi-layer interventions."""
from pathlib import Path
import argparse,json,os,time
from safetensors.torch import load_file
from research_common import load,paced,dump,log
from test_jspace_functions import suite,evaluate
from build_jspace_report import aggregate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--lens',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();gate=os.environ.get('LFM2_START_GATE')
    while gate and not Path(gate).exists():time.sleep(.1)
    args.out.mkdir(exist_ok=False)
    model,processor=load(args.model);tokenizer=processor.tokenizer
    versions={n:p._version for n,p in model.named_parameters()}
    items=suite()['items']+suite(True)['items']
    for item in items:
        item['source_prompt']=item['prompt'];item['answer_prefix']=''
        instruction='Complete the following sentence. Output only the missing answer, without explanation.\n'+item['prompt']
        item['prompt']=tokenizer.apply_chat_template([dict(role='user',content=instruction)],
            tokenize=False,add_generation_prompt=True)
    conditions=['clean','swap_26','swap_26_28','reference_swap_26_28','reference_swap_20_26_28',
        'random_26_0','reference_random_26_28','reference_random_20_26_28','wrong_concept_26',
        'ablate_26','ablate_26_rescue_28']
    protocol=dict(items=items,conditions=conditions,scope='predeclared format-transfer follow-up on known concepts',
        prior_results_not_overwritten=True,reference_swap='fixed clean-pass target coefficients at every intervened layer',
        ordinary_swap='swaps current coefficients; repeated swaps may reverse earlier edits',
        full_answer_scoring=True,diagnostic_output_tokens=12,rest_ratio=1.)
    dump(args.out/'protocol.json',protocol)
    report=json.loads((args.lens/'result.json').read_text())
    if report['protocol']['checkpoint_id']!=model.config.memory_checkpoint_id:raise ValueError('wrong lens checkpoint')
    matrices={int(k):v for k,v in load_file(str(args.lens/'jacobians.safetensors')).items()};rows=[]
    for item in items:
        for condition in conditions:
            row=paced(evaluate,model,tokenizer,item,condition,matrices)
            rows.append(row);dump(args.out/'trials.json',rows);log('jspace_transfer',**row)
    fixed=all(p._version==versions[n] and p.grad is None for n,p in model.named_parameters())
    if not fixed:raise RuntimeError('inference body changed')
    result=dict(summary=aggregate(rows),body_fixed=fixed,scope=protocol['scope'],
        global_workspace_established=False)
    dump(args.out/'result.json',result);log('jspace_transfer_complete',**result)


if __name__=='__main__':main()
