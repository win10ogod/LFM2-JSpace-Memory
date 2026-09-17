"""Read-only development probe of query serialization in native concept search.

No memory weights, stored addresses, model parameters or scoring functions are
changed. All user text is retained. Chat-control positions are a separately
named diagnostic, never silently removed from the existing baseline result.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import time
import torch
from research_common import load,paced,dump,question_inputs,log


def main():
    p=argparse.ArgumentParser()
    for name in ['model','work','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();gate=os.environ.get('LFM2_START_GATE')
    while gate and not Path(gate).exists():time.sleep(.1)
    a.out.mkdir(parents=True,exist_ok=False)
    model,processor=load(a.model);archive=model.open_physical_archive(a.work/'archive')
    helper=importlib.import_module(type(model).__module__.rsplit('.',1)[0]+'.concept_capture')
    versions={n:x._version for n,x in model.named_parameters()}
    questions=json.loads((a.work/'questions.json').read_text())
    pages={x['page']:x['unit_id'] for x in json.loads((a.work/'pages.json').read_text())}
    rows=[]
    for q in questions:
        text=q['question'];chat=question_inputs(processor,text)
        raw=processor(text=text,return_tensors='pt',truncation=False).to('cuda')
        serialized=processor.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True)
        encoded=processor.tokenizer(serialized,add_special_tokens=False,return_offsets_mapping=True)
        if encoded['input_ids']!=chat['input_ids'][0].tolist():raise RuntimeError('Chat offset tokenization differs')
        start=serialized.index(text);end=start+len(text)
        content=torch.tensor([[b>start and s<end for s,b in encoded['offset_mapping']]],device='cuda')
        keys={}
        keys['full-chat']=paced(archive.encode_query,chat)
        keys['raw-complete-question']=paced(archive.encode_query,raw)
        def contextual_content():
            with helper.capture_native_features(model,content) as features,torch.no_grad():
                model(**dict(chat,use_memory=False,use_cache=False,logits_to_keep=1))
            return archive.concept_lens.encode_features(features)
        keys['chat-context-content-positions']=paced(contextual_content)
        for variant,codes in keys.items():
            route=archive.query({},encoded_query=codes,top_k=3,device='cuda')
            row=dict(id=q['id'],variant=variant,expected=pages[q['page']],matches=route['matches'],
                top1=route['matches'][0]['unit_id']==pages[q['page']],user_text_unchanged=True)
            rows.append(row);dump(a.out/'routes.json',rows);log('query_format_route',**row)
    summary={v:dict(n=sum(x['variant']==v for x in rows),top1=sum(x['top1'] for x in rows if x['variant']==v))
             for v in dict.fromkeys(x['variant'] for x in rows)}
    fixed=all(x._version==versions[n] for n,x in model.named_parameters())
    if not fixed:raise RuntimeError('Query-format probe changed the body')
    dump(a.out/'result.json',dict(summary=summary,body_fixed=fixed,stored_memory_changed=False,
        role='development diagnosis of serialization mismatch, not a held-out test or memory training result'))
    archive.close()


if __name__=='__main__':main()
