"""Diagnose native conversation framing without changing stored memories."""
import argparse
import importlib
import json
import os
from pathlib import Path
import time
import torch
from research_common import load,paced,dump,log


def main():
    p=argparse.ArgumentParser()
    for name in ['model','data','work','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tokens',type=int,default=256,help='Explicit short framing diagnostic, not the full coherence score')
    a=p.parse_args();gate=os.environ.get('LFM2_START_GATE')
    while gate and not Path(gate).exists():time.sleep(.1)
    a.out.mkdir(parents=True,exist_ok=False)
    model,processor=load(a.model);archive=model.open_physical_archive(a.work/'archive')
    module=importlib.import_module(type(model).__module__.rsplit('.',1)[0]+'.sequence_memory')
    sources={s['id']:s for s in json.loads((a.data/'sources.json').read_text())}
    pages={s['source_id']:s for s in json.loads((a.work/'pages.json').read_text())}
    questions=[q for q in json.loads((a.data/'questions.json').read_text()) if q['kind']=='coherence']
    rows=[];versions={n:x._version for n,x in model.named_parameters()}
    for q in questions:
        inputs=processor.tokenizer.apply_chat_template([dict(role='user',content=q['questions'][0])],
            tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True).to('cuda')
        unit=archive.session.bank.load_unit(archive._path(pages[q['required_sources'][0]]['unit_id']))
        prefix_length=sum(s.count for s in unit.sequences)
        with torch.no_grad():
            query_embedding=model.get_input_embeddings()(inputs['input_ids'])
            prefix=torch.cat([module.decode_segment(model,s) for s in unit.sequences]).to(query_embedding)[None]
        original=dict(inputs,inputs_embeds=torch.cat((prefix,query_embedding),dim=1),
            attention_mask=torch.ones((1,prefix_length+query_embedding.shape[1]),device='cuda',dtype=torch.long))
        if inputs['input_ids'][0,0]!=processor.tokenizer.bos_token_id:raise RuntimeError('No leading query BOS; do not guess a frame edit')
        e=original['inputs_embeds']
        moved=torch.cat((e[:,prefix_length:prefix_length+1],e[:,:prefix_length],e[:,prefix_length+1:]),dim=1)
        bos_first=dict(original,inputs_embeds=moved)
        for physical in [False,True]:
            for variant,prepared in [('original-prefix-before-bos',original),('bos-before-memory-prefix',bos_first)]:
                with torch.no_grad(),archive.session.bank.use(unit.adapters):
                    output=paced(model.generate,**prepared,memory_state=unit.graph,use_memory=physical,max_new_tokens=a.tokens,do_sample=False)
                tokens=output[0,inputs['input_ids'].shape[1]:]
                row=dict(id=q['id'],variant=variant,physical_weights=physical,tokens=len(tokens),
                    hit_generation_limit=len(tokens)==a.tokens,answer=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip(),
                    all_source_features_preserved=True,only_query_bos_relocated=variant.startswith('bos-'))
                rows.append(row);dump(a.out/'answers.json',rows);log('framing_diagnostic',**row)
        text='\n\n'.join(sources[k]['text'] for k in q['required_sources'])+'\n\n'+q['questions'][0]
        visible=processor.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,
            add_generation_prompt=True,return_tensors='pt',return_dict=True).to('cuda')
        with torch.no_grad():output=paced(model.generate,**visible,use_memory=False,max_new_tokens=a.tokens,do_sample=False)
        tokens=output[0,visible['input_ids'].shape[1]:]
        rows.append(dict(id=q['id'],variant='native-source-visible-chat',physical_weights=False,tokens=len(tokens),
            hit_generation_limit=len(tokens)==a.tokens,answer=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()))
        dump(a.out/'answers.json',rows)
    fixed=all(x._version==versions[n] for n,x in model.named_parameters())
    if not fixed:raise RuntimeError('Framing probe changed model parameters')
    dump(a.out/'result.json',dict(body_fixed=fixed,stored_memory_unchanged=True,diagnostic_max_new_tokens=a.tokens,
        scope='Controlled BOS relocation and native source-visible chat; no benchmark score substituted'))
    archive.close()


if __name__=='__main__':main()
