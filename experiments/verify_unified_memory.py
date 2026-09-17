"""Fresh-source acceptance of configured native concept storage and routing."""
from pathlib import Path
import argparse,hashlib,json,os,random,string,time
from research_common import load,paced,dump,log,question_inputs,generated,normalized,address_inputs
def observation(processor, source):
    return processor(text=source['source'],return_tensors='pt',truncation=False).to('cuda')


def measured(q,answer,condition,**values):
    # Content accuracy tolerates surrounding prose, but never fixes characters
    # or case. Exact answer and free-form content are reported separately.
    actual=normalized(answer['answer']);expected=normalized(q['answer'])
    row=dict(answer,id=q['id'],kind=q['kind'],condition=condition,expected=q['answer'],
        exact=actual==expected,contains=expected in actual,**values)
    if q['kind']=='visual':
        row['content_correct']=('紅' in actual and '圓' in actual and '藍' not in actual and '方' not in actual) if '紅' in expected else (
            '藍' in actual and ('正方' in actual or '方形' in actual) and '紅' not in actual and '圓' not in actual)
    else:row['content_correct']=row['contains']
    return row


def prepare(work):
    from PIL import Image,ImageDraw
    work.mkdir(exist_ok=False);seed=int.from_bytes(os.urandom(8),'little');rng=random.Random(seed)
    sources=[];questions=[]
    for i,name in enumerate(('黛蘭修復站','洛青測試站','望嵐研究站','沐珀維護站','蒼榆觀測站','杏澄資料站')):
        code=str(rng.randrange(100000,1000000));tag=''.join(rng.choices(string.ascii_letters+string.digits,k=8))
        page=f'text-{i}';sources.append(dict(id=page,kind='text',source=f'本次全新資料：{name}的維修碼是 {code}，備份識別字串是 {tag}。'))
        questions.extend([dict(id='code-'+str(i),page=page,kind='text',question=f'{name}的維修碼是什麼？只回答六位數字。',answer=code),
            dict(id='tag-'+str(i),page=page,kind='hash',question=f'{name}的備份識別字串是什麼？只回答八個原始字元，大小寫不變。',answer=tag)])
    nodes=[''.join(rng.choices(string.ascii_letters+string.digits,k=8)) for _ in range(34)]
    pairs=[f'{a} → {b}' for a,b in zip(nodes,nodes[1:])];rng.shuffle(pairs)
    sources.append(dict(id='chain',kind='text',source='燦石連結站的新映射表，每個箭頭通往唯一下一站：\n'+'\n'.join(pairs)))
    questions.extend([dict(id='hop1',page='chain',kind='chain',question=f'燦石連結站的映射表中，{nodes[11]} 的下一站是什麼？只輸出八個原始字元。',answer=nodes[12]),
        dict(id='hop3-all',page='chain',kind='chain',question=f'燦石連結站的映射表中，從 {nodes[11]} 沿箭頭走三次，依序列出三個到達節點，空格分隔並保留大小寫。',answer=' '.join(nodes[12:15])),
        dict(id='hop3-final',page='chain',kind='chain',question=f'燦石連結站的映射表中，從 {nodes[11]} 沿箭頭走三次，最後到哪裡？只回答最後節點的八個原始字元。',answer=nodes[14])])
    for i,(name,color,shape,answer) in enumerate((('艾沐影像站','#f02020','circle','紅色圓形'),('曦原影像站','#2020f0','square','藍色正方形'))):
        image=Image.new('RGB',(224,224),'white');draw=ImageDraw.Draw(image)
        box=(30+rng.randrange(12),30+rng.randrange(12),185+rng.randrange(12),185+rng.randrange(12))
        (draw.ellipse if shape=='circle' else draw.rectangle)(box,fill=color)
        path=work/f'image-{i}.png';image.save(path)
        sources.append(dict(id=f'visual-{i}',kind='visual',image=str(path),reference=f'這張圖是{name}本次提交的觀測。請記住圖片內容。'))
        questions.append(dict(id=f'visual-{i}',page=f'visual-{i}',kind='visual',question=f'{name}提交的圖片中，主要圖形是什麼顏色、什麼形狀？只輸出顏色與形狀。',answer=answer))
    dump(work/'sources.json',sources);dump(work/'questions.json',questions)
    dump(work/'protocol.json',dict(seed=seed,sources=len(sources),questions=len(questions),
        data_scope='new random codes, case-sensitive strings, shuffled hash chain, new native images with source references',
        writer='default observe + one source-only text next-token update + default append; generation forbidden',
        readers='new process; native router, correct-unit oracle, wrong-unit, empty and source-visible controls',
        source_visible_control='only after every memory-only response',output_tokens=64,rest_ratio=1.,
        gates=['all saved units v6','zero text generation while addressing','CPU/GPU same routing',
            'hot/cold oracle identical',
            'no lost answers that correct-unit oracle can recall','frozen body unchanged',
            'recall exceeds wrong-unit and empty-memory controls','text, random strings, and vision each have successful recall'],
        no_tuning_after_answers=True,no_long_context_claim=True,
        sources_sha256=hashlib.sha256((work/'sources.json').read_bytes()).hexdigest(),
        questions_sha256=hashlib.sha256((work/'questions.json').read_bytes()).hexdigest()))


def source_inputs(processor,source):
    if source['kind']!='visual':return observation(processor,source)
    from PIL import Image
    prompt=processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=source['reference'])])],
        tokenize=False,add_generation_prompt=False)
    with Image.open(source['image']) as im:
        return processor(text=prompt,images=[im.convert('RGB')],return_tensors='pt',truncation=False).to('cuda')


def write(args):
    import torch
    from safetensors import safe_open
    model,processor=load(args.model);archive=model.open_physical_archive(args.work/'archive')
    if archive.concept_lens is None:raise RuntimeError('native storage is not the checkpoint default')
    versions={n:p._version for n,p in model.named_parameters()};records=[];original=model.generate
    def forbidden(*a,**kw):raise RuntimeError('writer attempted to generate labels')
    model.generate=forbidden
    try:
        for source in json.loads((args.work/'sources.json').read_text()):
            inputs=source_inputs(processor,source)
            paced(archive.session.observe,**inputs,use_cache=False,logits_to_keep=1)
            if source['kind']=='text':paced(archive.session.learn,**inputs,labels=inputs['input_ids'].clone(),learning_rate=.001)
            saved=paced(archive.append)
            with safe_open(saved['unit_path'],framework='pt') as f:
                meta=json.loads(f.metadata()['physical_unit'])
                if meta['version']!=6 or meta['address']['text_labels_required']:raise RuntimeError('default storage did not switch')
            records.append(dict(page=source['id'],**saved,format=meta['version']))
            dump(args.work/'pages.json',records);log('native_default_saved',**records[-1])
    finally:model.generate=original
    # Probe questions are opened only after all writing and address extraction.
    by_page={p['page']:p for p in records};hot=[]
    for q in json.loads((args.work/'questions.json').read_text()):
        archive.mount_hash_async(by_page[q['page']]['memory_hash']).result()
        answer=generated(processor,q['question'],archive.session.generate)
        hot.append(measured(q,answer,'oracle-hot'));dump(args.work/'hot.json',hot)
    fixed=all(p._version==versions[n] for n,p in model.named_parameters())
    if not fixed:raise RuntimeError('body changed during writes')
    dump(args.work/'write-result.json',dict(units=len(records),all_v6=True,body_fixed=fixed,
        text_generation_during_write=0,peak_vram_gib=torch.cuda.max_memory_allocated()/2**30))
    archive.close()


def read(args):
    import torch
    model,processor=load(args.model);archive=model.open_physical_archive(args.work/'archive')
    versions={n:p._version for n,p in model.named_parameters()};rows=[];routes=[]
    pages=json.loads((args.work/'pages.json').read_text());by_page={p['page']:p for p in pages}
    questions=json.loads((args.work/'questions.json').read_text())
    def record(q,answer,condition,**extra):
        row=measured(q,answer,condition,**extra);rows.append(row)
        dump(args.work/'answers.json',rows);log('native_storage_recall',**row)
    for q in questions:
        inputs=question_inputs(processor,q['question']);keys=paced(archive.encode_query,address_inputs(processor,q['question']))
        cpu=archive.query({},encoded_query=keys,top_k=3,device='cpu')
        gpu=archive.query({},encoded_query=keys,top_k=3,device='cuda')
        same=[r['unit_id'] for r in cpu['matches']]==[r['unit_id'] for r in gpu['matches']]
        routes.append(dict(id=q['id'],expected=by_page[q['page']]['unit_id'],native=gpu['matches'],cpu_gpu_same=same))
        dump(args.work/'routes.json',routes)
        result=paced(archive.generate,{},encoded_query=keys,generation_inputs=dict(inputs),top_k=1,
            index_options=dict(device='cuda'),max_new_tokens=64,do_sample=False)
        tokens=result['tokens'][0,inputs['input_ids'].shape[1]:]
        record(q,dict(answer=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip(),generated_tokens=len(tokens),hit_generation_limit=len(tokens)==64),
            'native-default',selected=result['loaded_units'])
        archive.mount_hash_async(by_page[q['page']]['memory_hash']).result()
        record(q,generated(processor,q['question'],archive.session.generate),'oracle-cold')
        wrong=next(p for p in pages if p['page']!=q['page'])
        archive.mount_hash_async(wrong['memory_hash']).result()
        record(q,generated(processor,q['question'],archive.session.generate),'wrong-memory')
        record(q,generated(processor,q['question'],model.generate,use_memory=False),'empty')
    # Source-visible controls cannot feed information into the earlier routes.
    sources={s['id']:s for s in json.loads((args.work/'sources.json').read_text())}
    for q in questions:
        source=source_inputs(processor,sources[q['page']]);query=question_inputs(processor,q['question'])
        ids=torch.cat((source['input_ids'],query['input_ids']),dim=1)
        output=paced(model.generate,**dict(source,input_ids=ids,attention_mask=torch.ones_like(ids)),
            use_memory=False,max_new_tokens=64,do_sample=False)
        tokens=output[0,ids.shape[1]:]
        record(q,dict(answer=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip(),generated_tokens=len(tokens),hit_generation_limit=len(tokens)==64),'source-visible')
    scores={c:dict(n=len([r for r in rows if r['condition']==c]),
        exact=sum(r['exact'] for r in rows if r['condition']==c),
        content=sum(r['content_correct'] for r in rows if r['condition']==c)) for c in dict.fromkeys(r['condition'] for r in rows)}
    native={r['id']:r for r in rows if r['condition']=='native-default'}
    oracle={r['id']:r for r in rows if r['condition']=='oracle-cold'}
    hot={r['id']:r for r in json.loads((args.work/'hot.json').read_text())}
    lost=[q for q,r in oracle.items() if r['content_correct'] and not native[q]['content_correct']]
    same=sum(oracle[k]['answer']==r['answer'] for k,r in hot.items())
    fixed=all(p._version==versions[n] for n,p in model.named_parameters())
    gates=dict(all_v6=all(p['format']==6 for p in pages),body_fixed=fixed,
        cpu_gpu_same=all(r['cpu_gpu_same'] for r in routes),hot_cold_same=same==len(questions),
        no_oracle_recall_lost=not lost,
        recall_exceeds_negative_controls=scores['native-default']['content']>
            max(scores['wrong-memory']['content'],scores['empty']['content']),
        text_hash_and_visual_recalled=all(any(r['content_correct'] and r['kind']==kind
            for r in native.values()) for kind in ('text','hash','visual')))
    result=dict(scores=scores,gates=gates,passed=all(gates.values()),lost_oracle_answers=lost,
        top1_native=sum(r['native'][0]['unit_id']==r['expected'] for r in routes),
        questions=len(questions),
        hot_cold_identical=same,peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
    dump(args.work/'result.json',result);log('native_storage_verification_complete',**result)
    archive.close()
    if not result['passed'] and not getattr(args,'report_only',False):
        raise RuntimeError('unified memory acceptance gates failed; retain full results')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','write','read'])
    p.add_argument('--model',type=Path);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--report-only',action='store_true',help='Retain failed functional scores and continue other diagnostics; does not mark the result passed')
    a=p.parse_args();gate=os.environ.get('LFM2_START_GATE')
    while gate and not Path(gate).exists():time.sleep(.1)
    if a.mode=='prepare':prepare(a.work)
    else:globals()[a.mode](a)
