"""Independent full-matrix Jacobian lens for the native LFM language stack.

Estimator follows the *method* documented by Anthropic (Apache-2.0):
https://github.com/anthropics/jacobian-lens/blob/main/jlens/fitting.py
Sum cotangents over valid target positions, average gradients over valid
source positions, then average matrices over prompts. Causality eliminates
backward-in-time terms. This is not a per-position diagonal Jacobian.

No optimizer, memory write, concept-address dependency, or inference patch.
An estimated lens alone does not establish global-workspace properties.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import time

import torch


@contextmanager
def record_layers(layers, selected, *, start_graph=False):
    activations = {}
    handles = []
    first = min(selected)
    def hook(index):
        def capture(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError('expected a tensor from a native LFM decoder layer')
            if start_graph and index == first and not output.requires_grad:
                output.requires_grad_(True)
            activations[index] = output
        return capture
    try:
        for index in selected:
            handles.append(layers[index].register_forward_hook(hook(index)))
        yield activations
    finally:
        for handle in handles:
            handle.remove()


def jacobian_for_inputs(language_model, input_ids, source_layers, *, dim_batch=4,
                        skip_first=16, rest_ratio=0., progress=None):
    """Return complete d×d matrices, without truncating any input token.

    Runs on a frozen, evaluation-mode stack. Each replica carries a distinct
    output basis cotangent, avoiding vmap requirements in FlashAttention-2.
    """
    if language_model.training or any(p.requires_grad for p in language_model.parameters()):
        raise ValueError('Jacobian diagnostics require an eval-mode frozen model')
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError('provide one complete, unpadded calibration sequence')
    layers = language_model.layers
    target = len(layers)-1
    source_layers = sorted(set(source_layers))
    if not source_layers or min(source_layers)<0 or max(source_layers)>=target:
        raise ValueError('source layers must precede the final layer')
    if dim_batch<1 or skip_first<0 or not math.isfinite(rest_ratio) or rest_ratio<0:
        raise ValueError('invalid diagnostic scheduling')
    length = input_ids.shape[1]
    if length<=skip_first+1:
        raise ValueError('calibration sequence has no valid positions')
    selected = [*source_layers, target]
    device = input_ids.device
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type=='cuda' else (lambda: None)
    with torch.enable_grad(), record_layers(layers, selected, start_graph=True) as values:
        language_model(input_ids=input_ids.expand(dim_batch,-1), use_cache=False)
        output = values[target]
        width = output.shape[-1]
        matrices = {layer:torch.empty(width,width,dtype=torch.float32) for layer in source_layers}
        sources = [values[layer] for layer in source_layers]
        valid = torch.arange(skip_first,length-1,device=device)
        cotangent = torch.zeros_like(output)
        for start in range(0,width,dim_batch):
            synchronize(); tick=time.monotonic()
            count = min(dim_batch,width-start)
            batch = torch.arange(count,device=device)
            cotangent.zero_()
            cotangent[batch[:,None],valid[None,:],(start+batch)[:,None]]=1
            gradients = torch.autograd.grad(output,sources,grad_outputs=cotangent,
                retain_graph=start+count<width)
            for layer,gradient in zip(source_layers,gradients):
                rows = gradient[:count,valid].float().mean(dim=1).cpu()
                if not torch.isfinite(rows).all():
                    raise FloatingPointError('nonfinite Jacobian; refusing to publish lens')
                matrices[layer][start:start+count] = rows
            del gradients
            synchronize(); elapsed=time.monotonic()-tick
            if progress and (start//dim_batch%32==0 or start+count==width):
                progress(rows=start+count,total=width,seconds_per_batch=elapsed)
            if rest_ratio:
                time.sleep(elapsed*rest_ratio)
    return matrices


@torch.no_grad()
def inspect_prompt(model, tokenizer, prompt, jacobians, *, top_k=5):
    language_model=model.model.language_model
    device=next(language_model.parameters()).device
    ids=tokenizer(prompt,return_tensors='pt',add_special_tokens=True)['input_ids'].to(device)
    with record_layers(language_model.layers,sorted(jacobians)) as activations:
        native=language_model(input_ids=ids,use_cache=False).last_hidden_state
    native_logits=model.lm_head(native)[0].float()
    result=dict(prompt=prompt,tokens=[tokenizer.decode([i]) for i in ids[0].tolist()],layers=[],metrics=[])
    for layer,matrix in sorted(jacobians.items()):
        hidden=activations[layer][0]
        projected=hidden.float()@matrix.to(device).T
        logits=model.lm_head(language_model.embedding_norm(projected.to(hidden.dtype))).float()
        ordinary=model.lm_head(language_model.embedding_norm(hidden)).float()
        targets=ids[0,1:]
        def nll(value):
            return float(torch.nn.functional.cross_entropy(value[:-1],targets))
        result['metrics'].append(dict(layer=layer,jacobian_next_token_nll=nll(logits),
            identity_lens_next_token_nll=nll(ordinary),native_next_token_nll=nll(native_logits),
            top1_agreement=float((logits.argmax(-1)==native_logits.argmax(-1)).float().mean())))
        probabilities=logits.softmax(-1)
        scores,indices=probabilities.topk(top_k,dim=-1)
        cells=[[dict(token=tokenizer.decode([int(token)]),probability=float(score))
                for token,score in zip(row,values)] for row,values in zip(indices,scores)]
        result['layers'].append(dict(layer=layer,kind=language_model.config.layer_types[layer],cells=cells))
    scores,indices=native_logits.softmax(-1).topk(top_k,dim=-1)
    result['native']=[[dict(token=tokenizer.decode([int(token)]),probability=float(score))
                       for token,score in zip(row,values)] for row,values in zip(indices,scores)]
    return result


def render(path, report):
    payload=json.dumps(report,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<title>LFM2 Jacobian 診斷</title><style>
body{font:16px system-ui;margin:24px;background:#101827;color:#e2e8f0}a{color:#93c5fd}
select,button{padding:8px;background:#253248;color:inherit;border:1px solid #627188}
table{border-collapse:collapse}td,th{border:1px solid #475569;padding:8px;min-width:70px}
td{cursor:pointer}td:hover{background:#334155}.scroll{overflow:auto}pre{white-space:pre-wrap}
</style><h1>LFM2 原生層 Jacobian 檢測</h1>
<p>完整矩陣、小樣本校準。此頁僅呈現診斷，不代表已證明 J-space 的因果工作空間性質。
記憶寫入、索引與概念生成不依賴此工具。</p>
<details><summary>校準參數與 checkpoint 身分</summary><pre id="scope"></pre></details>
<label>檢視提示 <select id="choice"></select></label><pre id="prompt"></pre>
<div class="scroll"><table id="grid"></table></div>
<h2>點選格子查看該位置的詞彙讀出</h2><pre id="detail"></pre><h2>數值對照</h2><pre id="metrics"></pre>
<script>const data=PAYLOAD;
document.getElementById('scope').textContent=JSON.stringify(data.protocol,null,2);
const choice=document.getElementById('choice');data.probes.forEach((p,i)=>{
const o=document.createElement('option');o.value=i;o.textContent=p.prompt;choice.appendChild(o)});
function show(){const p=data.probes[Number(choice.value)],g=document.getElementById('grid');g.replaceChildren();
document.getElementById('prompt').textContent=p.prompt;
function row(label,cells,header=false){const r=document.createElement('tr');
const h=document.createElement('th');h.textContent=label;r.appendChild(h);
cells.forEach((c,i)=>{const t=document.createElement(header?'th':'td');
t.textContent=header?c:c[0].token;t.onclick=()=>document.getElementById('detail').textContent=
JSON.stringify({layer:label,position:i,source_token:p.tokens[i],readout:c},null,2);r.appendChild(t)});g.appendChild(r)}
row('輸入',p.tokens,true);p.layers.forEach(l=>row('層 '+l.layer+' / '+l.kind,l.cells));row('原生輸出',p.native);
document.getElementById('metrics').textContent=JSON.stringify(p.metrics,null,2)}
choice.onchange=show;show();</script></html>'''
    path.write_text(page.replace('PAYLOAD',payload),encoding='utf-8')


def main():
    from research_common import load,dump,log,paced
    from safetensors.torch import save_file
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--dim-batch',type=int,default=4)
    parser.add_argument('--layers',type=int,nargs='+')
    args=parser.parse_args()
    gate=os.environ.get('LFM2_START_GATE')
    while gate and not Path(gate).exists():time.sleep(.1)
    protocol=json.loads(args.protocol.read_text())
    prompts=protocol['calibration'];probes=protocol['probes']
    if len(prompts)<2 or not probes:raise ValueError('calibration and separate probes required')
    args.out.mkdir(exist_ok=False)
    model,processor=load(args.model);tokenizer=processor.tokenizer
    language_model=model.model.language_model
    versions={n:p._version for n,p in model.named_parameters()}
    layers=sorted(args.layers or [int(depth) for depth in model.config.language_ports])
    device=next(language_model.parameters()).device
    inputs=[tokenizer(p,return_tensors='pt')['input_ids'].to(device) for p in prompts]
    if any(ids.shape[1]>model.config.text_config.max_position_embeddings for ids in inputs):
        raise ValueError('calibration input exceeds native context; no implicit truncation')
    with torch.no_grad():reference=language_model(input_ids=inputs[0],use_cache=False).last_hidden_state.clone()
    total={};halves=[{},{}];counts=[0,0];convergence=[]
    for index,ids in enumerate(inputs):
        tick=time.monotonic()
        matrices=jacobian_for_inputs(language_model,ids,layers,dim_batch=args.dim_batch,
            skip_first=16,rest_ratio=1.,progress=lambda **v:log('jacobian_rows',prompt=index,**v))
        changes={}
        side=int(index>=len(inputs)//2);counts[side]+=1
        for layer,matrix in matrices.items():
            if layer in total:
                old=total[layer]/index
                changes[str(layer)]=float((matrix-old).norm()/((index+1)*old.norm().clamp_min(1e-12)))
                total[layer]+=matrix
            else:total[layer]=matrix.clone()
            halves[side][layer]=halves[side].get(layer,torch.zeros_like(matrix))+matrix
        save_file({str(l):(m/(index+1)).contiguous() for l,m in total.items()},str(args.out/'jacobians.safetensors'))
        convergence.append(dict(prompt=index,tokens=ids.shape[1],seconds=time.monotonic()-tick,mean_relative_change=changes))
        dump(args.out/'progress.json',convergence);log('jacobian_prompt_complete',**convergence[-1])
    means={layer:matrix/len(inputs) for layer,matrix in total.items()}
    stability={}
    for layer in layers:
        a=halves[0][layer]/counts[0];b=halves[1][layer]/counts[1]
        stability[str(layer)]=dict(split_cosine=float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0)),
            split_relative_difference=float((a-b).norm()/((a.norm()+b.norm())/2).clamp_min(1e-12)))
    probe_results=[paced(inspect_prompt,model,tokenizer,p,means) for p in probes]
    with torch.no_grad():after=language_model(input_ids=inputs[0],use_cache=False).last_hidden_state
    fixed=all(p._version==versions[n] and p.grad is None and not p.requires_grad for n,p in model.named_parameters())
    same=torch.equal(reference,after)
    if not fixed or not same:raise RuntimeError('diagnostic changed native weights or output')
    report=dict(protocol=dict(checkpoint_id=model.config.memory_checkpoint_id,
        calibration_prompts=len(prompts),calibration_tokens=[v.shape[1] for v in inputs],
        calibration_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        full_matrix=True,hidden_width=language_model.config.hidden_size,source_layers=layers,
        target_layer=len(language_model.layers)-1,skip_first=16,exclude_last=True,
        estimator='sum over valid causal targets; mean over valid sources; mean over prompts',
        scope='native frozen language stack; memory inactive; diagnostic only',
        calibration_scope='small authored calibration set; not a representative pretraining corpus',
        attention=language_model.config._attn_implementation,rest_ratio=1.),
        convergence=convergence,split_stability=stability,probes=probe_results,
        body_fixed=fixed,native_outputs_identical=same,J_space_established=False,
        peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
    dump(args.out/'result.json',report);render(args.out/'lens.html',report)
    log('jacobian_diagnostic_complete',body_fixed=fixed,native_outputs_identical=same,
        split_stability=stability,peak_vram_gib=report['peak_vram_gib'])


if __name__=='__main__':main()
