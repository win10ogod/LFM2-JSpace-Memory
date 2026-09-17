"""Render completed diagnostic artifacts; never runs or changes the model."""
from pathlib import Path
import argparse,json


def aggregate(rows):
    clean={r['id']:r for r in rows if r['condition']=='clean'}
    result={}
    for condition in dict.fromkeys(r['condition'] for r in rows):
        group=[r for r in rows if r['condition']==condition]
        eligible=[r for r in group if clean[r['id']]['original_prefix']]
        result[condition]=dict(n=len(group),original=sum(r['original_prefix'] for r in group),
            counterfactual=sum(r['counterfactual_prefix'] for r in group),
            clean_eligible=len(eligible),eligible_counterfactual=sum(r['counterfactual_prefix'] for r in eligible),
            mean_delta_log_odds=sum(r['counterfactual_log_odds']-clean[r['id']]['counterfactual_log_odds'] for r in group)/len(group),
            tasks={task:dict(n=len([r for r in group if r['task']==task]),
                counterfactual=sum(r['counterfactual_prefix'] for r in group if r['task']==task))
                for task in dict.fromkeys(r['task'] for r in group)})
    return result


def build(root):
    phases=[]
    for name,title in [('jspace-functions-v1','探索：層位與介入'),
                       ('jspace-validation-v1','新概念：同層對照'),
                       ('jspace-privilege-v1','表徵分解：J-component 與 remainder'),
                       ('jspace-transfer-v1','原生對話與固定目標座標')]:
        directory=root/name
        if name=='jspace-transfer-v1' and not directory.exists():continue
        if not (directory/'result.json').exists():raise RuntimeError('Incomplete experiment: '+name)
        rows=json.loads((directory/'trials.json').read_text())
        result=json.loads((directory/'result.json').read_text())
        phases.append(dict(name=name,title=title,rows=rows,result=result,summary=aggregate(rows)))
    lens=json.loads((root/'jacobian-v2/result.json').read_text())
    storage=root/'native-storage-acceptance-v1/result.json';deployment=root/'jspace-deployment.json'
    summary=dict(checkpoint=lens['protocol']['checkpoint_id'],lens=lens['protocol'],
        split_stability=lens['split_stability'],phases=[{k:v for k,v in p.items() if k!='rows'} for p in phases],
        memory_storage_switched=deployment.exists(),global_workspace_established=False,
        native_storage=json.loads(storage.read_text()) if storage.exists() else None,
        deployment=json.loads(deployment.read_text()) if deployment.exists() else None,
        scope='measured functional evidence; no universal claim from sparse decomposition alone')
    (root/'jspace-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    payload=json.dumps(dict(summary=summary,phases=phases),ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>LFM2.5 J-space 功能檢測</title>
<style>body{font:16px system-ui;margin:24px;background:#101827;color:#e2e8f0;line-height:1.5}
a{color:#93c5fd}button,select{padding:8px;background:#263449;color:inherit;border:1px solid #64748b}
table{border-collapse:collapse}td,th{padding:8px;border:1px solid #475569;white-space:nowrap}
td{cursor:pointer}td:hover{outline:2px solid #f8fafc}.scroll{overflow:auto}pre{white-space:pre-wrap}
.panel{background:#182438;padding:16px;margin:18px 0}small{color:#cbd5e1}</style>
<h1>LFM2.5：J-space 探測與功能性介入</h1>
<p>此頁呈現已完成的實驗。J-space 的稀疏非負構造與 global-workspace 功能證據分別評估。
測試使用本機已合併 LoRA 的 LFM2.5-VL-3B 衍生檢查點。</p>
<p id="storage-status"></p><details><summary>記憶保存驗收與模型位置</summary><pre id="storage"></pre></details>
<p><a href="jacobian-v2/lens.html">逐層／逐 token 的 Jacobian 讀出</a> ·
<a href="jspace-summary.json">完整摘要 JSON</a></p><details><summary>校準與權重身分</summary><pre id="protocol"></pre></details>
<div class="panel"><label>實驗 <select id="phase"></select></label>
<p id="scope"></p><div class="scroll"><table id="summary"></table></div></div>
<h2>逐題介入</h2><p>格子顯示相對 clean 的反事實答案 log odds 變化；綠色代表往目標方向移動，
不等同完整回答正確。✓ 代表自由生成以目標答案開頭。點選查看原始輸出、擾動與評分。</p>
<div class="scroll"><table id="grid"></table></div><pre class="panel" id="detail"></pre>
<script>const data=PAYLOAD;const picker=document.getElementById('phase');
data.phases.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=p.title;picker.append(o)});picker.value=data.phases.length-1;
document.getElementById('protocol').textContent=JSON.stringify({checkpoint:data.summary.checkpoint,lens:data.summary.lens,stability:data.summary.split_stability},null,2);
document.getElementById('storage-status').textContent=data.summary.memory_storage_switched?'已發布預設使用原生概念地址的模型；舊模型與舊記憶保留。':'記憶切換尚待驗收與發布。';
document.getElementById('storage').textContent=JSON.stringify({acceptance:data.summary.native_storage,deployment:data.summary.deployment},null,2);
function cell(row,text,tag='td'){const c=document.createElement(tag);c.textContent=text;row.append(c);return c}
function show(){const p=data.phases[Number(picker.value)];
document.getElementById('scope').textContent=p.rows.length+' 次介入；完整 global-workspace 證據尚未由此表自動判定。';
const s=document.getElementById('summary');s.replaceChildren();const h=document.createElement('tr');
['條件','原答案／全部','目標答案／全部','目標／clean 可答題','平均 Δ log odds'].forEach(t=>cell(h,t,'th'));s.append(h);
Object.entries(p.summary).forEach(([name,v])=>{const r=document.createElement('tr');
[name,v.original+'/'+v.n,v.counterfactual+'/'+v.n,v.eligible_counterfactual+'/'+v.clean_eligible,v.mean_delta_log_odds.toFixed(3)].forEach(t=>cell(r,t));s.append(r)});
const g=document.getElementById('grid');g.replaceChildren();const conditions=Object.keys(p.summary),ids=[...new Set(p.rows.map(r=>r.id))];
const head=document.createElement('tr');cell(head,'問題','th');conditions.forEach(t=>cell(head,t,'th'));g.append(head);
ids.forEach(id=>{const r=document.createElement('tr');cell(r,id,'th');const clean=p.rows.find(v=>v.id===id&&v.condition==='clean');
conditions.forEach(c=>{const v=p.rows.find(v=>v.id===id&&v.condition===c);const d=v.counterfactual_log_odds-clean.counterfactual_log_odds;
const td=cell(r,(v.counterfactual_prefix?'✓ ':'')+d.toFixed(2));td.style.background=d>0?'rgba(22,163,74,'+Math.min(.65,d/12)+')':'rgba(220,38,38,'+Math.min(.65,-d/12)+')';
td.onclick=()=>document.getElementById('detail').textContent=JSON.stringify(v,null,2)});g.append(r)});
document.getElementById('detail').textContent='點選逐題格子查看實際回答。'}picker.onchange=show;show();</script></html>'''
    (root/'jspace-report.html').write_text(page.replace('PAYLOAD',payload),encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    summary=build(args.root)
    print(json.dumps({p['name']:p['summary'] for p in summary['phases']},ensure_ascii=False))
