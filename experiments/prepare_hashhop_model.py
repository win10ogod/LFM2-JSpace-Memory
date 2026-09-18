"""Inherit a validated SFT checkpoint for a small memory-dependent continuation."""
import argparse
import importlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from research_common import load,dump,log
from lfm2_export_artifacts import sync_runtime


def parameter_digest(parameter):
    """Hash actual tensor bytes, including BF16, without retaining a model copy."""
    import torch
    data=parameter.detach().reshape(-1).to('cpu').contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()


def capture_parameters(model):
    return [(name,p,p._version,str(p.dtype),tuple(p.shape),parameter_digest(p))
            for name,p in model.named_parameters()]


def verify_inherited_parameters(model,snapshot):
    """Parametrization registration can change _version without changing values."""
    import torch
    present={id(p) for p in model.parameters()};rows=[];failed=[]
    for name,p,version,dtype,shape,checksum in snapshot:
        unchanged=(id(p) in present and str(p.dtype)==dtype and tuple(p.shape)==shape
                   and parameter_digest(p)==checksum)
        rows.append(dict(name=name,sha256=checksum,dtype=dtype,shape=shape,
                         version_before=version,version_after=p._version,unchanged=unchanged))
        if not unchanged:failed.append(name)
    # A preserved base is insufficient if an added adapter has a nonzero output.
    outputs=[(n,p) for n,p in model.named_parameters()
             if n.endswith('.adapter_B') or '.lora_B.' in n]
    nonzero=[n for n,p in outputs if bool(torch.count_nonzero(p.detach()))]
    result=dict(status='passed' if not failed and not nonzero and outputs else 'failed',
                inherited_tensors=len(rows),version_changes_without_value_changes=sum(
                    r['unchanged'] and r['version_before']!=r['version_after'] for r in rows),
                zero_adapter_outputs=len(outputs)-len(nonzero),changed_parameters=failed,
                nonzero_adapter_outputs=nonzero,parameters=rows)
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ['base','work','data','preprocessing']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--scope',choices=['memory','all'],default='memory');a=parser.parse_args()
    gate=os.environ.get('LFM2_START_GATE')
    while gate and not Path(gate).exists():time.sleep(.1)
    prep=json.loads((a.preprocessing/'preprocessing.json').read_text())
    if prep['status']!='passed' or not prep['full_source_and_answer_preserved']:raise RuntimeError('Incomplete native episode verification')
    target=a.work/'model';adapter=a.work/'adapter-init'
    if target.exists() or adapter.exists():raise FileExistsError('Continuation staging already exists')
    target.mkdir(parents=True)
    config=json.loads((a.base/'config.json').read_text())
    shards=set(json.loads((a.base/'model.safetensors.index.json').read_text())['weight_map'].values())
    for name in shards|{'model.safetensors.index.json','memory_graph.safetensors','generation_config.json',
        'processor_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'}:
        if name in shards:os.link(a.base/name,target/name)
        else:shutil.copy2(a.base/name,target/name)
    config.update(native_joint_sft=True,native_sft_rest_ratio=1.,
        native_memory_recall=dict(prep['episode_spec'],train_scope=a.scope),concept_memory=None)
    dump(target/'config.json',config)
    root=Path(__file__).resolve().parents[1];runtime=sync_runtime(root/'src/lfm2_titans',target)
    native,processor=load(target)
    log('hashhop_inheritance_snapshot')
    original=capture_parameters(native)
    helper=importlib.import_module(type(native).__module__.rsplit('.',1)[0]+'.sft_lora')
    sys.path.insert(0,'/mnt/f/稠密轉MOE試驗/LlamaFactory/src');os.environ['DISABLE_VERSION_CHECK']='1'
    from llamafactory.hparams import FinetuningArguments,ModelArguments
    from llamafactory.model.adapter import init_adapter
    factory=SimpleNamespace(FinetuningArguments=FinetuningArguments,ModelArguments=ModelArguments,init_adapter=init_adapter,WORK=a.work)
    model,_,_,_,audit=helper.apply_sft_lora(native,factory,rank=8,memory_edge_rank=8)
    helper.apply_training_scope(native)
    inheritance=verify_inherited_parameters(native,original)
    dump(a.work/'adapter-inheritance.json',inheritance)
    if inheritance['status']!='passed':raise RuntimeError('Adapter initialization changed inherited values; see adapter-inheritance.json')
    log('hashhop_inheritance_verified',**{k:v for k,v in inheritance.items() if k!='parameters'})
    if a.scope=='memory' and any(p.requires_grad for p in native.model.parameters()):raise RuntimeError('Native body was not frozen')
    model.peft_config['default'].base_model_name_or_path=str(target)
    model.save_pretrained(adapter,safe_serialization=True);helper.save_custom_lora(model,adapter)
    trainable={}
    for name,p in native.named_parameters():
        if p.requires_grad:
            group=name.split('.')[0];trainable[group]=trainable.get(group,0)+p.numel()
    manifest=dict(expected_steps=32,records=256,method='native LlamaFactory memory-dependent HashHop SFT',
        native_lora_rank=8,native_lora_alpha=16,train_scope=a.scope,physical_batch=8,cutoff_len=4096,
        mask_history=True,training_chunks=0,source_only_writes=True,source_kv_reused=False,
        inherited_weights_unchanged=True,trainable_by_group=trainable,source_checkpoint_id=config['memory_checkpoint_id'],
        reference='https://magic.dev/blog/100m-token-context-windows')
    dump(a.work/'training-manifest.json',manifest)
    dump(a.work/'model-preparation.json',dict(**manifest,runtime=runtime,adapter=str(adapter),model=str(target)))
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
