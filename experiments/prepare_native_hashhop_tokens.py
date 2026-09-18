"""Native LlamaFactory preprocessing of complete two-turn memory episodes."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ['DISABLE_VERSION_CHECK']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path.insert(0,'/mnt/f/稠密轉MOE試驗/LlamaFactory/src')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    import torch
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    from llamafactory.data import get_dataset,get_template_and_fix_tokenizer
    from lfm2_titans.memory_recall_training import split_episodes
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);a.output.mkdir(parents=True,exist_ok=True)
    args=dict(model_name_or_path=str(a.model),trust_remote_code=True,stage='sft',do_train=True,do_eval=(a.data/'validation.jsonl').exists(),
        eval_dataset='hashhop_validation' if (a.data/'validation.jsonl').exists() else None,
        finetuning_type='lora',lora_rank=8,lora_alpha=16,template='lfm2_vl',dataset='hashhop_curriculum',
        dataset_dir=str(a.data),tokenized_path=str(a.output/'tokenized'),cutoff_len=4096,mask_history=True,default_system='',
        preprocessing_num_workers=2,preprocessing_batch_size=32,output_dir=str(a.output/'trainer'),
        use_cpu=True,report_to=[],seed=41)
    model_args,data_args,training_args,_,_=get_train_args(args)
    module=load_tokenizer(model_args);tokenizer=module['tokenizer']
    template=get_template_and_fix_tokenizer(tokenizer,data_args)
    datasets=get_dataset(template,model_args,data_args,training_args,stage='sft',**module)
    data=datasets['train_dataset']
    original=[json.loads(line) for line in (a.data/'train.jsonl').read_text().splitlines()]
    if len(data)!=len(original):raise RuntimeError('Native preprocessing changed the number of memory episodes')
    spec=dict(user_header_ids=tokenizer.encode('<|im_start|>user\n',add_special_tokens=False),
        turn_end_id=tokenizer.convert_tokens_to_ids('<|im_end|>'),write_learning_rate=.001,
        objective_version=2,compressed_recall_weight=.25,reconstruction_weight=.05)
    def verify_rows(data,original):
        if len(data)!=len(original):raise RuntimeError('Native preprocessing changed row count')
        source_tokens=targets=0
        for i,row in enumerate(data):
            ids=torch.tensor([row['input_ids']]);labels=torch.tensor([row['labels']]);valid=torch.tensor([row['attention_mask']])
            native_ids=tokenizer.apply_chat_template(original[i]['messages'],tokenize=True,
                add_generation_prompt=False,return_dict=True)['input_ids']
            if row['input_ids']!=native_ids:raise RuntimeError('Factory/native chat IDs differ')
            episode=split_episodes(ids,labels,valid,spec)[0]
            source=tokenizer.decode(episode['source'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
            if source!=original[i]['messages'][0]['content']:raise RuntimeError('Observation changed or truncated')
            answer=tokenizer.decode(labels[labels!=-100],skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
            if answer!=original[i]['messages'][-1]['content']:raise RuntimeError('Assistant target changed or truncated')
            source_tokens+=len(episode['source']);targets+=int((labels!=-100).sum())
        return source_tokens,targets
    source_tokens,targets=verify_rows(data,original)
    validation=datasets.get('eval_dataset');validation_receipt=None
    if validation is not None:
        originals=[json.loads(line) for line in (a.data/'validation.jsonl').read_text().splitlines()]
        vt,vl=verify_rows(validation,originals)
        validation_receipt=dict(records=len(validation),source_tokens=vt,assistant_targets=vl)
    receipt=dict(status='passed',records=len(data),validation=validation_receipt,source_tokens=source_tokens,assistant_targets=targets,
        max_input_tokens=max(map(len,data['input_ids'])),mask_history=True,custom_sft_chunks=False,
        full_source_and_answer_preserved=True,native_chat_ids_identical=True,episode_spec=spec,columns=data.column_names,
        loader=get_dataset.__module__+'.'+get_dataset.__name__)
    (a.output/'preprocessing.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
