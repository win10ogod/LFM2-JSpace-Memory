"""CPU codec-only objective ablation; never saves or publishes model weights.

All arms start at the same real native-input VAE, use LoRA 8/16, identical
source positions, batches and optimizer. No query or answer enters this probe.
This diagnoses feature preservation, not end-to-end recall or consolidation.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from transformers import AutoProcessor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lfm2_titans.dream_memory import DreamWeightVAE
from lfm2_titans.sft_lora import WeightLoRALinear
from research_common import dump,digest


def discrimination(decoded,target,ids,temperature=.1):
    """Repeated observed token identities are positives, never false negatives."""
    logits=F.normalize(decoded,dim=-1)@F.normalize(target.detach(),dim=-1).T/temperature
    positives=ids[:,None]==ids[None,:]
    return (logits.logsumexp(-1)-logits.masked_fill(~positives,-torch.inf).logsumexp(-1)).mean()


def main():
    p=argparse.ArgumentParser()
    for name in ('model','train','validation','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=1024)
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--token-reconstruction',action='store_true')
    p.add_argument('--adapter',type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    config=json.loads((a.model/'config.json').read_text())
    mapping=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
    prefix='dream_memory.feature_vae.heads.native_input.'
    names=[n for n in mapping if n.startswith(prefix)]+['model.language_model.embed_tokens.weight']
    with ExitStack() as stack:
        readers={f:stack.enter_context(safe_open(str(a.model/f),framework='pt',device='cpu')) for f in {mapping[n] for n in names}}
        spec=config['dream_memory']['feature_vae']
        original=DreamWeightVAE(config['text_config']['hidden_size'],spec['hidden_size'],spec['latent_size']).eval()
        original.load_state_dict({n.removeprefix(prefix):readers[mapping[n]].get_tensor(n) for n in names if n.startswith(prefix)})
        if a.adapter:
            with safe_open(str(a.adapter),framework='pt',device='cpu') as trained,torch.no_grad():
                for n,module in original.named_modules():
                    if isinstance(module,nn.Linear):
                        module.weight.add_((trained.get_tensor(prefix+n+'.adapter_B')@
                                            trained.get_tensor(prefix+n+'.adapter_A'))*2.)
        embedding=readers[mapping[names[-1]]].get_tensor(names[-1])
        processor=AutoProcessor.from_pretrained(a.model,local_files_only=True,trust_remote_code=True)

        def observations(path,records,positions):
            texts=[]
            with path.open() as f:
                for i,line in enumerate(f):
                    if i>=records:break
                    texts.append(json.loads(line)['messages'][0]['content'])
            ids=torch.cat([processor(text=t,return_tensors='pt',truncation=False)['input_ids'][0] for t in texts])
            indices=torch.linspace(0,len(ids)-1,min(positions,len(ids))).long()
            ids=ids[indices];raw=F.embedding(ids,embedding).float()
            mean=raw.mean(-1,keepdim=True);scale=(raw.var(-1,unbiased=False,keepdim=True)+1e-5).sqrt()
            return ids,(raw-mean)/scale,mean,scale

        train_ids,train,train_mean,train_scale=observations(a.train,64,4096)
        val_ids,val,val_mean,val_scale=observations(a.validation,128,2048)
        # Whole vocabulary for the bounded identity test; this is not a model
        # recall implementation and does not use evaluation answers.
        vocabulary=F.normalize(embedding.float(),dim=-1)
        generator=torch.Generator().manual_seed(711)
        vocabulary_ids=torch.unique(torch.cat((train_ids,torch.randint(len(vocabulary),(4096,),generator=generator))))
        sampled_vocabulary=vocabulary[vocabulary_ids]

        @torch.no_grad()
        def evaluate(head):
            head.eval();decoded,mu,logvar=head(val,sample=False)
            mse=float(F.mse_loss(decoded,val));cos=float(F.cosine_similarity(decoded,val).mean())
            selected=torch.linspace(0,len(val)-1,min(256,len(val))).long()
            restored=decoded[selected]*val_scale[selected]+val_mean[selected]
            similarities=F.normalize(restored,dim=-1)@vocabulary.T
            predicted=similarities.argmax(-1)
            return dict(mean_reconstruction_mse=mse,zero_decoder_mse=float(val.square().mean()),
                normalized_cosine=cos,kl_per_coordinate=float(.5*(mu.square()+logvar.exp()-1-logvar).mean()),
                posterior_variance=float(logvar.exp().mean()),token_identity_exact=int((predicted==val_ids[selected]).sum()),
                token_identity_n=len(selected),full_vocabulary_token_ce=float(F.cross_entropy(similarities/.02,val_ids[selected])))

        result=dict(checkpoint_id=config['memory_checkpoint_id'],steps=a.steps,learning_rate=a.learning_rate,
            train_sha256=digest(a.train),validation_sha256=digest(a.validation),
            scope='Diagnostic only: LoRA codec optimization on bounded train-source positions; held-out source strings. Not full-model training. No weights saved.',
            source_records=dict(train=64,validation=128),positions=dict(train=len(train),validation=len(val)),
            validation_positions_with_seen_token_identity=int(torch.isin(val_ids,train_ids).sum()),
            lora_rank=8,lora_alpha=16,batch=128,kl_weight=.001,initial=evaluate(original),arms={},
            adapter_sha256=digest(a.adapter) if a.adapter else None,
            token_objective_classes=len(vocabulary_ids),token_objective_temperature=.02,
            token_objective_class_selection='All train-position identities plus 4096 seeded random vocabulary draws; validation never used for class selection.')
        dump(a.out,result);print(json.dumps(dict(stage='initial',**result['initial'])),flush=True)
        objectives=('mean_and_identity','categorical_reconstruction') if a.token_reconstruction else ('existing_vae','mean_and_identity')
        for objective in objectives:
            torch.manual_seed(731);head=deepcopy(original).train()
            for name,module in list(head.named_modules()):
                if isinstance(module,nn.Linear):
                    parent,_,leaf=name.rpartition('.')
                    setattr(head.get_submodule(parent) if parent else head,leaf,WeightLoRALinear(module,8,16))
            opt=torch.optim.AdamW([v for v in head.parameters() if v.requires_grad],lr=a.learning_rate)
            for step in range(a.steps):
                tick=time.monotonic();head.train()
                rows=torch.randint(len(train),(128,));x=train[rows]
                decoded,mu,logvar=head(x,sample=True)
                loss=F.mse_loss(decoded,x)+.001*.5*(mu.square()+logvar.exp()-1-logvar).mean()
                if objective in ('mean_and_identity','categorical_reconstruction'):
                    deterministic=head.decoder(mu)
                    loss=loss+F.mse_loss(deterministic,x)+.25*discrimination(deterministic,x,train_ids[rows])
                if objective=='categorical_reconstruction':
                    restored=deterministic*train_scale[rows]+train_mean[rows]
                    logits=F.normalize(restored,dim=-1)@sampled_vocabulary.T/.02
                    loss=loss+F.cross_entropy(logits,torch.searchsorted(vocabulary_ids,train_ids[rows]))
                opt.zero_grad();loss.backward();opt.step()
                time.sleep(time.monotonic()-tick)
                if (step+1)%256==0:
                    metrics=evaluate(head)
                    result['arms'].setdefault(objective,[]).append(dict(step=step+1,**metrics))
                    dump(a.out,result);print(json.dumps(dict(arm=objective,step=step+1,**metrics)),flush=True)
            if a.steps%256:
                result['arms'].setdefault(objective,[]).append(dict(step=a.steps,**evaluate(head)))
        result['completed']=True;dump(a.out,result)


if __name__=='__main__':main()
