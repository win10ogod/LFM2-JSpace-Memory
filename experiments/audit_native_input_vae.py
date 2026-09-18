"""CPU-only codec audit on observed text embeddings, not generated answers."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from safetensors import safe_open
from transformers import AutoProcessor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lfm2_titans.dream_memory import DreamWeightVAE
from research_common import dump,digest


def main():
    p=argparse.ArgumentParser()
    for k in ['model','sources','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--sample-tokens',type=int,default=256)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(17)
    config=json.loads((a.model/'config.json').read_text())
    mapping=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
    prefix='dream_memory.feature_vae.heads.native_input.'
    names=[n for n in mapping if n.startswith(prefix)]+['model.language_model.embed_tokens.weight']
    with ExitStack() as stack:
        readers={f:stack.enter_context(safe_open(str(a.model/f),framework='pt',device='cpu')) for f in {mapping[n] for n in names}}
        spec=config['dream_memory']['feature_vae']
        head=DreamWeightVAE(config['text_config']['hidden_size'],spec['hidden_size'],spec['latent_size']).eval()
        head.load_state_dict({n.removeprefix(prefix):readers[mapping[n]].get_tensor(n) for n in names if n.startswith(prefix)})
        embedding=readers[mapping[names[-1]]].get_tensor(names[-1])
        processor=AutoProcessor.from_pretrained(a.model,local_files_only=True,trust_remote_code=True)
        ids=torch.cat([processor(text=s['text'],return_tensors='pt',truncation=False)['input_ids'][0]
                       for s in json.loads(a.sources.read_text())])
        choose=set(torch.linspace(0,len(ids)-1,min(a.sample_tokens,len(ids))).long().tolist())
        all_mu=[];mse=[];kl=[];variance=[];cosine=[];targets=[];decoded=[];selected_ids=[];zero_mse=[]
        with torch.no_grad():
            for start in range(0,len(ids),128):
                x=F.embedding(ids[start:start+128],embedding).float()
                mean=x.mean(-1,keepdim=True);scale=(x.var(-1,unbiased=False,keepdim=True)+1e-5).sqrt()
                normalized=(x-mean)/scale
                reconstruction,mu,logvar=head(normalized,sample=False)
                all_mu.append(mu);mse.extend((reconstruction-normalized).square().mean(-1).tolist())
                zero_mse.extend(normalized.square().mean(-1).tolist())
                kl.extend((.5*(mu.square()+logvar.exp()-1-logvar)).mean(-1).tolist())
                variance.extend(logvar.exp().mean(-1).tolist())
                restored=reconstruction*scale+mean
                cosine.extend(F.cosine_similarity(restored,x,dim=-1).tolist())
                for local in range(len(x)):
                    if start+local in choose:
                        targets.append(normalized[local]);decoded.append(restored[local]);selected_ids.append(ids[start+local])
            vocab=F.normalize(embedding.float(),dim=-1)
            chosen_ids=torch.stack(selected_ids);restored=torch.stack(decoded)
            predicted=(F.normalize(restored,dim=-1)@vocab.T).argmax(-1)
            identity=float((predicted==chosen_ids).float().mean())
            true_values=F.normalize(F.embedding(chosen_ids,embedding).float(),dim=-1)
            identity_reference=float(((true_values@vocab.T).argmax(-1)==chosen_ids).float().mean())
            sample=torch.stack(targets);centered=sample-sample.mean(0,keepdim=True)
            singular=torch.linalg.svdvals(centered)
            rank=spec['hidden_size'];rank_bound=float(singular[rank:].square().sum()/sample.numel())
            sampled=[]
            for _ in range(5):sampled.append(float((head(sample,sample=True)[0]-sample).square().mean()))
        mu=torch.cat(all_mu);is_hash=[any('a'<=c<='z' or 'A'<=c<='Z' for c in processor.tokenizer.decode([int(i)],skip_special_tokens=True)) for i in ids]
        avg=lambda values:sum(values)/len(values)
        result=dict(checkpoint_id=config['memory_checkpoint_id'],scope='CPU native-input feature codec only; all text source positions. This is not end-to-end answer accuracy.',
            source_file_sha256=digest(a.sources),positions=len(ids),hash_token_positions=sum(is_hash),
            normalized_reconstruction_mse=avg(mse),zero_decoder_mse=avg(zero_mse),hash_token_mse=avg([m for m,h in zip(mse,is_hash) if h]),
            kl_per_latent_coordinate=avg(kl),posterior_variance=avg(variance),mean_reconstructed_feature_cosine=avg(cosine),
            mu_coordinate_variance=mu.var(0,unbiased=False).tolist(),
            decoder_final_affine_width=rank,feature_width=sample.shape[1],
            bounded_token_identity_probe=dict(tokens=len(chosen_ids),decoded_nearest_embedding_accuracy=identity,
                original_nearest_embedding_accuracy=identity_reference,sampled_posterior_mse=sampled,
                best_rank_hidden_width_mse_on_sample=rank_bound,
                meaning='Observed-token discrimination diagnostic; no memory question is answered by this embedding lookup.'))
        dump(a.out,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
