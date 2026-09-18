"""Compare the actual merged tensor bytes against the inherited SFT model."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from research_common import dump


def main():
    parser=argparse.ArgumentParser()
    for key in ('before','after','out'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    maps=[json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
          for root in (args.before,args.after)]
    if maps[0].keys()!=maps[1].keys():raise RuntimeError('Merged tensor names changed')
    rows=[];groups=Counter();changed_groups=Counter();changed_native=[]
    with ExitStack() as stack:
        readers=[{shard:stack.enter_context(safe_open(str(root/shard),framework='pt',device='cpu'))
                  for shard in set(mapping.values())} for root,mapping in zip((args.before,args.after),maps)]
        for i,name in enumerate(sorted(maps[0])):
            tensors=[reader[mapping[name]].get_tensor(name) for reader,mapping in zip(readers,maps)]
            hashes=[hashlib.sha256(memoryview(t.reshape(-1).view(torch.uint8).numpy())).hexdigest() for t in tensors]
            equal=(hashes[0]==hashes[1] and tensors[0].dtype==tensors[1].dtype and tensors[0].shape==tensors[1].shape)
            group='native' if name.startswith(('model.','lm_head.')) else name.split('.')[0]
            groups[group]+=1
            if not equal:
                changed_groups[group]+=1
                if group=='native':changed_native.append(name)
            rows.append(dict(name=name,group=group,equal=equal,before_sha256=hashes[0],after_sha256=hashes[1]))
            del tensors
            if (i+1)%100==0:print(json.dumps(dict(checked=i+1,total=len(maps[0]))),flush=True)
    passed=not changed_native and all(changed_groups[g]>0 for g in ['memory','physical_memory','dream_memory'])
    result=dict(status='passed' if passed else 'failed',tensor_counts=dict(groups),
                changed_tensors=dict(changed_groups),changed_native=changed_native,parameters=rows)
    dump(args.out,result);print(json.dumps({k:v for k,v in result.items() if k!='parameters'}),flush=True)
    if not passed:raise RuntimeError('Memory training inheritance/update audit failed')


if __name__=='__main__':main()
