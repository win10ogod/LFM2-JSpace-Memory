"""Shared loading and measurement helpers for the published experiments."""
import hashlib
import json
from pathlib import Path
import re
import time

ATTENTION = 'kernels-community/flash-attn2@f50dc99ed079b35990bc895d43fd353ea0cb376d'


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def log(stage, **values):
    print(json.dumps(dict(time=time.time(), stage=stage, **values), ensure_ascii=False), flush=True)


def paced(call, *args, **kwargs):
    import torch
    start = time.monotonic()
    result = call(*args, **kwargs)
    torch.cuda.synchronize()
    time.sleep(time.monotonic() - start)
    return result


def load(path):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.set_num_threads(2)
    log('loading_model', path=str(path))
    model, info = AutoModelForImageTextToText.from_pretrained(
        path, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16,
        device_map='cuda', attn_implementation=ATTENTION, output_loading_info=True)
    for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        if info.get(key):
            raise RuntimeError(f'Checkpoint load {key}: {info[key]}')
    model.eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    return model, processor


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def normalized(value):
    return re.sub(r'[\s。．.,，！!？?「」"\u201c\u201d：:；;]+', '', value.strip())


def question_inputs(processor, question):
    return processor.tokenizer.apply_chat_template([dict(role='user', content=question)],
        tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=True).to('cuda')


def generated(processor, question, call, **kwargs):
    inputs = question_inputs(processor, question)
    output = paced(call, **inputs, max_new_tokens=64, do_sample=False, **kwargs)
    tokens = output[0, inputs['input_ids'].shape[1]:]
    return dict(answer=processor.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                generated_tokens=tokens.numel(), hit_generation_limit=tokens.numel() == 64)
