"""Merge the completed native SFT and audit every effective adapter tensor."""
import argparse
import importlib
import json
import os
from pathlib import Path
import time
import hashlib

from research_common import load, dump, log, paced
from lfm2_export_artifacts import sync_runtime


def tensor_hash(value):
    import torch
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main():
    import torch
    from peft import PeftModel
    from peft.tuners.lora import Linear
    parser = argparse.ArgumentParser()
    for name in ('base', 'adapter', 'target', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-steps', type=int, default=464)
    args = parser.parse_args()
    gate = os.environ.get('LFM2_START_GATE')
    while gate and not Path(gate).exists():
        time.sleep(.1)
    state = json.loads((args.adapter / 'trainer_state.json').read_text())
    if state['global_step'] != args.expected_steps or state['epoch'] < .999:
        raise RuntimeError('Refuse to publish incomplete SFT as the final model')
    if args.target.exists():
        raise FileExistsError(args.target)
    native, processor = load(args.base)
    helper = importlib.import_module(type(native).__module__.rsplit('.', 1)[0] + '.sft_lora')
    model = PeftModel.from_pretrained(native, args.adapter, is_trainable=False)
    helper.load_custom_lora(model, args.adapter, required=True, trainable=False)
    model.eval().requires_grad_(False)
    expected = {}
    with torch.no_grad():
        for name, layer in native.named_modules():
            if isinstance(layer, Linear):
                base = layer.get_base_layer().weight
                expected[name + '.weight'] = tensor_hash(base + layer.get_delta_weight('default').to(base.dtype))
        for item in native._sft_custom_lora_spec['targets']:
            parent, name = helper._split(native, item['path'])
            value = getattr(parent, name)
            if item['kind'] == 'linear':
                expected[item['path'] + '.weight'] = tensor_hash(value.weight)
            else:
                expected[item['path']] = tensor_hash(value)
    inputs = processor(text='新觀測站的代號是什麼？', return_tensors='pt').to('cuda')
    with torch.no_grad():
        before = paced(model, **inputs, use_cache=False, logits_to_keep=1).logits.float().cpu()
    source_id = native.config.memory_checkpoint_id
    merged = helper.merge_sft_lora(model).eval().requires_grad_(False)
    for path, checksum in expected.items():
        parent, name = helper._split(merged, path)
        if tensor_hash(getattr(parent, name)) != checksum:
            raise RuntimeError('Merged effective tensor mismatch: ' + path)
    with torch.no_grad():
        after = paced(merged, **inputs, use_cache=False, logits_to_keep=1).logits.float().cpu()
    kl = float(torch.nn.functional.kl_div(after.log_softmax(-1), before.log_softmax(-1),
               log_target=True, reduction='batchmean'))
    if not torch.isfinite(after).all() or kl > .01:
        raise RuntimeError(f'Merge behavior check failed: KL={kl}')
    residual = [n for n, _ in merged.named_parameters() if '.lora_' in n or n.endswith(('.adapter_A', '.adapter_B'))]
    if residual:
        raise RuntimeError('Unmerged LoRA tensors: ' + str(residual))
    previous = dict(merged.config.finetuning)
    previous.pop('source_checkpoint', None)
    merged.config.finetuning = dict(stage='sft_merged', native_step=state['global_step'], epochs=state['epoch'],
        method='native LlamaFactory, all-module LoRA rank 8 alpha 16, joint memory auxiliary loss',
        dataset_counts=dict(VisualWebInstruct=2718, Omnimodal_image_only=994),
        cutoff_len=4096, physical_batch=8, training_chunks=0, previous_stage=previous)
    for name in list(merged.config.to_dict()):
        if name == 'native_joint_sft' or name.startswith('native_sft_'):
            delattr(merged.config, name)
    # Native backbone weights changed. An old lens must not be advertised as current.
    merged.config.concept_memory = None
    merged.save_pretrained(args.target, safe_serialization=True, max_shard_size='4GB')
    processor.save_pretrained(args.target)
    root = Path(__file__).resolve().parents[1]
    runtime = sync_runtime(root / 'src/lfm2_titans', args.target)
    for file in args.target.glob('*.py'):
        if file.name not in runtime:
            file.unlink()
    receipt = dict(status='passed', native_step=state['global_step'], epochs=state['epoch'],
        source_checkpoint_id=source_id, checkpoint_id=merged.config.memory_checkpoint_id,
        verified_effective_weight_tensors=len(expected), unmerged_parameters=residual,
        logits_kl=kl, logits_max_abs_difference=float((before-after).abs().max()),
        next_token_equal=bool(torch.equal(before.argmax(-1), after.argmax(-1))),
        prediction_identity_claimed=False, lens_recalibration_required=True, runtime=runtime)
    dump(args.receipt, receipt)
    log('merge_complete', **{k:v for k,v in receipt.items() if k!='runtime'})


if __name__ == '__main__':
    main()
