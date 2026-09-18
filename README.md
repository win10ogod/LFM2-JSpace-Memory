# LFM2.5-VL-3B: MaleCNS–Titans memory and J-space experiments

This project extends Liquid AI's LFM2.5-VL-3B with independently stored, hot-mountable physical memory units. Each unit combines sparse MaleCNS–Titans fast weights, direct FFN memory weights, and ordered VAE-encoded native observations. Retrieval addresses come from the model's internal representations through a calibrated Jacobian lens. The model does not generate `concepts` or `intents` JSON to index a memory.

The design separates a fixed inference backbone, expandable disk storage, and bounded active memory composition. The runtime in `src/lfm2_titans` contains one integrated archive and recall path. Historical comparisons are evidence files, not alternate inference modes.

## Read the results

- [Architecture and storage contract](ARCHITECTURE.md)
- [J-space methods, controls, results, and interpretation](EVIDENCE.md)
- [Native LlamaFactory SFT and provenance](TRAINING.md)
- [Standalone interactive evidence viewer](evidence.html) — download and open locally.
- [Raw research artifacts](evidence/pre-sft) and [provenance/checksums](evidence/provenance.json)
- [Calibration matrix release](https://github.com/win10ogod/LFM2-JSpace-Memory/releases/tag/pre-sft-evidence-v1)

The current evidence contains **960 functional test conditions**, including clean and negative controls. All were run on the LFM2.5-VL-3B-derived checkpoint `1238a963afc04c1ba75e1e08e2572378`, after prior memory PT and partial SFT, with the memory branch disabled for the J-space intervention experiments. These are not measurements of an untouched official Liquid AI checkpoint.

The strongest native-chat intervention redirected 13/32 responses to the counterfactual target; matched random controls redirected 0/32. A separate sparse-component experiment redirected 5/16 responses with the J component and 0/16 with its norm-matched remainder. This is positive, limited functional evidence. It does not establish all properties of a global workspace or perfect memory.

A separate fresh-source memory pilot correctly routed 17/17 queries and recalled 15/17 answers exactly. Two three-hop hash questions still failed even when given the correct memory unit. See the complete controls and failures in [EVIDENCE.md](EVIDENCE.md).

## Model release status

- **sft:** [win10/Lfm2-MaleCNS-Titans-VL-3B-V2-SFT](https://huggingface.co/win10/Lfm2-MaleCNS-Titans-VL-3B-V2-SFT), verified commit `267e2478cf00e0103ee7ac79f9d780dc9fa88f2d`.
- **hashhop:** [win10/Lfm2-MaleCNS-Titans-VL-3B-V2-HashHop](https://huggingface.co/win10/Lfm2-MaleCNS-Titans-VL-3B-V2-HashHop), verified commit `a36b628297ab53b0527efb69ac08369ce9a48372`.

These are separate complete model repositories. Each receipt verifies every remote file against the local package using Windows-native tooling. A stage absent from this list is not yet published.

[Post-SFT evidence](evidence/post-sft) belongs to checkpoint `95c663dda67d4dbe8ec77a4681db3fc2`; the 960 historical conditions above remain separate. Post-SFT functional studies include 688 conditions. Memory/generation results include failures and are not a claim of perfect recall.

[HashHop training and evaluation protocol](HASHHOP.md) describes the subsequent memory-only continuation.

## Reproduce the functional tests

Use the recorded environment in [evidence/environment.json](evidence/environment.json), the appropriate checkpoint, and its matching lens directory containing `jacobians.safetensors` and `result.json`.

```bash
export PYTHONPATH="$PWD/src:$PWD/experiments"
python experiments/test_jspace_functions.py --model /path/to/checkpoint --lens /path/to/lens --out ./pilot
python experiments/test_jspace_functions.py --model /path/to/checkpoint --lens /path/to/lens --out ./validation --validation
python experiments/test_jspace_privilege.py --model /path/to/checkpoint --lens /path/to/lens --out ./privilege
python experiments/verify_jspace_transfer.py --model /path/to/checkpoint --lens /path/to/lens --out ./transfer
```

The scripts reject a lens with the wrong checkpoint identity. They never optimize the inference body. The historical checkpoint's exact weights were local at the time of publication; recording a UUID and checksums does not by itself make those weights publicly downloadable. The supplied matrices and trial outputs remain inspectable independently. Repeating on the new SFT release is a new experiment and receives its own evidence directory.

CPU regression tests:

```bash
PYTHONPATH=src pytest -q tests/test_jspace_interventions.py tests/test_export_artifacts.py
# Also install the user's LlamaFactory integration for the joint-SFT tests:
LLAMAFACTORY_SRC=/path/to/LlamaFactory/src PYTHONPATH=src pytest -q tests/test_joint_native_sft.py
```

## Sources and attribution

This is an independent implementation and evaluation, not an Anthropic or Liquid AI release. The J-lens method is based on [Anthropic's global-workspace research](https://transformer-circuits.pub/2026/workspace/index.html) and its [reference implementation](https://github.com/anthropics/jacobian-lens). The backbone is [LiquidAI/LFM2.5-VL-3B](https://huggingface.co/LiquidAI/LFM2.5-VL-3B); its model license remains applicable. The graph derives from [MaleCNS v1.0](https://male-cns.janelia.org/), whose dataset attribution requirements remain applicable. Training sources are linked in [TRAINING.md](TRAINING.md). Third-party dataset examples and training images are not bundled in this repository.
