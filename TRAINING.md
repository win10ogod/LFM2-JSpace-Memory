# Joint visual and memory SFT

## Data and native workflow

This run uses exactly two local dataset views:

| Source | Selected examples |
|---|---:|
| [TIGER-Lab/VisualWebInstruct](https://huggingface.co/datasets/TIGER-Lab/VisualWebInstruct), existing selected local view | 2,718 |
| [RUC-NLPIR/Omnimodal-Agent-SFT-2K](https://huggingface.co/datasets/RUC-NLPIR/Omnimodal-Agent-SFT-2K), complete local **image-only** view | 994 |
| Total | **3,712** |

The 685 video trajectories are not part of this run. “Complete” means all 994 rows of the selected image-only view, not all records in the upstream multimodal dataset. This local view contains 1,044 image references. The combined normalized data contain 2,505 image references before native truncation. Data are not deduplicated.

LlamaFactory's native ShareGPT loader, `SupervisedDatasetProcessor`, multimodal collator, Trainer, and backward/optimizer loop process the data. Role and tool-call normalization converts the selected local formats to the native LFM chat/tool template while retaining thought and tool text. There is no per-dataset training scheduler, trajectory-specific loop, or custom SFT chunking. Long samples are truncated by the native 4,096-token cutoff, as selected for this run; their truncated tails are not represented as having been trained.

The LFM vision plugin uses the original HF processor's image placeholder expansion, including image tiles and start/end markers. Eleven samples whose native cutoff intersected an image span required removal of the incomplete span and matching trailing image metadata. These remain single examples, and their retained assistant labels are unchanged. No fake placeholder image is added to text-only rows.

The resulting native HF DatasetDict contains 3,712 rows, 5,658,894 input tokens, and 1,652,087 supervised assistant targets. Its columns are the ordinary native token/label/image fields; it contains no record-window/session scheduling fields. Packing is not enabled. Native length grouping reduces padding.

## Optimization settings

| Setting | Value |
|---|---|
| Outer LoRA | Rank 8, alpha 16, dropout 0 |
| Scope | Language, vision tower, projector, memory, VAE, controller |
| Physical batch / accumulation | 8 / 1 |
| Epochs / optimizer steps | 1 / 464 |
| Learning rate | 1e-5, cosine decay, no warmup |
| Weight decay / gradient clip | 0.01 / 1.0 |
| Precision | BF16 native model; FP32 memory computation/state |
| Attention | FlashAttention-2, pinned kernel revision |
| Activation checkpointing | Non-reentrant, native forward |
| Data cutoff | 4,096 tokens |
| GPU workers | One |

LoRA tensor ranks are capped by the dimensions of a target tensor: four graph channels permit rank four, while scalars/vectors use the appropriate rank-one parametrization. “Rank eight” does not imply an impossible rank-eight scalar update. There are 26,855,601 trainable outer parameters in this run. All requested module groups produced finite, nonzero gradients in actual training. Disabled forgetting scalars and some write-scale factors do not receive a gradient in this objective; the report does not claim every scalar updates.

The direct physical FFN memory rank is 16 and is independent of the outer training LoRA rank. The latter is merged into the inference checkpoint after SFT.

## Loss and acceleration

The primary loss is native LFM supervised cross-entropy over the native assistant label mask. One native VL forward executes per batch. The model adds an auxiliary memory loss with coefficient 0.05:

1. Real differentiable Titans writes followed by associative reconstruction.
2. Per-port feature VAE reconstruction and KL (beta 0.001).
3. Graph and FFN weight-VAE reconstruction/KL.
4. A reconstruction-derived scheduling loss for the sleep controller.

Thirty-two feature pairs per port and example are sampled for the auxiliary objective. This sampling applies only to that auxiliary training objective; it is not a storage limit or an inference context cap. All native targets retained by the chosen cutoff remain in the primary CE.

FFN updates reuse gradients from the native supervised backward. Hooks return the original gradients unchanged and separately rescale captured values to per-example target means. Actual physical optimizer updates provide a detached replay batch for the following batch's weight-VAE loss. The replay buffer is checkpointed and restored. This avoids an additional full language-model source backward.

Expanded custom LoRA weights are cached within a forward and restored consistently during non-reentrant checkpoint recomputation. Shared fresh graph priors can be read together across batch rows. A graph compute tile of 256 bounds internal sparse work; it does not split a training example.

Actual 8 × 4,096-token batches with the earlier extra-source-backward implementation took 53–57 seconds of compute and peaked at 87.24 GiB. Subsequent batches using native-backward capture took about 32–36 seconds and peaked around 63–64 GiB. These are different batches of the same shape, **not a controlled identical-input speed benchmark**. Liger/fused CE is not enabled. A compute/rest ratio of 1:1 is retained for the user's low-pressure WSL constraint, so wall time per update also includes rest and checkpoint I/O.

## Provenance and continuation

The starting checkpoint already contained memory PT (536 updates within the prior two-hour budget) and an earlier partial SFT checkpoint at native step eight. It is not presented as a fresh untouched backbone. A dedicated `native_input` VAE head was added by exact copying of the old first-language-port head; all existing parameters were checked unchanged before the new SFT.

The first two genuine optimizer updates of this run used the extra source-gradient auxiliary implementation. Updates three onward reuse the native backward and next-batch FFN replay. Valid first updates were retained; the run resumed from step four with optimizer, RNG, and data position state. This is an auxiliary-training schedule change, not a claim of identical loss to the earlier implementation. Main native CE remained unchanged.

All 464 updates subsequently completed with exit code zero. After a later worker interruption, training resumed from a validated step-352 checkpoint with optimizer, scheduler, RNG and data position retained. The merged checkpoint is `95c663dda67d4dbe8ec77a4681db3fc2`: 424 effective adapter tensors were audited, with no unmerged parameters. The merge probe measured logit KL 0.0007121 and maximum absolute logit difference 0.140625; identical predictions on every input are not claimed.

Fresh J-lens calibration and cold-process checks were performed. A changed backbone invalidates the previous calibration; reusing its UUID or merely copying its old lens is not accepted. Training logs and optimizer artifacts remain outside the model directory.

The original LlamaFactory `lfm2_vl` template lacked the leading BOS emitted by the HF native chat template. This was corrected for the subsequent HashHop continuation, not retroactively for these 464 completed updates. The published ordered reader also places the query BOS before recalled features, and indexes complete query text separately from chat control tokens. Neither repair changes the completed SFT weights. Matched before/after evaluation artifacts identify the reader version.

The subsequent memory-dependent HashHop stage has a separate repository and checkpoint identity. Its extra training must not be attributed to this visual/agent SFT stage.
