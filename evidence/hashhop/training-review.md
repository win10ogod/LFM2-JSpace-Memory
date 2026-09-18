# HashHop first-epoch review

Training completed all 32 updates over 256 records, with exit code zero. The 424 effective adapter tensor merge checks passed. All 707 native tensor entries match the parent SFT checkpoint byte-for-byte; 17 graph-memory, 19 physical-memory and 24 Dream-memory tensor entries changed.

## Training measurements

These are online training-time teacher-forced measurements. Both model weights and examples changed between steps; they are not a held-out evaluation of the final checkpoint.

| Mean metric over all 32 batches | Value |
|---|---:|
| Total objective | 10.0417 |
| Supervised answer loss | 7.9675 |
| Memory auxiliary loss | 2.0742 |
| Correct-memory answer NLL | 7.7690 |
| Empty-memory answer NLL | 6.9264 |
| Wrong-memory answer NLL | 7.8548 |

Correct-memory NLL was lower than empty-memory NLL in 0/32 batches and lower than wrong-memory NLL in 26/32 batches. Source/posterior coverage was 219,290/219,290 observed token positions. Peak allocated GPU memory was 40.02 GiB. The epoch therefore does not establish useful memory-conditioned recall, despite finite losses and real updates to all three memory groups.

The resumed Trainer printed train_loss=9.3613. That scalar omits the first two probe-step losses from its accumulated numerator while retaining global_step=32 in the denominator. The all-step mean above is reconstructed from the original per-step records.

## Final-checkpoint evaluation

The coordinator is running the fixed protocol against the merged checkpoint, including native routing, correct-unit oracle, empty and wrong memory, physical/latent components, and complete coherent generation. Their eventual result files, not this training review, determine recall performance. No improvement or convergence claim is made here.
