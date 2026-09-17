# J-space evidence on an LFM2.5-VL-3B-derived checkpoint

## Claim and scope

The experiments find J-lens directions with measurable causal effects on reporting and some downstream tasks in a hybrid LFM language stack. They also find a sparse fitted component that outperforms its norm-matched remainder in a small intervention pilot. These results support further use of native concept coordinates for memory addressing. They do **not** prove perfect recall, every global-workspace criterion, consciousness, or a universal advantage over other address representations.

The measured model is a derivative of `LiquidAI/LFM2.5-VL-3B`, after earlier memory PT and partial SFT: checkpoint ID `1238a963afc04c1ba75e1e08e2572378`. During J-space tests the model is frozen, memory reads/writes are disabled, and the native language stack is intervened on. This distinguishes the backbone representation measurements from memory retrieval results.

## Method and calibration

The definition and functional criteria follow [Anthropic's global-workspace study](https://transformer-circuits.pub/2026/workspace/index.html). A sparse nonnegative combination of a few strongly active J-lens directions is not automatically evidence of reportability, control, reasoning mediation, flexible use, or selectivity; those properties need interventions and controls.

Our independent estimator follows the averaging method in the [J-lens reference implementation](https://github.com/anthropics/jacobian-lens). It computes complete matrices at layers 4, 8, 14, 20, 26, and 28 toward layer 29. It skips the first 16 positions and excludes the final target position. There is no learned probe optimizer. The larger calibration used four authored paragraphs plus the first 128 tokens of eight Formosa articles, not a representative pretraining corpus.

Split-half matrix cosine similarities were 0.500, 0.534, 0.664, 0.800, 0.952, and 0.985 at those respective layers. Early-layer estimates are materially less stable. Parameter versions and native outputs were checked before and after calibration; both remained unchanged.

The calibration matrix is a release asset with a SHA-256 in `evidence/provenance.json`. Third-party article text is not redistributed. The calibration provenance includes article IDs, token windows, and text hashes. Exact regeneration additionally requires those upstream texts and the recorded checkpoint/tokenizer.

## Functional studies

| Study | Conditions including controls | Main result |
|---|---:|---|
| Exploratory layer interventions | 272 | Layer-26 swaps: 5/16 target prefixes; earlier layers weak |
| Four new countries, matched same-layer controls | 240 | Layer-26 swaps: 2/16; three random seeds and wrong-concept control: 0/16 each |
| Sparse J component versus remainder | 96 | J component: 5/16; norm-matched remainder: 0/16 |
| Native chat format and fixed clean-reference edits | 352 | Best tested fixed-reference pattern: 13/32; matching random edit: 0/32 |
| **Total** | **960** | Includes clean runs; not 960 nonzero edits |

Every trial, including failures and controls, appears under `evidence/pre-sft`. The interactive HTML shows the original outputs and intervention measurements.

### New-concept controls and rescue

The second study used Kenya, Norway, Canada, and Vietnam, following exploratory France/Japan/Italy/China probes. The original answer was correct on only 9/16 clean prompts. Among those nine, ablation retained 5/9 correct answers, while restoring the clean representation at a later layer recovered 9/9. This is a small causal-rescue result; the incomplete clean baseline limits what the test can say about general reasoning.

When the model was asked to focus on a concept while copying the same surface text, the four tested concepts had positive layer-26 cosine changes (approximately 0.0031–0.0074), and all four copying responses remained correct. Same-surface readouts use teacher forcing, with a separate free-copy check. The size and task range are insufficient to establish broad voluntary control.

### Sparse-component comparison

Independent `Tell me about ...` probes were centered using 100 baseline concepts. Positive matching pursuit with 16 coordinates and 48 projected NNLS iterations produced a fitted J component and a remainder. The interventions were matched in norm:

| Intervention | Target prefixes / 16 |
|---|---:|
| Full probe | 3 |
| J component | 5 |
| Remainder | 0 |
| Remainder, with J coordinates clamped | 0 |
| J component, with remainder clamped | 5 |

The five J-component successes comprised three report tasks, one language task, and one continent task. The remainder retained roughly 85–89% of the probe energy at layer 26. A nonnegative-fit remainder is **not** assumed to be orthogonal to the entire overcomplete J-space. This experiment tests local causal privilege, not all possible representations or tasks.

### Native chat transfer

This follow-up used 32 native-chat questions about the same eight already investigated concepts. It is format transfer, not a blind held-out-concept result. Clean generation answered 27/32 correctly.

| Edit | Target responses / 32 |
|---|---:|
| Swap at layer 26 | 8 |
| Sequential ordinary swaps at 26 and 28 | 0 |
| Fixed clean-reference coordinates at 26 and 28 | 10 |
| Fixed clean-reference coordinates at 20, 26, and 28 | 13 |
| Norm-matched random edit at the same three layers | 0 |

The 13 successes split into report 8/8, language 4/8, capital 1/8, and continent 0/8. Eleven of those successes occurred among the 27 clean-correct cases. Repeated ordinary swaps can undo earlier changes; the zero-result condition is retained rather than omitted. Ablation left 23/32 original answers correct, and rescue recovered 27/32.

Full-answer teacher-forced log probabilities accompany free-generated prefix scores. A visible diagnostic output cap of 12 tokens applies to these short-answer intervention tests. It is not the model's general output limit.

## Property-level interpretation

| Property | Present evidence | Remaining gap |
|---|---|---|
| Reportability | Concept swaps alter short verbal reports | Wider concepts, instructions, and model checkpoints |
| Controllability | Small positive focus-versus-mention changes with identical copied output | Stronger, broader directed-control tests |
| Silent intermediate reasoning | Some downstream answers change when an implicit concept is edited | A broad compositional, multi-step benchmark |
| Causal mediation | Selective perturbation and later clean-state rescue | Broader necessity/sufficiency tests with matched difficulty |
| Flexible downstream use | Report and language tasks transfer; limited capital transfer | Weak continent performance and sparse task coverage |
| Selectivity/broadcast | Fitted J component outperforms remainder locally | Broad routine-computation controls and circuit-level broadcast |

No automatic `global_workspace_established=True` conclusion is produced by the scripts.

## Separate memory acceptance pilot

Nine newly written units contained six random code/string sources, one shuffled hash chain, and two newly drawn native images with source references. The writer used only the observations and source-derived labels. Questions were opened only after all writes. Cold reads occurred in a new process; source-visible controls ran last.

| Condition | Exact / 17 | Content / 17 |
|---|---:|---:|
| Native concept router + ordered recall | 15 | 15 |
| Prior pooled router, same ordered decoder | 12 | 13 |
| Correct-unit oracle, cold load | 15 | 15 |
| Wrong unit | 0 | 1 |
| Empty memory | 0 | 0 |
| Source visible directly | 14 | 15 |

Native top-one routing was 17/17, versus 12/17 for the previous pooled router. CPU and GPU rankings matched; all 17 hot/cold oracle outputs matched. Two three-hop questions failed even with the correct unit, so retrieval alone does not solve their reasoning failures. The wrong-unit visual hit illustrates chance answers in a tiny two-shape test.

The old pooled router is retained only as a historical comparison result, not as a branch in the current runtime. The tiny corpus, named source references, and short observations do not establish arbitrary code-manual recall, million-unit indexing performance, or extreme-context capacity. No test answer was used to optimize a write or route.

## New SFT results

Results above predate the new 3,712-example SFT. Post-SFT lens calibration, memory checks, and functional controls must carry the new checkpoint identity and be reported separately. Unfinished or failed checks remain visible; no old score is substituted for a missing new result.
