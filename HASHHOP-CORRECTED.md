# Corrected HashHop: recall improved, compressed memory remains weak

Checkpoint: `273e195bd4ad4c10ace63c3bcd5516bd`. This is the completed
128-update, all-module LoRA 8/16 continuation. It is **not a replacement
release** for the published HashHop model yet. Actual generation still has
substantial failures. The following numbers are measurements, not projected
benefits from adding a VAE or a sleep controller.

## What the loss means

The completed run optimized:

```
complete-memory answer CE
  + 0.25 * posterior-code-and-weight answer CE
  + 0.05 * combined Dream reconstruction auxiliaries
```

The final held-out complete-memory answer CE was approximately 0.201; the
compressed-memory answer CE was 6.264. **6.264 is not the VAE reconstruction
MSE.** Exact numerical corrections are present in the complete-memory read
and absent in the compressed auxiliary. A low complete-memory CE therefore
does not demonstrate that the learned compression works.

The exact path is detached. The compressed auxiliary supplies task gradients
to the codec. Physical-only recall is measured without gradients; it is not
an independently optimized positive objective in this run. Graph and FFN
weights receive gradients from the positive dual-memory queries, but those
queries can also exploit the fully reconstructed source features.

## Actual generation

All readers cold-loaded saved units. The model body stayed fixed. The matched
protocol, prompts, distractors and generation limits were kept unchanged.

| Condition | Exact hash answers |
|---|---:|
| Native retrieval and composition | 13/42 |
| Provided correct units | 20/42 |
| Original source visible, memory disabled | 19/42 |
| Wrong memory | 0/42 |
| Memory disabled, source absent | 0/42 |
| Provided units, physical weights only (subset) | 0/12 |
| Provided units, ordered VAE plus exact corrections only (subset) | 5/12 |
| Provided units, posterior codes plus weights without corrections (subset) | 0/12 |

The old 32-update pilot scored 0/42 with native retrieval and 1/42 with correct
units in the same matched protocol. The improvement is real; the remaining
failures are also real. Correct-unit composition answered 3/6 cross-unit
questions; native routing answered 0/6.

A separate held-out counterfactual set reused the same query with two
different observed contexts. The supplied-current-unit reader answered
76/128 correctly, versus 0/128 without memory. It answered **both versions**
correctly in 34/64 pairs. This tests memory dependence with caller-selected
units, not global retrieval. No question or answer entered the observation
writer. Teacher-forced NLL and free-generation exact match are different
measurements.

The short text/vision suite scored 14/17 exact, 16/17 under its original
content rubric, with 17/17 native routing and identical hot/cold answers.
Wrong/empty controls were 0/17. The content rubric can accept extra text, so
strict exact scores and raw answers remain available.

All 32 multi-turn coherence outputs were reviewed. Successful initial reads
often preserved identifiers and filenames, but follow-ups omitted requested
identifiers, confused retries with total attempts, or introduced unsupported
restart/redeployment steps. The native reader's first manual also contradicted
its own checksum-failure instructions. Component ablations sometimes repeated
text to the unchanged 1,024-token limit. This is not reliable manual-following
generation yet.

## Direct codec measurements

The native-input VAE was loaded directly from the completed checkpoint,
together with its input embedding table, on CPU. All 12,995 positions in the
16 held-out source texts were measured, without answers or query prompts.

- Normalized deterministic reconstruction MSE: **0.97218**.
- Mean reconstructed raw-feature cosine: **0.00579**.
- KL per latent coordinate: **0.02604**.
- Mean posterior variance: **0.96383**.
- A bounded 256-position nearest-input-embedding identity diagnostic recovered
  **0/256** from decoded features, versus **256/256** from original features.
- Sampled-posterior MSE on that bounded probe was about **0.997–0.999**.

This is evidence of weak feature preservation. The embedding lookup is an
offline diagnostic; it is not a recall engine or a way to supply answers to
the model. Low KL by itself does not establish useful memory. These numbers
do not by themselves identify KL regularization as the cause.

The decoder ends in a 64-to-2,048 linear map. Its normalized output is therefore
limited to an affine subspace of dimension at most 64. This is a real structural
constraint, but it is not proof that the current failure is unavoidable:
the best affine rank-64 reconstruction on the bounded sample had MSE about
0.244, substantially below the current codec. That bound is fitted to that
sample and is not an unseen-data performance claim.

### Controlled codec-only optimization

Two diagnostic arms began from the same checkpoint head, with the same LoRA
8/16, 4,096 training positions, identical sampled batches, seed, AdamW learning
rate 0.001 and 1,024 updates. Only the objective differed. The second arm added
deterministic-mean reconstruction and a multi-positive feature discrimination
term to the sampled-reconstruction and KL loss. Repeated token identities
were treated as positives, not false negatives.

| Head | Validation MSE | Identity diagnostic |
|---|---:|---:|
| Inherited checkpoint | 0.9720 | 0/256 |
| Further optimization, existing VAE objective | 0.5875 | 58/256 |
| Further optimization, mean reconstruction + discrimination | 0.6027 | 76/256 |

Validation used 2,048 positions from held-out source strings; 1,491 positions
had subword identities also seen among the training positions. This is not
an unseen-vocabulary test. Both arms use the same budget, but the codec-only
learning rate and update count differ from the completed whole-model run.
No diagnostic weights were saved or installed. This is neither an end-to-end
recall improvement nor a new published checkpoint.

The result supports two practical conclusions: the inherited head can learn
more without enlarging it, and smaller MSE does not necessarily mean better
identity preservation. It does not isolate the added mean term from the
discrimination term or establish a generally optimal loss.

## What Dream actually does, and what is missing

Code inspection confirms that normal training reconstructs graph updates,
FFN updates and native features. Its scheduler auxiliary ranks one observed
item above a zero-signal dummy item. It does **not** measure which replay
improves recall. Training does not execute the full
`replay -> consolidate -> fresh query` loop as its Dream objective.

The explicit inference `dream_consolidate` API does construct a candidate:
reconstruct graph and FFN weights, replay the ordered observations through
the native model, commit graph observations, and invoke a caller's retention
validator. Generated language-port samples are reported but not substituted
into this graph replay; only accepted generated visual samples replace that
port's native replay features. The scheduler does not select between competing
compressed and physical answers. Normal generation always reads the complete
ordered representation with physical weights.

An actual GPU experiment ran that API on one text unit, one hash-chain unit,
and one visual unit. The validator captured the candidates and declined all
publication, without access to questions or answers. Six existing questions
were opened after candidate construction and generated before/after replay:

| Diagnostic reader | Before Dream | After Dream |
|---|---:|---:|
| Complete memory | 4/6 exact | 4/6 exact |
| Physical weights alone | 0/6 exact | 0/6 exact |

Every answer was unchanged within its reader condition. Visual content was
correct but used an extra labeled format in both conditions, so it failed the
strict exact rubric. All 256 generated visual features were rejected by the
existing latent-cycle gate. This small experiment found no recall gain; it
does not establish that Dream can never help. Weight reconstruction distortion
was about 1.003 on the text and chain units. The much lower visual-unit aggregate
must not be interpreted as better visual memory: it averages graph and many
FFN-factor reconstruction terms, including near-zero update tensors.

### Fixed lifecycle defect

The candidate FFN factors returned from Dream reconstruction were frozen decoder
outputs. Accepting them made a later physical gradient write invalid. This was
confirmed on the real checkpoint's three candidates as well as a small CPU
fixture. The source now detaches a candidate into independent writable leaves
before validation/publication. The regression test performs an actual
`observe -> learn -> seal -> accept Dream -> learn -> seal` cycle and checks
that base-model parameters remain unchanged. This repairs continued writing;
it is not evidence of improved retention. The benchmark candidate was kept
unchanged, so its recorded results still describe its original runtime.

## Storage is part of effective capacity

The exact reconstruction corrections have a measurable cost. For the first
35-position text unit, the original BF16 input-feature tensor would occupy
143,360 bytes. Its ordered VAE representation plus correction tensors occupies
296,560 bytes, approximately **2.07 times** that amount. The entire physical
unit occupies 85,955,964 bytes including graph weights/momentum, FFN weights
and optimizer moments, port codes and retrieval addresses.

This does not mean every future source has the same ratio. It means the current
sample cannot be advertised as effective compression merely because the file
contains VAE codes. The separate storage audit records all nine short-suite
units from their safetensors byte offsets.

## Implementation changes and next training requirements

The current source now reports sampled reconstruction, deterministic
reconstruction, zero-decoder baseline, cosine, KL and posterior variance
separately from query CE and the combined Dream auxiliary. Regression checks
confirm the added diagnostics preserve the existing objective, gradients and
RNG state. There are no new model parameters or additional inference branches.
The lifecycle repair and current native-SFT, recall, LoRA and export tests pass
(22 targeted checks). Historical window-based SFT tests still describe an
intentionally removed API; they are not evidence about the current path.

The next learning objective needs separate evidence for:

1. **Useful compressed content:** improve code-only reconstruction and fresh
   query answers, while measuring actual stored bytes and independently
   checking the physical-only read. Exact corrections must not mask this test.
2. **Functional consolidation:** supervise the actual replayed candidate's
   subsequent recall and retention, rather than treating codec MSE or scheduler
   ranking as a substitute. Retention comparisons must not feed test answers
   into the writer.
3. **Model-directed comparison:** train and evaluate query-conditioned reading
   and comparison of memory substrates. A controller that always selects the
   exact representation could lower combined answer loss while leaving the
   compressed substrate useless; that would not meet this project's goal.

These are requirements still to implement and validate, not capabilities of
the current checkpoint. The architecture direction—MaleCNS–Titans physical
memory plus persistent VAE representations, expandable archived units, and
model-derived concept indexing—is retained. A VAE optimizes a variational
objective; it does not itself guarantee exact recall or improved consolidation
([original VAE paper](https://arxiv.org/abs/1312.6114)).
