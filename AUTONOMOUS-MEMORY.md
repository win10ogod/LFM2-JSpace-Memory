# Autonomous comparison and functional Dream training

This revision keeps the existing LFM2.5-VL backbone, MaleCNS–Titans graph,
physical FFN factors, VAE heads and MaleCNS controller. It adds **no learned
parameters**. The new behavior requires continued training of the inherited
controller and codecs; wiring the mechanism is not evidence of perfect recall.

## Ordinary inference

Associate a caller-owned archive with the model once:

```python
with model.use_memory_archive(archive, top_k=3, index_options={"device": "cuda"}):
    tokens = model.generate(**query_inputs, max_new_tokens=1024, do_sample=False)
```

`top_k` in the context manager is the number of retrieved physical units;
`top_k` passed to `generate` remains the native sampling option. Query length,
output budget, sampling and other generation arguments are preserved.

The normal `generate` call now retrieves model-derived J-lens addresses, loads
the selected units and lets the model's existing controller compare three
query-prefix predictions:

1. Complete ordered representation and physical weights.
2. VAE posterior means and physical weights, without exact corrections.
3. Physical weights alone.

The controller receives predictive entropy, disagreement between the three
distributions, peak probability, source-prediction error when available,
relative read cost when available, and candidate identity. Ordinary read
comparison currently supplies zero for the unavailable source-error/cost
features; it does not claim to minimize latency. These are numeric model
signals. No `concepts`/`intents` JSON or reference answer is provided.

One native answer is generated with the selected substrate. This is currently
three comparison prefills plus one generation, not a claim that comparison is
free or faster than a single read. The selected action, scores and observable
signals are returned in the archive receipt. A standalone physical session
exposes its last decision as `last_memory_action`.

Bindings use `ContextVar`; callers choose their own archive, time/goal scope
and access policy. The model does not invent application-level agent identities
or permission rules. Read scoring starts its controller activity afresh for each
query, matching training. Conversation history comes from the supplied query;
an untrained hidden history cannot change a cold-loaded unit's read policy.

## Automatic consolidation

Ordinary `seal` and subsequent live-session `generate` calls service pending
writes. They no longer require the caller to invoke a Dream API or provide a
retention validator. Candidate construction:

1. Reconstruct the observed graph delta and physical FFN factors with the
   weight VAE.
2. Decode stored native-input posterior means and replay them through the
   native LFM2 stack. Replay the stored visual-port posteriors when present.
3. Commit the resulting observations into the candidate Titans graph.
4. Compare original/candidate compressed source predictions with the original
   complete source predictions. Also check the candidate's complete read.
5. Ask the model controller to rank keeping the original versus adopting the
   replayed candidate. Publish only if it selects the candidate, compressed
   source-prediction KL improves and complete-source KL stays within the
   configured retention tolerance (initially 0.01).

All observed positions contribute to these KL measurements. The output head
is processed in bounded buffers; no source tokens are sampled or truncated.
This checks predictive retention on the source, not all possible future
questions. The inference body stays frozen. Accepted factors become independent
writable leaves, so further physical updates remain possible.

Before an accepted automatic update replaces the active state, the previous
physical unit is saved as an immutable file. Its native concept address remains
queryable, and the new address references the prior memory hash as a parent.
A standalone session with no storage directory defers adoption until `seal`
provides one; an archive supplies its directory when opened. Old files are
never deleted. Failed writes leave the active state unchanged.

The explicit `dream_consolidate` method remains available for callers supplying
their own validator, but it calls the same candidate-construction implementation.
It is no longer the only route to replay. Historical alternative replay/read
implementations have not been added.

## Native LlamaFactory training

The model's custom loss runs inside the native LlamaFactory Trainer. The
tokenizer, whole-example batches, optimizer, backward and checkpoint mechanism
remain native. There is no SFT chunk/session scheduler.

Each source is observed and learned before its question is introduced. Fresh
queries supervise all four positive paths:

```
L = complete CE
  + 0.25 * compressed CE
  + 0.25 * physical-only CE
  + 0.25 * post-consolidation compressed CE
  + 0.10 * (read-policy loss + consolidation-policy loss)
  + 0.05 * VAE auxiliaries
```

The weights are explicit training settings. Wrong and absent memories remain
read-only controls; the optimizer cannot reduce the objective by worsening
those controls.

Policy cross-entropy is decomposed into the entropy of its soft quality target
and excess KL. If original and replayed reads have equal quality, the binary
target is uniform and its entropy is `log(2)`, approximately 0.693. That value
is not evidence of failed convergence. Conversely, three-way CE near `log(3)`
with a decisive target indicates little probability separation.

An inherited-controller range probe found a score spread of about 0.010 over
81 synthetic inputs within the read-feature ranges. This is a scale diagnostic,
not a global bound or a measurement of real queries. Policy-score temperature
is therefore scheduled to change from 1.0 to 0.01 **after checkpoint 32 is fully
saved**. Quality-target temperature remains 0.25. This changes the conditioning
of policy training without changing `argmax` decisions at fixed weights. Raw
and temperature-adjusted CE are both reported; a numerical drop caused only
by rescaling is not counted as improved recall or convergence. All optimizer,
scheduler, RNG and adapter state is retained across the checkpoint transition.

Read-policy inputs use logits just before the first supervised answer token.
The target ranks measured per-example answer NLLs. Consolidation-policy inputs
come only from source predictions; its targets rank fresh-query NLL before and
after actual replay. Answer-derived targets are detached and never enter the
writer, candidate construction or policy input features.

VAE auxiliaries now include sampled reconstruction, posterior-mean reconstruction,
KL and feature discrimination. Identical observed feature rows are positives,
so repeated observations are not incorrectly used as negatives. The old
observed-item-versus-zero-item scheduler objective is removed.

### Observed-text reconstruction objective

Checkpoint 32 changed all eight native-input codec LoRA tensors, but its fixed
validation feature MSE remained approximately 0.972. An additional controlled
CPU experiment compared the existing mean/discrimination objective with an
observed-token categorical reconstruction term. Both arms inherited the same
checkpoint-32 codec, used LoRA 8/16, the same 4,096 training positions, AdamW
learning rate 0.001 and 1,024 updates. The identity diagnostic used the full
128,000-token native vocabulary on 256 held-out positions:

| Objective | Feature MSE | Token identity | Full-vocabulary token CE |
|---|---:|---:|---:|
| Mean reconstruction + feature discrimination | 0.603 | 76/256 | 10.01 |
| Plus categorical source reconstruction | 0.877 | 158/256 | 3.06 |

The categorical experiment trained against observed training identities plus
4,096 seeded random vocabulary draws. Its validation used the full vocabulary.
It did not use validation identities to select training classes. This is a
codec diagnostic, not a whole-model recall result; no diagnostic weights were
installed in the model.

After the complete checkpoint 64 is saved, continuation is scheduled to add
`0.25 * source_codec_CE`. The production objective uses **all vocabulary entries
and all observed text positions**, through bounded, checkpointed output-head
buffers. The frozen native input embedding table defines reconstruction logits.
Training averages posterior-mean and sampled-posterior reconstruction; evaluation
uses the mean. It trains both codec heads without adding model parameters.
Source token IDs are reconstruction targets only. No query answer enters this
term, and the memory reader does not acquire a token lookup operation. This
HashHop stage is text-only; visual features are not trained to predict an image
placeholder token.

The added likelihood term changes the numerical total loss. Each weighted
contribution is logged separately, and success still requires improved held-out
memory recall rather than a lower aggregate scalar.

The source-replay and retention forwards are batched over independent units.
The optimized implementation uses five native consolidation forwards per
training batch instead of five per row, while retaining every source position.
CPU parity tests compare the batched graph, FFN factors, predictions and KL
against independent per-unit execution.

The convergence audit distinguishes stable high loss from useful convergence.
It checks repeated fixed validation sets, like-for-like per-example NLL against
empty/wrong memories, reconstruction against a zero decoder, decision regret,
and post-replay gains. The final detailed audit groups correlated counterfactual
questions by shared source/query identity before bootstrapping gains. Hashes
used for this analysis are never inputs to the model or memory writer.

## Evidence and remaining quality gates

Tests cover real native forwards/backwards, inherited controller gradients,
future-answer isolation, ordinary generate routing, argument passthrough,
automatic save/replay, exact all-position buffered KL, physical-history
preservation, writable post-Dream state and HF AutoClass reload.

The first real GPU training probe used batch 8, LoRA 8/16, native cutoff 4096
and the existing counterfactual HashHop data. Its first successful step reached
43.40 GiB peak allocated memory and nonzero gradients in the controller, feature
VAE and weight VAE. This establishes execution and trainability, not quality.

Reports distinguish selected actions, best observed actions, policy regret,
compressed/physical/consolidated answer CE, consolidation gains and per-codec
distortion/KL. A policy that always chooses complete memory may be appropriate
while the other substrates are weak, but that alone does not establish useful
compressed memory or successful consolidation. Held-out free generation,
coherence and visual retention remain release criteria.
