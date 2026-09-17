# Architecture: fixed backbone, physical memory units, and native concept addressing

## 1. Native LFM2.5-VL inheritance

`Lfm2TitansForConditionalGeneration` inherits Hugging Face's `Lfm2VlForConditionalGeneration`. The native vision tower, multimodal projector, tokenizer, chat template, hybrid language stack, and output head remain in the model. The inherited language stack has 30 layers, combining 22 short-convolution layers and 8 attention layers, with hidden width 2,048. Language memory ports attach at zero-based layers 4, 14, and 26. The visual port reads native vision patch/tile features; image observations are not replaced by generated captions.

Each port has its own feature projections and gates. There is no shared, narrow cross-modal input/output latent. The port interface is still parameter-intensive; adding dense interface parameters is not presented as an increase in persistent memory capacity.

## 2. Sparse MaleCNS–Titans substrate

The inherited selected topology has **90,839 nodes, 1,983,608 directed edges, four channels, and two graph microsteps**. Edge indices are stored in `memory_graph.safetensors`, rather than millions of JSON integers. Destination/source order is part of the checkpoint contract. Four channels provide 7,934,432 fast scalars; graph fast weights plus momentum require 63,475,456 bytes in FP32, before other unit components.

For a port feature `x`, independent projections drive its assigned input nodes. A dendritic graph operator propagates for two steps; the port reads its assigned read nodes and projects back into the native feature space. Compiled finite-step execution plans and CSR sparse multiplication evaluate the reachable computation. Custom sparse autograd preserves the outer derivatives needed to train an online gradient writer. Small compute tiles bound temporary tensor sizes; these are not SFT sample windows.

Within one observation block, every read uses the same snapshot `W_t`. After the complete causal forward has produced its observations, a single commit creates `W_(t+1)`. Thus a later write cannot alter an earlier read in that block.

The writer uses next-feature associations for language and observed-feature reconstruction for vision. It differentiates a mean squared reconstruction objective with respect to physical synaptic weights. A separate write-key projection and learned value projection are present. In the selected configuration:

```text
g = sum of port gradients masked by edge/channel ownership
g_normalized = g / (RMS over writable scalars + epsilon)
momentum_next = beta * momentum + g_normalized
W_next = W - learning_rate * momentum_next
```

Non-writable entries preserve both their previous weights and momentum. Automatic decay is disabled. RMS normalization remains enabled, and the optional surprise-magnitude gate is **not active in this inherited checkpoint**. Accordingly, it would be inaccurate to claim that raw gradient magnitude presently controls write strength.

## 3. Structural write ownership

For port `p`, an edge `e=(u,v)` is structurally eligible when:

```text
distance(input_nodes[p], u) + 1 + distance(v, read_nodes[p]) <= microsteps
```

Eligible edge/channel scalars receive an owner, rotated across channels with seed 17. A shared-association budget of 0.25 is applied only where multiple ports can reach an edge. The resulting masks restrict plasticity, not reading. This separates independently writable capacity from intentionally shared associations.

Structural eligibility is an upper bound on influence. It is not a measured number of facts the model can remember. The retained visual-protection mask covers 6,210 input-subgraph edges; ownership improves write separation, but that mask alone is not proof that every visual association is protected. This model does not establish that the biological topology outperforms a degree-rewired random graph.

There is no persistent neuron-activation vector in this checkpoint. Persistent graph memory is synaptic weight plus momentum; graph activations are recomputed for each feature. The design must not be described as a complete recurrent simulation of the fly brain.

## 4. Direct FFN physical weights

Each memory unit also contains rank-16 physical factors for `w1`, `w2`, and `w3` at the three memory depths. These change the native FFN's computation. They are separate from the rank-8 outer SFT LoRA used to train this release.

The caller owns these factors and their first/second optimizer moments. An observation-derived supervised loss can update them while the backbone remains frozen. Immutable units include their physical FFN state, graph state, VAE codes, ordered observations, and addressing metadata. Loading a memory is therefore loading numerical state used by the forward pass, rather than attaching a textual description to a prompt.

## 5. Dual memory: VAE observations and Titans weights

The VAE participates in ordinary memory storage, not just dreaming. Independent per-port encoders produce a posterior mean and log-variance, with per-feature normalization statistics. Their hidden width is 64 and latent width is 16. The new `native_input` head is initialized as an exact copy of the prior first-language-port head, then trained for the actual input-embedding distribution.

Ordered native input embeddings include the native projected visual features at image positions. Their VAE means are decoded in order. Unit-local residual factors and sparse numerical correction values preserve information the shared decoder cannot reconstruct. Each decoded segment is checked against the original feature checksum. Encoding tiles bound SVD scratch space; all observed feature rows are stored.

This is not a fixed-size lossless compressor. Incompressible observations can require more bytes than raw BF16 features; the previous short-source pilot measured roughly 2.1 times raw feature size for its ordered representation. A checksum proves feature reconstruction, not semantic recall or multi-hop reasoning.

At recall, the selected ordered features are decoded and passed through all native language layers, while the selected graph and FFN physical weights contribute to the forward computation. Native attention/convolution caches are rebuilt and remain ephemeral. Stored token IDs, source strings, pixel arrays, or KV caches are not the serialized representation of the ordered memory.

The current ordered reader accepts a fresh, unpadded, batch-one text query; stored observations may include images. Selected features plus query and generation budget must fit the backbone's native 32,768-position context. Exceeding this raises an explicit error. Unbounded disk growth does not imply unlimited simultaneous attention to every archived observation.

## 6. J-lens-derived addresses

For each calibrated layer, a complete 2,048 × 2,048 averaged Jacobian maps that layer's native residual features toward the final language layer. The estimator sums over valid causal target positions, averages over valid source positions, then averages over calibration prompts. It is not merely a same-position diagonal derivative.

The vocabulary output vectors, final normalization weights, and Jacobian define an overcomplete dictionary of directions. Positive matching pursuit followed by projected nonnegative least-squares refitting selects 16 coordinates per observed position. The stored representation includes vocabulary-coordinate indices, nonnegative coefficients, and residual energy. Vocabulary IDs identify numerical dictionary directions; decoding them to words is only a display operation. No hand-written concept list or generated JSON controls storage.

All observed positions at each memory language depth are retained. Query and memory codes are compared using sparse late interaction on CPU or GPU. The present index performs an exact scan; there is no deployed million-unit ANN throughput claim or alternative pooled-key reader. Physical weight tensors load only after the query chooses candidate units.

Addresses include a checkpoint identity, lens hash, recorded time, optional event time, logical time, and causal parent hashes. A content-derived SHA-256 reference supports exact programmatic lookup. Agent ownership, permissions, and application goals remain caller concerns; the memory layer does not impose a hard-coded multi-agent policy.

## 7. Disk units, hot mounting, and composition

The archive is append-only and has no software unit-count ceiling. Disk space grows with retained information. An active reader pins a complete immutable unit; a background loader stages another unit before an atomic swap. Unsaved changes cannot be silently discarded by mounting a different unit. Readers already holding a snapshot finish with that snapshot.

Multiple selected units keep separate graph/FFN evaluations. Their contributions are combined at the read interface; weights are not blindly averaged. Active composition has an explicit resident-unit bound. The default archive generation call selects one unit, and callers may request more within the configured bound. Selection quality and interference still require measurement.

The published runtime accepts the unified format-6 physical unit with ordered observations and version-2 native concept addresses. Old label-driven, pooled, graph-only, and associative-softmax recall branches are absent. Old artifact conversion, if needed, is an offline operation, not an inference fallback.

## 8. Dream consolidation

A shared weight VAE encodes normalized chunks of 1,024 values through hidden width 128 and latent width 32. It reconstructs graph and FFN deltas. Per-port feature VAEs provide source-conditioned replay. A small MaleCNS-derived sleep controller scores reconstruction-based scheduling signals.

Dreaming constructs a candidate state without mutating the current unit. Ordered decoded observations are replayed through the native model; accepted visual latent replay contributes native visual features. Actual graph commits update the candidate. A caller-provided functional retention validator must accept it before atomic publication. Latent-cycle similarity alone is not treated as a guarantee that an imagined observation is historically true. Existing physical units remain retained.

The sleep-controller topology is biological inspiration. This implementation and its current experiments do not demonstrate biological dreaming or a measured capacity gain from consolidation.

## 9. Joint SFT and release compatibility

Native LlamaFactory owns preprocessing, batching, Trainer, backward, and optimizer updates. The model returns native supervised cross-entropy plus memory auxiliary loss. Real Titans writes/readbacks train the memory, while VAE reconstruction/KL and scheduling objectives train the Dream components. FFN write gradients are captured from the native supervised backward without modifying those gradients; a detached batch of actual updates trains the weight VAE on the following batch.

The backbone, vision tower, projector, graph interfaces, physical FFN priors, VAE heads, and controller receive outer LoRA updates. The final checkpoint merges those adapters. J-lens calibration is then repeated because the native weights changed. Checkpoint-bound old units and addresses are not silently relabeled as compatible with new weights.

See [TRAINING.md](TRAINING.md) for the exact data, auxiliary sampling, timing, and prior-update provenance; see [EVIDENCE.md](EVIDENCE.md) for measured behavior.
