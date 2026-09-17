# Memory-dependent HashHop continuation

## Status

The 3,712-example visual/agent SFT completed all 464 optimizer steps. The next
stage is separate and must follow baseline cold recall and generation tests.
The owner has authorized separate Windows-native Hugging Face publication as
`win10/Lfm2-MaleCNS-Titans-VL-3B-V2-SFT` and
`win10/Lfm2-MaleCNS-Titans-VL-3B-V2-HashHop`. Each upload requires completed
training, an audited merge, matching calibration, and remote file verification.

The new objective has passed small native LFM-VL CPU tests, including compiled
sparse writes, LoRA, non-reentrant checkpointing and optimizer updates. GPU
performance and recall improvement have not yet been established.

## Why source/query separation matters

The observation turn contains only a shuffled random hash table. The writer
cannot inspect the later question or answer. It performs actual Titans graph
updates, a source-only first-order FFN update, and VAE posterior encoding.
The query forward starts with no previous attention or convolution cache.
Its inputs consist of the query and decoded stored posterior codes, while
written graph/FFN weights act through the normal memory ports.

The training read deliberately excludes the exact sequence reconstruction
corrections. Query loss therefore trains the existing VAE encoder and decoder
to retain task-relevant information. An additional query loss uses physical
weights alone. Empty and wrong-memory controls measure conditional likelihood
and supply a margin loss with gradients through both positive and negative
readouts. All observed positions enter the writer and posterior
store; there is no query-dependent selection of facts to remember.

The production archive retains its complete sequence corrections and its one
canonical reader. Training interventions do not delete stored information or
introduce alternative legacy inference architectures. No model parameters are
added by the objective. Parameter shapes remain inherited from the completed
SFT checkpoint.

Changing future query/answer tokens in the CPU test leaves the written graph
weights, FFN factors and every VAE posterior tensor bit-for-bit identical.
Both memory paths receive gradients from the query objective. A second test
confirms that the native language/vision weights stay unchanged under a
memory-only LoRA optimizer step.

## Native training workflow

LlamaFactory retains its dataset loader, template, collator, Trainer, backward
and optimizer. Each sample is one standard four-message conversation:

1. User: observation table.
2. Assistant: a constant acknowledgment, excluded from the supervised targets.
3. User: start key, hop count and answer format.
4. Assistant: exact target values.

Native `mask_history: true` selects the final answer. The model's objective
splits the two native user turns internally; no SFT window scheduler or custom
Trainer is used. The complete source and answer are verified after native
preprocessing. The initial trial has 256 records, a maximum of 2,212 native
input tokens per record, physical batch 8, and LoRA rank 8 / alpha 16. Its first
trial scope is the memory modules; the completed native language and vision
weights remain frozen. Frozen native weights also mean native-only J-space
routing cannot improve through backbone changes in this particular trial;
index failures must be distinguished from read/write failures.

The subsequent HashHop template restores the native leading BOS; all 256
tokenized conversations match the HF native chat template exactly. This does
not retroactively change the completed visual/agent SFT's tokenization. The
shared ordered reader places that BOS before recalled features and retains
every memory feature. Addressing uses the complete raw query separately from
generation's chat serialization. These changes are evaluated on identical SFT
weights and archived units before the HashHop continuation starts.

## HashHop protocol

Reference: [Magic's 100M-token-context article](https://magic.dev/blog/100m-token-context-windows)
and [official generator](https://github.com/magicproduct/hash-hop/blob/main/hashhop/generate.py).
The local variant uses arrow notation and space-separated traces from the
article, with native chat formatting and cold physical-memory controls.

- Random 8- and 16-character strings use the 52 ASCII letters.
- All target and distractor chains have equal depth within an observation.
- Pair order is shuffled, and every requested edge is present.
- Hash queries contain no table name, source ID or storage location.
- Training and evaluation graphs and symbols are disjoint.
- Training covers 1, 2, 4 and 8 hops, both traces and final-only answers.
- Development tests include 10-hop extrapolation and chains across four units.
- The generator's graph solver is never used to answer model queries.

The development evaluation compares native routing, correct-unit oracle,
wrong memory, empty memory and source-visible controls. Additional declared
oracle interventions isolate physical weights, full corrected latent memory,
and posterior codes plus weights without corrections. The source-visible
controls run only after memory-only answers have been recorded.

Hash grading preserves every character and its case. Full coherent multi-turn
answers are saved for manual assessment alongside fact retention, repetition
and output-limit diagnostics. These heuristics alone do not establish coherence.
A zero-shot HashHop failure is not treated as a measured capacity ceiling.
Before/after memory storage bytes and exact recall should be reported together;
random hash training does not remove the information cost of retaining random
associations.
