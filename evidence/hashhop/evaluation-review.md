# HashHop continuation: actual generation results

Parent SFT: `95c663dda67d4dbe8ec77a4681db3fc2`. HashHop: `0b5ec95eac3849e38867dcba9048c623`. All 278 generation conditions completed under the same protocol.

## Short fresh-memory recall

Native routing selected the expected unit in 17/17 cases.

| Condition | Exact answers | Recorded content score |
|---|---:|---:|
| native-default | 13/17 | 15/17 |
| oracle-cold | 13/17 | 15/17 |
| wrong-memory | 0/17 | 0/17 |
| empty | 0/17 | 0/17 |
| source-visible | 12/17 | 13/17 |

Content scores use the original substring/visual rules and can reward an expected string embedded in a wrong chain. Strict exact scores and raw answers remain primary. The short protocol is not a matched before/after improvement claim because its old SFT reader had different framing.

The generation condition named `empty` calls `use_memory=False`: it disables the memory branch. It differs from the training objective’s empty condition, which uses fresh pre-write graph/FFN states. Those NLL and generation controls should not be conflated.

## Matched unseen HashHop queries

| Condition | Parent SFT exact | HashHop exact | Gained / lost cases |
|---|---:|---:|---:|
| native-routed | 0/42 | 0/42 | 0 / 0 |
| oracle-units | 1/42 | 1/42 | 0 / 0 |
| wrong-unit | 0/42 | 0/42 | 0 / 0 |
| empty | 0/42 | 0/42 | 0 / 0 |
| source-visible | 4/42 | 4/42 | 0 / 0 |
| oracle-physical-only | 0/12 | 0/12 | 0 / 0 |
| oracle-latent-only | 1/12 | 1/12 | 0 / 0 |
| oracle-codes-and-weights | 0/12 | 0/12 | 0 / 0 |

Hash exact match compares whitespace-separated hash sequences: every hash character and its case must match; whitespace separators may differ. These development queries include 1/2/4/8/10 hops and cross-unit chains. Training graphs and symbols are disjoint. Oracle conditions supply the correct unit selector; source-visible controls supply the original text. They are diagnostics, not replacements for native routing. This is not an extreme-context capacity benchmark.

## Coherent generation

- native-routed: 2/4 parent outputs and 1/4 candidate outputs hit the generation limit.
- oracle-units: 2/4 parent outputs and 0/4 candidate outputs hit the generation limit.
- wrong-unit: 0/4 parent outputs and 0/4 candidate outputs hit the generation limit.
- empty: 0/4 parent outputs and 0/4 candidate outputs hit the generation limit.
- oracle-physical-only: 0/4 parent outputs and 1/4 candidate outputs hit the generation limit.
- oracle-latent-only: 0/4 parent outputs and 0/4 candidate outputs hit the generation limit.
- oracle-codes-and-weights: 0/4 parent outputs and 0/4 candidate outputs hit the generation limit.
- source-visible: 0/4 parent outputs and 0/4 candidate outputs hit the generation limit.

Unabridged answers are in `generation/answers.json`; the separately authored manual review assesses factual consistency, repeated text and instruction following. Generation length alone is not a quality score.
