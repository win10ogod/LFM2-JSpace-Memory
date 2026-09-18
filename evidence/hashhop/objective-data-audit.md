# Training objective and dataset audit

This audit concerns the completed 32-update pilot. It identifies reproducible training-design defects without changing the architecture or attributing every failure to one cause.

## 1. The examples do not require the requested hop count

All 256 labels are arithmetically correct. All 256 questions start at a chain root and request the complete path to its terminal node. Each of the 160 graphs has only one start/hop combination; trace and final-answer variants share it.

A deliberately incorrect solver that ignores the requested hop count and always walks to the terminal node gets **256/256 training examples correct**, but only **10/42 evaluation examples correct**. Training does not distinguish the desired fixed-hop rule from that shortcut. Lower-hop evaluation questions are prefixes of deeper chains, a relationship absent from training.

The repair is to sample graph depth, start position and requested hop count independently, with multiple valid distances from the same start and queries beginning inside chains. The difficult evaluation should remain intact.

## 2. No same-question counterfactual memories

There are zero identical-query/different-memory answer pairs. Query hashes are globally unique between generated training graphs, so the dataset alone cannot rule out memorizing query-to-answer mappings in slow weights.

Use paired observations with the same query key and requested hop count but different valid associations and different gold answers. Each memory context must be trained toward its own correct answer. This forces a query-only predictor to fail without adding hand-written concept labels.

## 3. The negative contexts are trainable degradation targets

The implemented contrast is

```
0.1 * (relu(0.25 + correct_nll - empty_nll)
     + relu(0.25 + correct_nll - wrong_nll))
```

Both sides backpropagate. At correct=8, empty=6, wrong=7.9, the gradients are +0.2, -0.1, -0.1 respectively. Gradient descent can reduce this term by increasing empty/wrong-context NLL while leaving correct recall unchanged. This establishes an objective loophole; it does not prove the trained model exploited it in every failed case.

The wrong-unit selector also chooses the first different observation. For eight distinct rows it returns `[1,0,0,0,0,0,0,0]`, concentrating negative pressure on one unit. Fixing the training design should prioritize positive correctness in counterfactual memory pairs, protect the reference condition from deliberate degradation, and balance negative assignment if it remains useful. Simply changing the epoch count does not remove these issues.

## 4. Training coverage differs from the deployed task

The pilot supplies one freshly written unit directly to each query. It does not train archive selection or cross-unit composition. Native backbone parameters are frozen, while native concept addressing runs with memory disabled. More iterations of that same objective therefore do not directly optimize the deployed index.

The main training read uses VAE posterior means and physical weights without sequence corrections. Production recall uses the complete corrected observation representation. These need separately named losses and measurements: actual deployed recall, compressed-code fidelity, and optional component diagnostics. The current pilot's training NLL cannot be treated as the production reader's NLL.

## What the actual tests establish

Small new identifiers survive writing and cold reload: 12/12 code/string answers and one one-hop answer are exact; image colors/shapes are also recalled. On the separate matched HashHop development protocol, native routing remains 0/42 and correct-unit oracle remains 1/42. These observations support preserving the current architecture while correcting training objectives and data first.

A controlled continuation on inherited weights is required to measure how much the repairs improve held-out recall. No corrected-training result is claimed by this audit.
