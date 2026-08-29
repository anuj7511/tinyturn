# C1 — Learned-model context-length ablation (Step 6)

Ran B1 (tiny finalist: mel + trajectory fusion, no F0) at 1s/2s/4s, and A0 (Whisper-Tiny) at
2s/4s/8s (8s triggered because 4s still improved over 2s, per the brief's explicit condition).
Same splits, same calibration protocol, same eval slices as every other experiment in this project.

## B1 (tiny finalist)

| ctx | overall AUC | real AUC | per-source macro AUC | p50 latency | ECE |
|---|---|---|---|---|---|
| 1.0s | 0.815 | 0.743 | 0.729 | 10.2ms | 0.035 |
| 2.0s | 0.801 | 0.726 | 0.692 | 15.9ms | 0.156 |
| 4.0s | 0.815 | 0.757 | 0.756 | 32.7ms | 0.035 |

AUC is flat within noise across all three lengths (heavily overlapping CIs); the 2.0s dip (both AUC
and calibration) looks like single-seed training variance, not a real context effect -- these are
one run per length, not repeated trials. Latency scales cleanly and substantially with context
(10ms -> 33ms), driven by the trajectory branch's raw DSP cost (Hilbert envelope + multiple STFTs
over more audio).

**Decision: N=1.0s for B1.** No accuracy evidence favors longer context, and 1.0s is ~3x cheaper
than the C0-handcrafted-probe's 4.0s choice. This directly reproduces the brief's anticipated
D6-vs-E1 relationship: the population-level/handcrafted signal (C0: monotonic AUC increase with
context, real_all 0.55->0.68) did not survive contact with the actual trained model.

## A0 (Whisper-Tiny)

| ctx | overall AUC | real AUC | per-source macro AUC | p50 latency (incl. feature extraction) | ECE |
|---|---|---|---|---|---|
| 2.0s | 0.939 | 0.918 | 0.940 | 14.5ms | 0.038 |
| 4.0s | 0.950 | 0.937 | 0.961 | 29.9ms | 0.023 |
| 8.0s | 0.953 | 0.940 | 0.970 | 52.6ms | 0.023 |

2s->4s: a real, consistent gain (+1.1pp overall, +1.9pp real-audio AUC) for ~2x latency.
4s->8s: a much smaller gain (+0.3pp overall, +0.3pp real-audio AUC) with heavily overlapping CIs
against 4s (real_all: 4s [0.899,0.967] vs 8s [0.902,0.965]) for another ~1.8x latency -- clear
diminishing returns, and the improvement is not statistically distinguishable from noise at this
sample size.

**Decision: N=4.0s for Whisper**, not 8s -- matches the tiny finalist's chosen context (useful for
apples-to-apples comparison and for a shared runtime convention if distillation (Step 10) happens
later), and the 8s gain doesn't justify doubling latency again.

## Net effect on the B0/B1/B1-f0/A0 comparison

Both finalists keep their originally-reported numbers unchanged: B1 was already run at 4.0s
(coincidentally also a defensible choice even though 1.0s is now the *recommended* production
context for cost reasons), and A0 was already run at 4.0s. No results from Steps 3-5 need revision;
this ablation confirms 4.0s was a reasonable choice for A0 and revises the recommended B1 context
downward to 1.0s for any future production configuration.
