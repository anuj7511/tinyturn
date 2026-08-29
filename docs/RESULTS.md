# TinyTurn — Final Results: Finalist Selection and Official Test Evaluation

*Continues from [docs/EXPERIMENTS.md](EXPERIMENTS.md) Section 4, which left A0 disqualified as a
distillation teacher under its original prediction-conditioned flip-rate gate. This document covers
everything from there to the final shipped checkpoint: a corrected metric audit, a distillation
ablation, a pairwise-ranking experiment, finalist selection, a 3-seed confirmation, temperature
scaling, and the official test evaluation. Bottom line: **the corrected metric audit meaningfully
softens the "A0 is disqualified" picture; distillation is dropped; the ranking experiment is a
real, kept result but doesn't unseat either finalist; the finalists remain the hold-loss objective
at λ=0.5 with proportional real/synthetic sampling, and λ=0.5 with 50:50 sampling.***

## 1. Metric audit: the original teacher-qualification gate was inflated

The original teacher-qualification gate's "safety-critical flip rate"
(`qualify_teacher_a0_ci_gated.py::direction_specific_gates`) conditions only on the **canonical
prediction** (canonical says incomplete → alt flips to
complete). It never checks whether canonical was actually right — a flip counted there can be a
*correction* (truth is really complete, canonical was wrong, the alt boundary fixed it), not a
safety problem.

A new script, `scripts/ground_truth_conditioned_metric_audit.py`, recomputes the same event conditioned on ground
truth **and** canonical correctness:

```text
introduced false completion: truth=incomplete, canonical=continue (correct), alternative=complete (now wrong)
introduced delay:            truth=complete,   canonical=complete (correct), alternative=continue (now wrong)
```

| | A0_original | whisper_tiny_boundary_robust_retrain | 2% bound |
|---|---|---|---|
| Introduced false completion (alt_threshold) | 3.21% [2.17, 4.74] — decisive-fail | **1.06%** [0.54, 2.08] — inconclusive | ≤2% |
| Introduced false completion (silero) | 3.61% [2.50, 5.21] — decisive-fail | **2.13%** [1.31, 3.42] — inconclusive | ≤2% |
| Introduced delay (alt_threshold) | 9.09% | 5.88% (improved) | — |
| Introduced delay (silero) | 7.84% | 6.83% (improved) | — |

The old gate reported boundary-robust A0 at 8.0–10.5%, "4-6x over the 2% bound" — decisively failing.
Conditioned on ground truth, it is **statistically indistinguishable from the bound** (inconclusive,
not decisive-fail). This doesn't reverse the teacher-qualification FAIL verdict (inconclusive still
blocks, per this project's own qualification rule), but it means A0 is much closer to viable than
the original flip-rate gate suggested. Also reported per-boundary FCR-at-recall95 (canonical vs. alt vs. Silero) standalone —
no absolute degradation from either alternative boundary on whisper_tiny_boundary_robust_retrain in any slice.

Artifact: `experiments/metric_audit_ground_truth_conditioned.json`.

## 2. Distillation ablation: dropped

Two runs, B1@1s student, whisper_tiny_boundary_robust_retrain as an offline teacher (despite its own gate FAIL),
fixed recipe (T=2, α=0.5, teacher loss on final clips only, hard labels on internal
holds, student boundary augmentation enabled). Teacher logits precomputed once over all 12,797 D2
train clips (`data_cache/teacher_logits_a0_boundary_robust_train.parquet`).

| | Overall AUC | Real AUC | Δ overall | Δ real |
|---|---|---|---|---|
| B1 baseline (`mel_trajectory_1s_speech_aligned_contract`) | 0.8279 | 0.7402 | — | — |
| D1 (canonical-boundary teacher) | 0.8277 | 0.7168 | −0.02pp | **−2.34pp** |
| D2 (mean-of-3 teacher) | 0.8269 | 0.7186 | −0.10pp | **−2.16pp** |

Neither run clears this project's own keep bar (+0.5pp overall AUC *or* +1pp real AUC) — both move
backward on both axes. **Dropped**. No need to run the ground-truth-conditioned
VAD audit on these checkpoints; the AUC gate alone disqualifies both.

### 2b. Correction: D1/D2 weren't an isolated test of distillation

D1/D2 changed two things vs. the untouched B1 baseline simultaneously — teacher logits *and*
student boundary augmentation — so "distillation failed" wasn't actually attributable to
distillation alone. A control, **D0** (`scripts/train_distillation_isolation_control.py`): identical
protocol, boundary augmentation enabled, but hard labels only (`alpha=1.0`, so
`train_distill._distill_loss`'s soft/teacher term is multiplied by zero and contributes nothing).

| | Overall AUC (Δ vs. baseline) | Real AUC (Δ vs. baseline) |
|---|---|---|
| B1 baseline (no augmentation, no teacher) | 0.8279 | 0.7402 |
| **D0** (boundary-aug, hard labels only) | 0.8000 (**−2.79pp**) | 0.6626 (**−7.76pp**) |
| D1 (boundary-aug + canonical teacher) | 0.8277 (−0.02pp) | 0.7168 (−2.34pp) |
| D2 (boundary-aug + mean-of-3 teacher) | 0.8269 (−0.10pp) | 0.7186 (−2.16pp) |

**This flips the attribution.** Boundary augmentation *alone* costs a large −7.76pp real AUC (D0)
— far worse than D1/D2's −2.16 to −2.34pp. Adding the teacher-logit signal on top of that same
augmentation *recovers* roughly two-thirds of the loss (D1/D2 vs. D0: +5.0 to +5.4pp real AUC). The
distillation signal was helping, substantially — it just wasn't enough to fully cancel out boundary
augmentation's own cost and come out ahead of the plain, unaugmented baseline.

The shipping decision doesn't change: per this project's own rule ("even if distillation improves
over D0, do not ship it unless it also beats the ordinary B1 baseline"), D1/D2 still trail the untouched
baseline on real AUC (−2.16 to −2.34pp), so **distillation stays dropped for Step 10**. What changes
is the *scientific* conclusion: this is not "distillation failed" — it's "boundary augmentation has
a real cost that distillation partially, but not fully, offsets." Worth remembering if boundary
augmentation is ever revisited for B1 independent of distillation (it currently only exists in this
project as part of A0's boundary-robust remediation retrain and these D0/D1/D2 controls, never evaluated as a
plain B1 training-time addition on its own without either teacher logits or hold-loss terms).

New code: `tinyturn/train_distill.py`, `scripts/precompute_teacher_logits.py`,
`scripts/train_distillation_ablation.py`. `tinyturn/dataset.py` gained `augment_boundaries` and
`teacher_logit_path` support (mirroring `WhisperTurnDataset`'s remediation-retrain mechanism); the
on-disk feature cache is disabled whenever `augment_boundaries=True` since it's keyed by
`(row_id, context_s)` only, not by which boundary produced the window.

## 3. Pairwise-ranking experiment: kept, but doesn't dominate

Replaces λ=0.75. Within-utterance margin loss, pairing each completed clip's final score against
one internal-hold score from *the same clip* (at most one pair per clip per epoch):

```text
L = L_final_BCE + 0.1 * max(0, 0.2 - s_final + s_hold)
```

Main BCE stays final-clips-only; checkpoint selection stays final-clips-only; no student boundary
augmentation (not part of this experiment's recipe). New code: `tinyturn/train_ranking.py`,
`scripts/train_pairwise_ranking.py`.

3-seed result (seeds 42/43/44, identical plateau protocol: epochs≤40, early_stop_patience=6,
lr_schedule=plateau — see Section 5 for why this took a retrain):

| Arm | Overall AUC | Real AUC (Δ vs. baseline) | Hold FCR, all | Hold FCR, real |
|---|---|---|---|---|
| B1 baseline | 0.8315 ± 0.002 | 0.7545 ± 0.009 | — | — |
| **ranking** | 0.8284 ± 0.002 | 0.7345 ± 0.008 (**−2.00pp**) | 0.138 ± 0.012 | 0.113 ± 0.030 |
| λ=0.5 all | 0.8312 ± 0.004 | 0.6998 ± 0.003 (−5.47pp) | 0.110 ± 0.010 | 0.110 ± 0.007 |
| λ=0.5 50:50 | 0.8258 ± 0.006 | 0.6962 ± 0.013 (−5.83pp) | 0.119 ± 0.011 | **0.039 ± 0.005** |

Ranking clears its own keep bar decisively (this project's rule required ≥5pp hold-FCR gain over
baseline at ≤2pp real-AUC cost; single-seed result was −15.8pp/-16.9pp hold-FCR at **+0.31pp** real
AUC). Across 3
seeds it has by far the smallest real-AUC cost of the three engineered arms, but its hold-FCR is
worse than both λ=0.5 variants (13.8% all vs. 11.0–11.9%; real hold-FCR ties λ=0.5-all but trails
50:50's 3.9% by a wide margin). **A genuine Pareto trade-off, not dominance in either direction** —
the differences are all multi-point, not sub-1pp noise.

## 4. Finalist selection: unchanged

Per this project's own rule ("replace one only if distillation or ranking clearly dominates"): distillation
failed its own gate outright; ranking is real but doesn't dominate either λ=0.5 arm. **Finalists
remain `λ=0.5 all` and `λ=0.5 50:50`.** Ranking stays documented as a viable alternative (smallest
real-AUC cost of any hold-aware objective tried) if the accuracy/hold-FCR trade-off is ever
revisited.

## 5. Three-seed confirmation: partial

A protocol mismatch was caught while assembling this table: the existing seed-42 checkpoints for
`λ=0.5 all`/`λ=0.5 50:50` (`experiments/pause_event_sampling_comparison/pause_events_holdloss0.5_proportional*`
and `.../pause_events_holdloss0.5_5050sampling*`) were trained under the **old
fixed-5-epoch protocol**, while the same arms at seeds 43/44 (and the baseline at every seed) use
the validated **plateau/early-stopping protocol** (epochs≤40, patience=6). This project's own
reuse rule ("existing seed-42 runs can be reused if their manifests and code hashes match") caught
exactly this — they didn't match, so seed 42 for both λ=0.5 arms was retrained under the matching
protocol (`experiments/pause_event_sampling_comparison/pause_events_holdloss0.5_proportional_seed42`
and `.../pause_events_holdloss0.5_5050sampling_seed42`), and ranking was run fresh at all
three seeds under the same protocol (it had only ever run under the old one). The table in Section 3
is fully protocol-consistent across all four arms and all three seeds.

**Not yet computed** (deferred by explicit user decision, not by default): hold FCR at
recall90/recall95-matched thresholds (vs. each arm's own calibrated threshold, used above);
short-complete recall; the Section 1 ground-truth-conditioned VAD-boundary audit applied to these
9 B1 checkpoints. The core finalist decision (Section 4) does not depend on these — they would add
detail, not change the call, since the AUC/hold-FCR gap between ranking and the λ=0.5 arms is
already multiple points wide, not a close call these metrics would flip.

## 5b. Correction: hold FCR at matched recall, calibration-then-validation

Section 3's hold-FCR numbers were computed at each arm's own independently-calibrated threshold
(target FCR=0.05 on final clips) — not comparable across arms with different calibration curves,
and not even the same threshold-selection method this project's own keep/promotion rules use (matched
complete-turn recall). Own-threshold hold-FCR made ranking (real AUC 0.7345) and `λ=0.5 all` (real
AUC 0.6998) look nearly tied on real-hold FCR (11.3% vs. 11.0%) — a much closer call than the
headline real-AUC gap suggested, worth checking properly rather than trusting.

`scripts/matched_recall_audit.py` recomputes this correctly, and also fixes a second,
subtler issue found in the *existing* precedent for this kind of analysis
(`train_pause_sampling_comparison.py`): that script selects its threshold from `val`'s own recall curve,
then reads hold-FCR off that *same* `val` split — circular. This audit selects the threshold from
the **calibration** split's recall curve (targeting 90%/95% recall on the complete class), then
evaluates recall / hold-FCR / short-complete / response-particle recall on **validation only**. No
threshold is ever chosen and evaluated on the same split. Run across all 12 already-trained
checkpoints (4 arms × 3 seeds) — inference only, no retraining.

| Arm | @ recall≈90%: hold FCR all / real / synth | @ recall≈95%: hold FCR all / real / synth |
|---|---|---|
| B1 baseline | 68.2% / **75.6%** / 66.7% | 80.3% / **87.4%** / 78.9% |
| ranking | 71.6% / **70.6%** / 71.8% | 82.7% / **86.0%** / 82.0% |
| λ=0.5 all | 60.7% / **67.4%** / 59.3% | 73.5% / **81.2%** / 71.9% |
| λ=0.5 50:50 | 63.6% / **55.6%** / 65.3% | 76.8% / **73.0%** / 77.6% |

(3-seed means; achieved val recall at these calib-selected thresholds is 0.91/0.96, close to the
0.90/0.95 targets as expected.)

**The matched-recall hypothesis does not hold up: ranking's real-hold FCR
is *worse* than `λ=0.5 all`'s at every matched recall point (70.6% vs. 67.4% @90; 86.0% vs. 81.2%
@95), not a near-tie.** The apparent tie under own-threshold calibration (11.3% vs. 11.0%) was an
artifact of each arm landing at a different, arm-specific point on its own hold-FCR-vs-recall curve
— forcing a shared operating point resolves the ambiguity in `λ=0.5 all`'s favor, not ranking's.
`λ=0.5 50:50` remains the best (lowest) real-hold-FCR arm at both matched points, consistent with
Section 3's own-threshold finding. Short-complete and response-particle recall (also computed at
these same calib-derived thresholds) show no arm meaningfully sacrificing short/particle replies
beyond the general recall-vs-FCR trade-off already visible in the headline numbers.

This sharpens, rather than overturns, Section 4's finalist call: `λ=0.5 all`/`λ=0.5 50:50` remain
the finalists, and this new evidence is a point *against* substituting ranking in for `λ=0.5 all`
(if anything), not for it. Full curves (50-point grid, calib-selected thresholds) and per-seed detail:
`experiments/matched_recall_audit_calib_then_val.json`; plot (seed 42):
`experiments/matched_recall_audit_seed42.png`.

## 6. Data scaling, 32k tier: decisive win for the baseline

No ASR transcription needed (plain B1/pause/ranking training only needs waveform, endpoint label,
metadata, speech boundaries, pause events, log-mel/trajectory features — `endfiller` ground truth
already ships natively on synthetic rows in the raw HF dataset, confirmed real rows have it null
100% of the time, same convention the 15,998-clip working set already relies on). This also removed
the biggest cost/time driver (and the OpenAI API spend) from the original scaling estimate.

**Whole-shard fetching**: `pipecat-ai/smart-turn-data-v3.2-train` is stored as 83
individually-downloadable parquet shard files (confirmed via `HfApi.list_repo_files`), ~3265 rows /
~500MB each — direct shard access works, no need to stream the full 270,933-row
dataset the way the original 16k build did (3.6 hours). Five shards (10/20/30/40/50) were checked
individually (metadata columns only, no audio) against the population-level composition reported in
[docs/EDA.md](EDA.md) Section 1 before committing — all five landed within ~1-2pp of the full
270,946-row population on every column checked (language eng/spa, synthetic rate, dataset chirp3_1
share). Decode + DSP feature extraction (no ASR) for the resulting ~15,277 new clips initially ran
on 8 threads (~1.9 clips/s, `librosa.pyin`-bound and thread/GIL-limited); switched to a
`ProcessPoolExecutor` (14 of 16 cores) after the user asked about GPU acceleration — not applicable
here (no GPU-accelerated code path for this DSP pipeline; the bottleneck is pure CPU) — but true
multiprocessing gave a **~4.3x speedup** (~7.9 clips/s), finishing all 5 shards in 23 minutes.
Appends only: the existing 15,998 D2 rows and their split assignments are untouched; every new row
is `split="train"` only, so val/calib stay fixed across tiers exactly as the data-scaling protocol requires.
(`scripts/build_data_scale_tier.py`.)

Trained B1 baseline and the user-selected obj(`λ=0.5 50:50`, chosen for having the best real-hold-FCR
of the four Section 5 arms) at 32k, one seed (42), identical plateau protocol to their 16k
counterparts:

| | Overall AUC | Real AUC |
|---|---|---|
| 16k baseline | 0.8316 | 0.7436 |
| **32k baseline** | **0.8522 (+2.06pp)** | **0.7988 (+5.52pp)** |
| 16k `λ=0.5 50:50` | 0.8241 | 0.7107 |
| 32k `λ=0.5 50:50` | 0.8500 (−0.22pp vs. 32k baseline) | 0.7391 (−5.97pp vs. 32k baseline) |

**32k clears the continue-bar (≥0.5pp overall or ≥1pp real AUC) decisively for the baseline** — this
is the largest single gain seen anywhere in this Step 10 pass. The `λ=0.5 50:50` trade-off persists
at 32k and, in relative terms, *widens*: more data helped the plain baseline roughly twice as much
(+5.52pp real AUC) as it helped the hold-aware objective (+2.84pp, from 0.7107→0.7391), while
`λ=0.5 50:50`'s excellent real-hold-FCR is essentially unchanged (4.71% at 32k, own threshold, vs.
the 32k baseline's 56.51% at that same threshold). The accuracy/hold-safety tension is not a
16k-specific artifact.

### 6b. 64k escalation: baseline gain confirmed unusually large; objective trained there too

Per the user's explicit instruction (the 32k gain was judged "unusually large" against this
project's own 64k-escalation trigger): ran a 64k B1 baseline, seed 42 only, same protocol, then trained the
chosen final objective (`λ=0.5 50:50`) at 64k since the baseline cleared the gate.

**Data build**: 11 more whole shards (5/15/25/35/45/55/60/65/70/75/80), same representativeness
check as the 32k tier (all 11 landed within ~1-3pp of population on every column checked). 33,727
new clips, 65,042 total rows (train 61,841; val/calib untouched at 1600/1601). Decode+feature
extraction took 4,867s (~81 min) with the same process-pool approach as the 32k tier.

**Training hit a real infrastructure problem worth recording**: after the user asked whether more
DataLoader workers would speed up training (the model is tiny — 110K params — so almost all epoch
time is data-loading/cache I/O, not compute), the first attempt bumped `num_workers` 4→12. This
oversubscribed system memory (12 train-loader workers + 12 persistent val-loader workers, each
importing torch/scipy/librosa on this 16GB-RAM machine) and crashed with a Windows paging-file
exhaustion error mid-epoch-1. Restarted at `num_workers=6` — stable, and still meaningfully faster
(epoch 1: 757s vs. the original attempt's 990s at `num_workers=4`; later epochs dropped as low as
~425s as OS file-caching warmed up).

| | Overall AUC | Real AUC |
|---|---|---|
| 32k baseline | 0.8522 | 0.7988 |
| **64k baseline** | **0.8704 (+1.82pp)** | **0.8164 (+1.76pp)** |
| 32k `λ=0.5 50:50` | 0.8500 | 0.7391 |
| **64k `λ=0.5 50:50`** | **0.8664 (+1.64pp vs. 32k; −0.40pp vs. 64k baseline)** | **0.7736 (+3.45pp vs. 32k; −4.28pp vs. 64k baseline)** |

64k clears the continue-bar decisively for the baseline (both criteria, well past the ≥0.5pp/≥1pp
threshold). The hold-aware objective **also** gains substantially from the extra data (+1.64pp
overall, +3.45pp real AUC vs. its own 32k self) while keeping essentially the same excellent
real-hold-FCR (4.43% at 64k vs. 4.71% at 32k, own threshold — vs. the 64k baseline's 69.25% at that
same threshold). The accuracy/hold-safety gap vs. baseline is real but **not monotonically widening
with scale** as Section 6 first suggested: −3.29pp (16k) → −5.97pp (32k) → −4.28pp (64k). Scale
helps both arms; it doesn't cleanly resolve or worsen the trade-off in one consistent direction.

Per this project's own rule ("do not run 64k/128k before submission unless the 32k gain is unusually
large"), 64k was the explicit exception case, run on direct user instruction — **128k was not run
and is not implied by this result**; that would need its own explicit decision.

Artifacts: `experiments/data_scale_32k_baseline/`, `experiments/data_scale_32k_holdloss0.5_5050sampling/`,
`experiments/data_scale_64k_baseline_seed42/`, `experiments/data_scale_64k_holdloss0.5_5050sampling_seed42/`,
`scripts/build_data_scale_tier.py` (generalized to accept `--shards` for any tier),
`scripts/train_b1_32k_baseline.py`, `scripts/train_b1_32k_lambda0.5_5050.py`,
`scripts/train_b1_64k_baseline.py`, `scripts/train_b1_64k_lambda0.5_5050.py`.

## 6c. 64k matched-recall correction + 3-seed 64k confirmation (done)

The Section 6b headline hold-FCR numbers (4.43% `λ=0.5 50:50` vs. 69.25% baseline) were each read at
that model's **own** FCR=0.05-calibrated threshold — not comparable to each other, and not the
matched-complete-turn-recall method this project's keep/promotion rules actually use (Section 5b's
correction). `matched_recall_audit_64k.py` (threshold selected on **calib**'s recall curve,
evaluated on **val** only — never the same split) was run first against seed 42 alone, then against
all three seeds once the confirmation runs below finished.

Trained `data_scale_64k_baseline_seed{43,44}` and `data_scale_64k_holdloss0.5_5050sampling_seed{43,44}` (4 runs) via
generalized `--seed` flags on `train_b1_64k_baseline.py` / `train_b1_64k_lambda0.5_5050.py` (the
lambda script picks its same-seed baseline checkpoint via `SEED_BASELINE_CHECKPOINTS`). Ran
sequentially (CPU-only, 16GB RAM — running two `num_workers=6` processes concurrently risks the same
pagefile exhaustion Section 6b hit), order baseline43 → lambda43 → baseline44 → lambda44. Took ~14h
wall-clock total. All 4 converged normally (early-stopped, patience=6, no crashes).

**Per-seed overall/real AUC, implicit-incomplete FCR, calibration (Brier/ECE):**

| | overall AUC | real AUC | implicit-incomplete FCR | Brier | ECE | own threshold |
|---|---|---|---|---|---|---|
| baseline seed42 | 0.8704 | 0.8164 | 0.0713 | 0.1444 | 0.0313 | 0.8100 |
| baseline seed43 | 0.8763 | 0.7950 | 0.0753 | 0.1447 | 0.0419 | 0.8255 |
| baseline seed44 | 0.8753 | 0.8204 | 0.0674 | 0.1471 | 0.0614 | 0.8761 |
| **baseline mean±std** | **0.8740±0.0026** | **0.8106±0.0110** | **0.0713±0.0032** | **0.1454±0.0012** | **0.0449±0.0126** | |
| λ=0.5 50:50 seed42 | 0.8664 | 0.7736 | 0.0621 | 0.1989 | 0.1960 | 0.5177 |
| λ=0.5 50:50 seed43 | 0.8658 | 0.7670 | 0.0845 | 0.1674 | 0.1117 | 0.6660 |
| λ=0.5 50:50 seed44 | 0.8641 | 0.7595 | 0.0740 | 0.1587 | 0.0890 | 0.6926 |
| **λ=0.5 50:50 mean±std** | **0.8654±0.0010** | **0.7667±0.0058** | **0.0735±0.0092** | **0.1750±0.0175** | **0.1322±0.0453** | |

Overall/real AUC gap (baseline mean real AUC 0.8106 vs. λ=0.5 50:50's 0.7667, a **4.4pp real-AUC
cost**) matches the seed-42-only picture from Section 6b — not a seed-42 artifact. The λ model's own
threshold is much lower and far more seed-variable (0.52–0.69) than the baseline's (0.81–0.88),
and its Brier/ECE are consistently worse (own-threshold calibration is *not* what the temperature
scaling below will fix directly — ECE here is computed at raw sigmoid outputs, before any
temperature adjustment).

**3-seed matched-recall (calib→val) hold-FCR, mean±std:**

| | actual val recall | hold FCR (all) | hold FCR (real) | hold FCR (synthetic) | short-complete recall |
|---|---|---|---|---|---|
| baseline @ recall≈90 | 0.8970 | 0.6306±0.0137 | 0.7821±0.0181 | 0.5997±0.0186 | 0.8088 |
| λ=0.5 50:50 @ recall≈90 | 0.9078 | 0.5205±0.0091 | 0.4866±0.0358 | 0.5274±0.0037 | 0.8033 |
| baseline @ recall≈95 | 0.9574 | 0.7652±0.0174 | 0.8818±0.0125 | 0.7414±0.0233 | 0.8798 |
| λ=0.5 50:50 @ recall≈95 | 0.9599 | 0.6607±0.0092 | 0.6824±0.0136 | 0.6563±0.0083 | 0.8689 |

The 3-seed matched-recall numbers **confirm** the seed-42-only correction in Section 6b: at the same
complete-turn recall, `λ=0.5 50:50` beats the baseline on real-hold-FCR by a real, seed-stable margin
(~29.6pp at recall90: 0.487 vs. 0.782; ~19.9pp at recall95: 0.682 vs. 0.882) — larger than the
seed-42-only estimate suggested, and with acceptably small std (≤3.6pp) across seeds. Short-complete
recall is now essentially tied between the two arms (previously λ looked slightly better on seed 42
alone; on 3 seeds it's marginally worse, within noise). Net: the selection decision (favor
`λ=0.5 50:50` for hold safety) is now on solid 3-seed footing, at a real, quantified cost to overall
accuracy and calibration quality that must be disclosed alongside it.

Artifacts: `experiments/data_scale_64k_baseline_seed{43,44}/`,
`experiments/data_scale_64k_holdloss0.5_5050sampling_seed{43,44}/`, `experiments/matched_recall_audit_64k.json`.

Ground-truth-conditioned VAD-boundary errors for all 6 64k checkpoints are done via the
new `scripts/vad_boundary_diagnostic_b1_64k.py` (never run for B1 at this data scale
before) — reuses the full-val VAD-boundary rerun's boundary computation (cached once as a local
intermediate file, not itself part of this published subset, so reuse across checkpoints is not
independently re-derivable here) with Section 1's ground-truth-conditioned introduced-error metrics
rather than the superseded flip-rate gate.

**3-seed mean±std, real-audio-only slice (n≈130–137 per seed for false-completion; n=6–52 per seed
for delay — see caveat below):**

| | alt_threshold: introduced false-completion | alt_threshold: introduced delay | silero_vad: introduced false-completion | silero_vad: introduced delay |
|---|---|---|---|---|
| baseline | 0.0564±0.0033 | 0.4754±0.0053 | 0.0922±0.0161 | 0.3104±0.0427 |
| λ=0.5 50:50 | 0.0494±0.0178 | 0.4762±0.1366 | 0.0568±0.0157 | 0.4207±0.0606 |

`λ=0.5 50:50` has a lower (better) introduced-false-completion rate under both alt boundary
detectors than the baseline — consistent with the hold-aware objective generalizing its
"don't complete on a non-canonical cutoff" behavior to boundary perturbation, not just to internal
pause holds. Introduced-delay rates are roughly comparable between arms, but **the `n` behind that
number is small and uneven per seed** (introduced_delay only applies where truth=complete AND
canonical is correct — the pool that survives that filter was n=47/44/52 for baseline but only
n=22/17/6 for `λ=0.5 50:50` across seeds 42/43/44; the λ seed-44 estimate in particular rests on just
6 examples, CI [0.30, 0.90]) — this comparison is directional, not statistically solid, and should
not be over-read.

**FCR-at-recall95 (real-only, canonical boundary) confirms the AUC-based expectation**: baseline
0.52–0.57 vs. `λ=0.5 50:50` 0.61–0.64 across seeds — the baseline's higher real AUC (Section 6c
above) translates into a lower (better) FCR at matched recall on the endpoint task itself, the
mirror image of the hold-FCR result where `λ=0.5 50:50` wins. This is the accuracy/hold-safety
trade-off restated in a third, independent metric — same story, not a new finding.

Artifacts: `experiments/vad_boundary_diagnostic_data_scale_64k.json`,
`scripts/vad_boundary_diagnostic_b1_64k.py` (its cached intermediate boundary file is not part of
this published subset).

**Padding counterfactual**, via the new `scripts/padding_counterfactual_b1_64k.py`
— never run for any B1 model before (the existing `padding_counterfactual.json` files are all
A0/Whisper). Genuine data-characteristic finding, not a script bug: at `context_s=1.0`, padding is a
far rarer scenario than at A0's `context_s=4.0` — val `last_active_t` has median 6.73s, and even the
loosest possible cut (ANY left-padding at all, i.e. `speech_end_s < 1.0s`) only reaches **n=8 of
1,600** val clips (the A0-precedent 0.8-fraction cut gives just n=4). Ran anyway at n=8 for
completeness, but this is exploratory, not a statistically meaningful pass/fail the way it was for
A0 (n=118):

| | mean abs Δprob | frac Δ>0.10 | decision flip rate |
|---|---|---|---|
| baseline seed42 | 0.0779 | 0.25 | 0.00 |
| baseline seed43 | 0.1667 | 0.50 | 0.00 |
| baseline seed44 | **0.3933** | **0.75** | 0.00 |
| λ=0.5 50:50 seed42 | 0.0638 | 0.25 | 0.00 |
| λ=0.5 50:50 seed43 | 0.1027 | 0.25 | 0.00 |
| λ=0.5 50:50 seed44 | 0.0614 | 0.25 | 0.00 |

All six fail the A0-derived numeric criteria (mean abs change ≤0.02 etc.) — expected at this n, not
a real signal on its own. The one thing worth flagging despite the tiny n: baseline seed44 shows a
notably larger probability swing under padding-fill variants (0.39 mean abs change, 75% of its 8
clips shifting >0.10) than every other checkpoint. **Decision flip rate is 0.00 for all six**,
though — none of these probability swings actually crossed that checkpoint's own decision threshold
on this tiny sample, so there is no evidence of an actual behavioral problem, just a wider margin of
uncertainty for baseline seed44 specifically that a larger padding-sensitive sample (not available
in this val split) would be needed to resolve.

Artifacts: `experiments/padding_counterfactual_data_scale_64k.json`,
`scripts/padding_counterfactual_b1_64k.py`.

With this, the "padding/VAD/real/implicit/hold" evaluation suite is now complete for the 64k
checkpoints: VAD-boundary (above), padding (above), real/implicit slices (already in each
checkpoint's own `metrics.json`, reported in the AUC/calibration table earlier in this section),
and hold-FCR (Section 6c's matched-recall audit). Still outstanding: explicit final
temperature-scaling/threshold calibration on calib only (next section), and freeze →
official-test evaluation → export (final section).

## 6f. Temperature scaling + final threshold on calibration only

Distinct from each run's own per-checkpoint `calibrate_threshold` (already picks a threshold on
calib for target FCR≤0.05, but off *raw*, uncalibrated sigmoid outputs) — this fits a single scalar
temperature `T` per checkpoint (Guo et al. 2017: minimize BCE(logit/T, label) on **calib only**,
never val/test) before re-selecting the threshold, motivated directly by Section 6c's finding that
`λ=0.5 50:50`'s raw ECE is much worse than the baseline's (up to 0.196).

New `scripts/calibrate_temperature_scaling_64k.py`. Verified explicitly, not assumed: temperature
scaling is a strictly monotonic transform of the logit (T>0), so it cannot change AUC or any
recall/FCR at a matched operating point — only the probability VALUE at a given decision changes.
Confirmed on every one of the 6 checkpoints: val recall at the (raw-threshold, raw-prob) operating
point exactly equals val recall at the (temp-threshold, temp-scaled-prob) operating point.

| | T | threshold (raw → temp) | val ECE (raw → temp) | val Brier (raw → temp) |
|---|---|---|---|---|
| baseline seed42 | 0.975 | 0.8100 → 0.8157 | 0.0313 → 0.0380 | 0.1444 → 0.1445 |
| baseline seed43 | 1.107 | 0.8255 → 0.8028 | 0.0419 → 0.0301 | 0.1447 → 0.1437 |
| baseline seed44 | 1.330 | 0.8761 → 0.8131 | 0.0614 → 0.0273 | 0.1471 → 0.1429 |
| λ=0.5 50:50 seed42 | **2.150** | 0.5177 → 0.5082 | **0.1960 → 0.1658** | 0.1989 → 0.1842 |
| λ=0.5 50:50 seed43 | 1.514 | 0.6660 → 0.6120 | 0.1117 → 0.1061 | 0.1674 → 0.1629 |
| λ=0.5 50:50 seed44 | 1.208 | 0.6926 → 0.6620 | 0.0890 → 0.0907 | 0.1587 → 0.1583 |

Temperature scaling meaningfully improves calibration quality for the worst offenders (baseline
seed44: ECE 0.061→0.027; `λ=0.5 50:50` seed42: ECE 0.196→0.166), confirming the hold-aware
objective's own logits are genuinely less well-calibrated (needs a much larger T=1.2–2.15 to correct
vs. the baseline's near-1.0 values) rather than just noisier. Two checkpoints (baseline seed42,
`λ=0.5 50:50` seed44) got very slightly *worse* val ECE post-scaling — expected and acceptable: T is
fit on calib only, and calib/val aren't identical distributions, so a calib-optimal T need not be
exactly val-optimal; the direction and magnitude of the fix is what matters, not that every single
seed improves.

Artifacts: `experiments/temperature_scaling_64k.json`, `scripts/calibrate_temperature_scaling_64k.py`.

Only one step remains: freeze one checkpoint, evaluate once against the official 31,527-row HF test
set, export, and submit. **This requires an explicit decision this project hadn't made yet: which
single checkpoint (arm + seed) to freeze** — the per-finalist convention means the official test set
gets touched exactly once, so this is not a call to make casually. Flagged for the user rather than
picked unilaterally.

## 6g. Freeze, official test evaluation, and export (FINAL)

**Frozen per explicit user decision**: `data_scale_64k_holdloss0.5_5050sampling_seed43`, temperature `T=1.5142`
(user said 1.514; stored value used verbatim), calibrated probability threshold `0.612`,
speech-aligned input contract, `context_s=1.0`. Checkpoint sha256:
`ddaf7a8ea95b6675022920b68b95e7a1f8202ab403c3e7e11e08dc5f0892694f`. Full manifest — architecture
config, every metric computed against this checkpoint this session (training full_report,
matched-recall audit, VAD-boundary diagnostic, padding counterfactual, temperature scaling detail) —
recorded in `experiments/data_scale_64k_holdloss0.5_5050sampling_seed43/FROZEN_MANIFEST.json` **before** the test
set was touched. The 64k baseline is kept only as the accuracy ablation reference (Section 6c);
per the user's instruction it was **not** evaluated on the official test.

**Official test evaluation** (new `scripts/run_official_test_evaluation.py`, three phases —
`fetch_features` / `infer` / `report` — so a bug in aggregation never forces re-running inference,
and a bug in feature extraction never forces re-downloading the ~4.84GB test set):
- Fetched all 10 parquet shards of `pipecat-ai/smart-turn-data-v3.2-test` (31,527 rows, 0 decode
  errors). No cached `last_active_t` exists for this set (unlike D2's train/val/calib), so the
  canonical v0 boundary was computed FRESH via `tinyturn.boundary.estimate_speech_end` — exactly the
  "recompute" path `tinyturn/dataset.py`'s docstring had anticipated for "a future official-test-set
  loader." No ASR: `endfiller`/`midfiller` ship natively on synthetic rows in the raw HF metadata
  (same convention D2 relies on for training), so nothing the standard evaluation needs was lost by
  skipping transcription.
- **Caught and fixed a real bug before it reached the test set**: a hand-derived frame-count formula
  (`1 + (16000 - frame_length) // hop_length`) gave 98 frames; the actual `librosa` `center=False`
  frame count depends on `n_fft` (512), not `win_length` (400), and is really 97 — caught by checking
  against an actual cached training feature array before running anything against test data, not
  assumed correct.
- Ran a smoke test (5 real test-set rows, full pipeline including a frozen-model forward pass) to
  confirm shapes/no exceptions — **not saved as results**, purely a code-correctness check — before
  the one, single official inference pass (12 seconds for all 31,527 clips). The `infer` phase
  refuses to run a second time if its output file already exists, enforcing the touched-once
  discipline at the code level, not just by convention.

**Official test results** (temperature-scaled probability, threshold 0.612):

| slice | n | AUC | recall | FCR | fcr_at_recall95 |
|---|---|---|---|---|---|
| overall | 31,527 | 0.8607 | 0.4576 | 0.0726 | 0.461 |
| real (eng+spa) | 5,863 | 0.7814 | 0.1105 | 0.0219 | 0.600 |
| real_eng | — | 0.788 | 0.114 | 0.023 | 0.584 |
| real_spa | 496 | 0.7196 | 0.0691 | 0.008 | 0.720 |
| synthetic_only_languages | 21,924 | 0.8849 | 0.5486 | 0.0844 | 0.400 |
| implicit_incomplete (synthetic only, see caveat) | 4,131 | — | — | FCR 0.1605 | — |

per_source_macro_auc: 0.8123. Calibration: Brier 0.168, ECE 0.112 (val for this same checkpoint was
Brier 0.163 / ECE 0.106 — very close; small drift is expected since T and the threshold were fit on
calib, never on test). **These test numbers are consistent with (not wildly different from) the
val-set numbers reported throughout Section 6c/6f for this exact checkpoint** — overall AUC 0.861
(test) vs. 0.866 (val), real AUC 0.781 (test) vs. 0.767 (val, actually slightly higher on test) — no
sign of overfitting to val/calib.

**Known, disclosed limitation**: `implicit_incomplete` ground truth needs `endfiller`, which is 100%
populated for synthetic test rows but 100% null for the 5,863 real-audio test rows (checked directly,
same pattern D2's real-audio subset has). Running ASR against 5,863 fresh clips to fill this in was
judged out of scope for "run the official test once" and wasn't requested — the slice above covers
synthetic clips only; real-audio rows are excluded, not silently folded in as "not implicit."

**Export**: `scripts/export_frozen_checkpoint_onnx.py` — FP32 ONNX (474,165 bytes) and INT8 dynamic-
quantized ONNX (208,596 bytes, 56% smaller) from the frozen checkpoint, independent of the test set.
Parity check against a real val clip: PyTorch vs. FP32-ONNX logits match to 1e-6; INT8 introduces a
small, expected drift (0.008 logit units). **Counterintuitive but measured, not assumed**: INT8 was
*slower* than FP32 on this CPU (17.8ms vs. 11.3ms p50, full pipeline including feature extraction) —
likely dynamic-quantization dequant overhead dominating for a model this tiny (7,045,264 MACs,
109,979 parameters); reported as-measured rather than assuming smaller means faster.

Artifacts: `experiments/data_scale_64k_holdloss0.5_5050sampling_seed43/FROZEN_MANIFEST.json`,
`data_cache/official_test_features.npz`, `data_cache/official_test_metadata.parquet`,
`experiments/official_test_per_clip_results.parquet`, `experiments/official_test_report.json`,
`experiments/data_scale_64k_holdloss0.5_5050sampling_seed43/{model.onnx,model_int8.onnx,export_manifest.json}`,
`scripts/run_official_test_evaluation.py`, `scripts/export_frozen_checkpoint_onnx.py`.

**This closes out the project. No further model changes are made per the user's explicit
instruction — the checkpoint, temperature, and threshold above are final.**

## 7. Artifacts

- `experiments/metric_audit_ground_truth_conditioned.json` — Section 1.
- `data_cache/teacher_logits_a0_boundary_robust_train.parquet` — Section 2.
- `experiments/distillation_isolation_control/`, `experiments/distillation_canonical_boundary_teacher/`,
  `experiments/distillation_mean3boundary_teacher/` — Sections 2 and 2b (dropped; the isolation
  control is the D0 control run).
- `experiments/pairwise_ranking_singleseed/`, `experiments/pairwise_ranking_seed42/`,
  `experiments/pairwise_ranking_seed43/`, `experiments/pairwise_ranking_seed44/` — Section 3.
- `experiments/pause_event_sampling_comparison/pause_events_holdloss0.5_proportional_seed42/`,
  `.../pause_events_holdloss0.5_5050sampling_seed42/` — Section 5 protocol-matched retrains.
- `experiments/matched_recall_audit_calib_then_val.json`,
  `experiments/matched_recall_audit_seed42.png` — Section 5b.
- New code: `scripts/ground_truth_conditioned_metric_audit.py`, `scripts/precompute_teacher_logits.py`,
  `tinyturn/train_distill.py`, `scripts/train_distillation_ablation.py`, `tinyturn/train_ranking.py`,
  `scripts/train_pairwise_ranking.py`, `scripts/train_b1_lambda0.5_seed42_plateau_protocol_fix.py`,
  `scripts/matched_recall_audit.py`.
  `tinyturn/dataset.py` and `tinyturn/pause_events.py` gained `augment_boundaries`/`teacher_logit`
  support.

## 8. Summary

- **Distillation isolation control (D0)**: see Section 2b. Confirms the shipping decision
  (distillation still dropped) but corrects the scientific attribution (boundary augmentation, not
  the teacher signal, is the source of the real-AUC cost; distillation partially offsets it).
- **Data scaling**: 32k was a decisive gain for the baseline (+5.52pp real AUC; the hold-aware
  objective's real-AUC cost persists and widens in relative terms at 32k, Section 6); 64k was run as
  the explicit large-gain exception and confirmed the same pattern at 3 seeds (Sections 6b–6c).
- **Everything above this point is final** — see Section 6g for the frozen checkpoint and the single
  official test-set evaluation.
