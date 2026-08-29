# TinyTurn — Experiment Log

This is a chronological catalog of every experiment run in the TinyTurn project, from initial
architecture selection through the final official test-set result. It exists so a reader can trace
*why* the finalist checkpoint looks the way it does — every design decision along the way, and the
actual number behind each one.

- For the deep dive on the final phase — the metric audit, distillation ablation, ranking
  experiment, finalist selection, 3-seed confirmation, temperature scaling, and the official test
  result — see [`docs/RESULTS.md`](RESULTS.md). This document summarizes up through where
  `RESULTS.md` picks up (Section 6 onward here overlaps with its early sections only for context)
  and does not re-derive its numbers.
- For the single headline result, see the [README](../README.md).

Every table below cites a real `config.json`/`metrics.json` (or nearest equivalent) under
`experiments/`; paths are given so any number can be independently re-checked. All AUCs are on the
15,998-clip working set's fixed val split (n=1,600) unless stated otherwise. "Real" means real
(non-synthetic) audio only, English + Spanish combined, n=282 in every 16k/32k/64k-tier val
evaluation (the val split itself never grows across data-scale tiers — only the *train* split does).

---

## 1. Architecture selection: B0 / B1 / B1-f0 / A0 (all at 4.0s context)

The first question: is a from-scratch tiny model (log-mel, optionally + a pitch/energy
"trajectory" branch) competitive with a small pretrained speech model (Whisper-Tiny, fully
fine-tuned)? All four trained under an identical protocol — 5 fixed epochs for the tiny models
(`scripts/train_mel_trajectory_model.py`), 2 fixed epochs for Whisper-Tiny
(`scripts/train_whisper_model.py`) — same splits, same 200ms-postroll input contract (later revised,
see Section 3).

| Experiment | Key config | Overall AUC | Real AUC | Params | p50 latency | ECE | Outcome |
|---|---|---|---|---|---|---|---|
| [`mel_only_baseline`](../experiments/mel_only_baseline/) | mel only, no trajectory branch | 0.8119 | 0.7010 | 100,594 | 8.3ms | 0.126 | Baseline tiny model |
| [`mel_trajectory_baseline`](../experiments/mel_trajectory_baseline/) | + pitch/energy trajectory branch | 0.8148 | 0.7568 | 109,979 | 32.7ms | 0.035 | **Tiny finalist** — trajectory branch adds +5.6pp real AUC over B0 and sharply improves calibration |
| [`mel_trajectory_with_f0`](../experiments/mel_trajectory_with_f0/) | + explicit F0 (pitch) channel | 0.8262 | 0.7482 | 110,003 | 324ms | 0.122 | Rejected — F0's gain concentrates in the easier synthetic slice; real AUC is actually *worse* than B1's, at ~10x the latency |
| [`whisper_tiny_baseline`](../experiments/whisper_tiny_baseline/) | Whisper-Tiny, full fine-tune | 0.9502 | 0.9371 | 8,307,202 | 29.9ms | 0.023 | Dominates every tiny variant by 12–14pp overall AUC (18–24pp real) at comparable latency |

**Reading it**: B1 (mel + trajectory, no F0) is the tiny-model finalist — its trajectory branch
helps almost entirely on real audio, where it matters most. B1-f0's pitch channel doesn't earn its
10x latency cost. But the number that actually shapes the rest of the project is A0 vs. B1: Whisper
beats every tiny variant by roughly 12–14pp overall AUC and 18–24pp real-audio AUC, at a latency
cost that's nearly a wash (29.9ms vs. B1's 32.7ms) — its log-mel front end is cheap and its exported
graph well-optimized, while B1's raw-DSP trajectory features are themselves expensive. Size (75x)
and compute (54x MACs) are A0's real cost, not inference latency. This gap directly motivates
Sections 4 (can A0 qualify as a distillation teacher?) and 7 (does distilling from A0 into B1 help?)
below.

---

## 2. Context-length probing

### 2a. Handcrafted probe (C0) — superseded by 2b

Before training any model, a real-audio-only handcrafted-feature probe
(`scripts/context_probe_real_only.py`) swept context length 0.5–4s to get a cheap initial read on
how much history the model needs:

| Context | Real (all) AUC | Real English AUC | Real Spanish AUC |
|---|---|---|---|
| 0.5s | 0.553 | 0.580 | 0.607 |
| 1.0s | 0.539 | 0.542 | 0.618 |
| 2.0s | 0.650 | 0.666 | 0.571 |
| 4.0s | 0.678 | 0.692 | 0.596 |

Source: [`experiments/context_probe_handcrafted/context_probe_results.csv`](../experiments/context_probe_handcrafted/),
config at the same path. Monotonic, not-yet-plateaued improvement with context length → chose
`N=4.0s` as the *provisional* default. As Section 2b shows, this conclusion did not survive contact
with an actual trained model.

### 2b. Learned-model ablation (C1) — the actual decision

Ran the trained B1 at 1s/2s/4s and A0 at 2s/4s/8s (`scripts/context_length_ablation.py`; each
individual run also has its own experiment directory).

| Experiment | Context | Overall AUC | Real AUC | p50 latency | ECE |
|---|---|---|---|---|---|
| [`context_ablation_mel_trajectory_1s`](../experiments/context_ablation_mel_trajectory_1s/) | B1 @ 1.0s | 0.8149 | 0.7428 | 10.2ms | 0.035 |
| [`context_ablation_mel_trajectory_2s`](../experiments/context_ablation_mel_trajectory_2s/) | B1 @ 2.0s | 0.8009 | 0.7256 | 15.9ms | 0.156 |
| `mel_trajectory_baseline` (Section 1) | B1 @ 4.0s | 0.8148 | 0.7568 | 32.7ms | 0.035 |
| [`context_ablation_whisper_tiny_2s`](../experiments/context_ablation_whisper_tiny_2s/) | A0 @ 2.0s | 0.9391 | 0.9185 | 14.5ms | 0.038 |
| `whisper_tiny_baseline` (Section 1) | A0 @ 4.0s | 0.9502 | 0.9371 | 29.9ms | 0.023 |
| [`context_ablation_whisper_tiny_8s`](../experiments/context_ablation_whisper_tiny_8s/) | A0 @ 8.0s | 0.9527 | 0.9399 | 52.6ms | 0.023 |

Full write-up: [`experiments/context_ablation_summary/summary.md`](../experiments/context_ablation_summary/summary.md).

**B1 (tiny model)**: AUC is flat within noise across 1–4s (single run per length, heavily
overlapping confidence intervals; the 2.0s dip looks like training-seed variance, not a real
context effect). Latency scales cleanly and substantially with context (10ms → 33ms), driven by the
trajectory branch's raw-DSP cost. **Decision: N=1.0s for B1** — no accuracy evidence favors longer
context, at ~3x lower cost than C0's provisional 4.0s. This reverses C0's own conclusion.

**A0 (Whisper)**: real gains up to 4s, sharply diminishing after (2s→4s: +1.1pp overall/+1.9pp real
for ~2x latency; 4s→8s: +0.3pp/+0.3pp, within noise, for another ~1.8x latency). **Decision: N=4.0s
for A0** stands, both for its own diminishing-returns curve and to match B1's context for any future
distillation.

---

## 3. Input-contract correction and convergence checks

A multi-part audit (documented internally as steps 8a–8d, 8h) found and fixed real bugs in how
padding, attention masking, and epoch budgets were handled, then re-ran the affected models.

**What changed**: the input contract was tightened to exactly *N* seconds ending at the detected
speech boundary (no baked-in 200ms post-roll — that's now a runtime policy decision, not a training
input); Whisper's own encoder was found to never apply an attention mask at all (`attention_mask` is
accepted but unused upstream); and a frame-mask misalignment (`center=False` mask reused against a
`center=True` feature extractor) was found to zero out the ~2 frames nearest the speech boundary —
exactly the most diagnostically important region — on every single A0 example.

| Experiment | What it is | Overall AUC | Real AUC | Threshold | Outcome |
|---|---|---|---|---|---|
| [`whisper_tiny_speech_aligned_contract_mask_bug`](../experiments/whisper_tiny_speech_aligned_contract_mask_bug/) | A0 retrained under the corrected contract, but **before** the frame-mask fix | 0.9387 | 0.9133 | 0.8792 | Superseded — kept as a documented negative result (see its `NOTE.md`); F1/recall (0.742→0.794, 0.620→0.702) moved much more than AUC once the mask bug was fixed |
| [`whisper_tiny_speech_aligned_contract`](../experiments/whisper_tiny_speech_aligned_contract/) | A0 retrained, contract + mask fix both applied | **0.9378** | **0.9122** | 0.8437 | Current canonical A0 checkpoint |
| [`mel_trajectory_1s_speech_aligned_contract`](../experiments/mel_trajectory_1s_speech_aligned_contract/) | B1@1s retrained under the corrected contract | **0.8279** | **0.7402** | 0.8004 | Current canonical B1@1s baseline — everything downstream (Sections 4–8) builds on this checkpoint |
| [`whisper_tiny_4s_earlystopped_longrun`](../experiments/whisper_tiny_4s_earlystopped_longrun/) | A0@4s, early-stopped (max 10ep, patience 3) instead of fixed-2-epoch | 0.9372 | 0.9058 | 0.8733 | best_epoch 3/6 — essentially ties the fixed-2-epoch checkpoint, see significance test below |
| [`mel_trajectory_1s_earlystopped_longrun`](../experiments/mel_trajectory_1s_earlystopped_longrun/) | B1@1s, early-stopped (max 40ep, patience 6, plateau LR) | **0.8316** | 0.7409 | 0.8908 | best_epoch 6/12, best_val_auc 0.8316 — **+0.37pp over the fixed-5-epoch checkpoint** (peaked epoch 3 at 0.8279, had already regressed by epoch 5); this early-stopped/plateau protocol becomes standard for every later B1 run |

**Was A0's original fixed-2-epoch checkpoint undertrained?** A paired bootstrap test between it and
the Kaggle early-stopped retrain, same 1,600-clip val set, n_boot=2000: observed AUC diff
+0.0006, 95% CI **[-0.0045, +0.0058]**, p=0.811 — not distinguishable from noise
([`experiments/convergence_significance_whisper_tiny.json`](../experiments/convergence_significance_whisper_tiny.json), via
`scripts/convergence_check_a0.py`). The two protocols are statistically equivalent here; A0 was not
meaningfully undertrained. (A companion A0@2s early-stopped run was also trained on the same Kaggle
pass — reported in project prose as overall AUC ≈0.9205 / real AUC ≈0.8892 at roughly 2.6x lower
latency than A0@4s — but its checkpoint was not saved as its own experiment directory in this
curated subset, so those two figures are not independently re-derivable from a committed artifact
here and should be treated as directional.)

B1's own convergence check ([`experiments/convergence_check_mel_trajectory.json`](../experiments/convergence_check_mel_trajectory.json),
via `scripts/convergence_check_b1.py`): fixed-epoch protocol peaked at epoch 3 (val AUC 0.8279) and
had *already* passed its peak by epoch 5 (0.8260); the longer early-stopped run found a new peak at
epoch 6 (0.8316, stopping at epoch 12) — the same +0.37pp figure in the table above, confirmed from
the raw per-epoch history.

---

## 4. Teacher-qualification track: is A0 viable as a distillation teacher?

Given Section 1's large A0-vs-B1 gap, a natural next question is whether A0 could serve as a
*teacher* for distilling into B1 (Section 7). Qualifying it required two robustness gates:
padding-invariance and boundary-perturbation (VAD) robustness.

| Check | Experiment / artifact | Result | Verdict |
|---|---|---|---|
| Padding counterfactual (8e) | [`whisper_tiny_speech_aligned_contract/padding_counterfactual.json`](../experiments/whisper_tiny_speech_aligned_contract/) (n=118, `scripts/padding_counterfactual_a0.py`) | mean abs Δprob 0.0, frac Δ>0.10 0.0, flip rate 0.0 | Passes on point estimates; under a later CI-gated re-review the two proportion gates came back **inconclusive** (n=118 too small for a 1%-tight bound even at a perfect empirical result) |
| Prefix-context reliance (8e-extended, exploratory) | `scripts/prefix_context_reliance_diagnostic.py` (n=1,419) | at 25% of context kept: mean \|Δprob\| 12.7%, flip rate 14.9%; real clips shift more than synthetic (17.3% vs 11.9% mean \|Δprob\| at 25% kept) | Exploratory only, not a gate — amount-vs-content correlation genuinely ambiguous (\|r\|≤0.09) |
| VAD-boundary, pilot (8f, n=43) | `experiments/vad_boundary_diagnostic_pilot.json`, via `scripts/vad_boundary_diagnostic_pilot.py` | frac Δprob>0.20 and flip rate both **borderline** (95% CIs straddle their thresholds) | Inconclusive at this n — real-audio n was only 7 |
| VAD-boundary, full rerun (8f, n=1,600) | [`experiments/vad_boundary_diagnostic_full_val.json`](../experiments/vad_boundary_diagnostic_full_val.json), via `scripts/vad_boundary_diagnostic_full_val.py` | vs. alt-threshold: flip rate 10.5% [9.1,12.1] — **decisive fail** (bound ≤5%); vs. Silero: frac Δ>0.20 11.6% [10.1,13.3] and flip rate 12.3% [10.7,13.9] — **both decisive fail** | Resolves 8f's earlier "borderline" status — a clear fail, not just sharper noise |
| Teacher qualification, original (8g) | `whisper_tiny_speech_aligned_contract/qualification_original.json`, via `scripts/qualify_teacher_a0.py` | FAIL, provisional (rested on 8f's inconclusive pilot) | Provisional FAIL |
| Teacher qualification, CI-gated (8g) | `whisper_tiny_speech_aligned_contract/qualification_ci_gated.json`, via `scripts/qualify_teacher_a0_ci_gated.py` | direction-specific safety-critical flip rate: 10.8–14.4% (alt-threshold) / 14.4–19.4% (Silero) vs. a ≤2% bound | **Decisive FAIL** — 5–10x over bound, worse on real audio |
| 8g remediation: boundary-robust retrain | [`experiments/whisper_tiny_boundary_robust_retrain/`](../experiments/whisper_tiny_boundary_robust_retrain/) (train-time boundary augmentation, `scripts/retrain_a0_boundary_robust.py`; precompute via `scripts/precompute_train_boundary_augmentation.py`) | overall AUC 0.9391 / real AUC 0.9068 (ties or slightly beats the unaugmented checkpoint, no accuracy cost); safety-critical flip rate improved ~25–35% (e.g. alt-threshold 10.8%→8.0%) | Still **FAIL** — every safety-critical gate remains 4–6x over its 2% bound, down from ~5–9x |
| Ground-truth-conditioned metric audit | [`experiments/metric_audit_ground_truth_conditioned.json`](../experiments/metric_audit_ground_truth_conditioned.json), via `scripts/ground_truth_conditioned_metric_audit.py` | introduced-false-completion rate (the corrected, ground-truth-aware version of the flip-rate gate): 1.06% (alt-threshold) / 2.13% (Silero) for `whisper_tiny_boundary_robust_retrain`, vs. 3.21%/3.61% for the original checkpoint | Meaningfully softens the picture — now **inconclusive**, not decisive-fail, though this doesn't reverse 8g's FAIL verdict (inconclusive still blocks) |

**Reading it**: A0's original flip-rate gate conflated genuine safety regressions with cases where
the alternative boundary was simply *correcting* an error the canonical detector made. Boundary
augmentation is a real, measured, no-cost improvement — but even combined with the corrected
(ground-truth-conditioned) metric, A0 does not clear qualification as a Step-10 distillation
teacher. It gets used anyway as an *offline* teacher for the distillation ablation in Section 7,
specifically to test whether its signal still helps despite failing its own gate — see that
section's result.

---

## 5. Pause-event / hold-loss objective (the "P1" family)

The core problem this line of experiments addresses: without any training signal about
mid-utterance pauses, B1 calls a pause "complete" far too often. Internal-pause continuation events
(audio cut partway into a qualifying pause, labeled incomplete) are added to the training set to fix
this — at some cost to main-task accuracy. This family went through several iterations as the
protocol and sampling strategy were refined.

### 5a. Original recipe: plain, unweighted blend

[`experiments/pause_events_original_recipe/`](../experiments/pause_events_original_recipe/) (pre-input-contract-fix, config
`use_trajectory=true`, `context_s=1.0`, no `lambda_hold`): 17,334 pause events unweighted-blended
with B1@1s's 12,797 final clips (`train_set_size` in its `metrics.json`).

| Metric | Baseline (B1@1s, pre-fix) | P1 (+pause events) |
|---|---|---|
| Overall AUC | 0.8149 (`context_ablation_mel_trajectory_1s`) | 0.7968 |
| Real AUC | 0.7428 | 0.6783 |
| FCR at holds, all (own threshold) | 18.2% | **5.0%** |
| FCR at holds, real | 12.5% | **3.3%** |

Source for the FCR-at-holds pair:
[`experiments/pause_events_original_recipe/fcr_at_holds_fair_compare.json`](../experiments/pause_events_original_recipe/) —
each model evaluated at its *own* calibrated threshold (an earlier pass had incorrectly compared the
baseline at P1's calibrated threshold instead of its own, understating the baseline's problem as
40.9% instead of the corrected 18.2% — the table above is the corrected version). **A genuine
trade-off, not a win**: pause events cut false-completion-at-holds by ~3.6x, but cost 1.8pp overall /
6.5pp real-audio AUC.

### 5b. Contract-corrected plain P1 + P1a/P1b λ sweep (fixed 5-epoch, later found undertrained)

P1a (mean-normalized weighted hold loss, `lambda_hold` ∈ {0.1, 0.25, 0.5}) + P1b (at most one pause
event per clip per epoch, proportional real/synthetic sampling), via
`scripts/train_pause_event_refinement.py`:

| Experiment | λ | Overall AUC | Real AUC | FCR@holds (own threshold), all |
|---|---|---|---|---|
| [`pause_events_contract_fixed_plain`](../experiments/pause_events_contract_fixed_plain/) | plain (Section 5a recipe, contract-fixed) | 0.8107 | 0.6731 | 9.5% |
| [`pause_events_holdloss0.1_5epoch`](../experiments/pause_events_holdloss0.1_5epoch/) | 0.1 | 0.8360 | 0.7289 | 16.1% |
| [`pause_events_holdloss0.25_5epoch`](../experiments/pause_events_holdloss0.25_5epoch/) | 0.25 | 0.8329 | 0.7187 | 11.5% |
| [`pause_events_holdloss0.5_5epoch`](../experiments/pause_events_holdloss0.5_5epoch/) | 0.5 | 0.8321 | 0.7109 | 10.9% |

All four runs (this plain-P1 retrain plus all three λ variants) were **still improving at their
final (5th) epoch** — this fixed-epoch protocol was later shown (Section 3) not to match this
architecture's actual convergence rate, so both the absolute numbers and the λ ranking here are
flagged as provisional. Matched-threshold curves (recall on complete turns vs. FCR at holds, swept
across threshold rather than compared at one point) are in
[`experiments/pause_events_matched_threshold_curves/matched_threshold_results.json`](../experiments/pause_events_matched_threshold_curves/).
Every P1a+P1b variant protects main-task and real-audio AUC far better than the plain blend, and all
three actually beat the no-pause-event baseline on main AUC — a genuinely useful refinement, even
before the undertraining is fixed.

### 5c. Controlled early-stopped rerun + 3-arm real/synthetic sampling comparison

Re-ran under the validated early-stopping protocol (max 40 epochs, patience 6, plateau LR), and
added a 3-way comparison of how pause events are sampled across the real/synthetic mix, via
`scripts/train_pause_sampling_comparison.py` (results grouped under
`experiments/pause_event_sampling_comparison/`, alongside the seed-42 protocol-matched retrains from
Section 5d below):

| Experiment | Sampling | Overall AUC | Real AUC | Brier / ECE | FCR@holds@recall95, all / real |
|---|---|---|---|---|---|
| [`baseline_no_pause_events_seed42`](../experiments/pause_event_sampling_comparison/baseline_no_pause_events_seed42/) | no pause events | 0.8316 | 0.7436 | 0.167/0.056 | 77.7% / 87.0% |
| [`pause_events_plain_recipe`](../experiments/pause_event_sampling_comparison/pause_events_plain_recipe/) | plain (Section 5a recipe) | 0.8075 | 0.6477 | 0.225/**0.200** | 71.7% / 77.3% |
| [`pause_events_holdloss0.25_proportional`](../experiments/pause_event_sampling_comparison/pause_events_holdloss0.25_proportional/) | λ=0.25, proportional | 0.8361 | 0.7254 | 0.166/0.046 | 75.7% / 79.5% |
| [`pause_events_holdloss0.5_proportional`](../experiments/pause_event_sampling_comparison/pause_events_holdloss0.5_proportional/) | λ=0.5, proportional ("all") | 0.8336 | 0.7061 | 0.178/0.103 | **71.6%** / 76.2% |
| [`pause_events_holdloss0.5_5050sampling`](../experiments/pause_event_sampling_comparison/pause_events_holdloss0.5_5050sampling/) | λ=0.5, 50:50 real/synth | 0.8241 | 0.7107 | 0.192/0.116 | 75.3% / 68.1% |
| [`pause_events_holdloss0.5_realonly_sampling`](../experiments/pause_event_sampling_comparison/pause_events_holdloss0.5_realonly_sampling/) | λ=0.5, real-only | 0.8119 | 0.6417 | 0.193/0.112 | 78.3% / **64.3%** |

**Reading it**: `λ=0.5 (all)` now essentially ties plain P1's aggregate hold-suppression while
paying far less AUC cost — a materially stronger result than the undertrained 5c-predecessor run
suggested. The natural pause-event pool is heavily synthetic; biasing sampling toward real data
(`real_only`) or balancing it (`50:50`) substantially improves real-hold FCR suppression, at a rising
main-task AUC cost as more synthetic supervision is excluded. `real_only` is not a free lunch — its
calibration is the worst of the three λ=0.5 arms and its synthetic-hold suppression regresses below
baseline. No single arm dominates; this maps out a genuine 3-way trade (overall balance vs.
real-hold suppression vs. calibration), carried into 3-seed confirmation next.

### 5d. Protocol-matched retrain + 3-seed promotion

A protocol mismatch was caught before 3-seed confirmation: the seed-42 checkpoints for `λ=0.5 all`
and `λ=0.5 50:50` above were flagged as needing a matching-protocol retrain to be safely averaged
with the seed-43/44 runs below (which already used the validated plateau protocol). Retrained via
`scripts/train_b1_lambda0.5_seed42_plateau_protocol_fix.py`:

| Experiment | Arm | Seed | Overall AUC | Real AUC |
|---|---|---|---|---|
| [`pause_events_holdloss0.5_proportional_seed42`](../experiments/pause_event_sampling_comparison/pause_events_holdloss0.5_proportional_seed42/) | λ=0.5 all | 42 | 0.8309 | 0.7030 |
| [`pause_events_holdloss0.5_proportional_seed43`](../experiments/pause_event_sampling_comparison/pause_events_holdloss0.5_proportional_seed43/) | λ=0.5 all | 43 | 0.8362 | 0.7003 |
| [`pause_events_holdloss0.5_proportional_seed44`](../experiments/pause_event_sampling_comparison/pause_events_holdloss0.5_proportional_seed44/) | λ=0.5 all | 44 | 0.8265 | 0.6962 |
| [`pause_events_holdloss0.5_5050sampling_seed42`](../experiments/pause_event_sampling_comparison/pause_events_holdloss0.5_5050sampling_seed42/) | λ=0.5 50:50 | 42 | 0.8261 | 0.7080 |
| [`pause_events_holdloss0.5_5050sampling_seed43`](../experiments/pause_event_sampling_comparison/pause_events_holdloss0.5_5050sampling_seed43/) | λ=0.5 50:50 | 43 | 0.8329 | 0.7031 |
| [`pause_events_holdloss0.5_5050sampling_seed44`](../experiments/pause_event_sampling_comparison/pause_events_holdloss0.5_5050sampling_seed44/) | λ=0.5 50:50 | 44 | 0.8184 | 0.6776 |
| [`baseline_no_pause_events_seed43`](../experiments/pause_event_sampling_comparison/baseline_no_pause_events_seed43/) | no-pause baseline | 43 | 0.8339 | 0.7533 |
| [`baseline_no_pause_events_seed44`](../experiments/pause_event_sampling_comparison/baseline_no_pause_events_seed44/) | no-pause baseline | 44 | 0.8290 | 0.7666 |

Averaging these three seeds per arm reproduces `docs/RESULTS.md`'s own 3-seed table exactly (e.g.
`λ=0.5 all` mean overall AUC 0.8312±0.004, real AUC 0.6998±0.003) — see that document for the full
comparison against the pairwise-ranking arm (Section 8 below) and for the finalist decision.

A per-seed significance check
([`3seed_significance.json`](../experiments/pause_event_sampling_comparison/3seed_significance.json))
confirms the real-AUC cost of the hold-aware objective is a real effect, not seed noise, in 5 of 6
arm×seed combinations (paired bootstrap vs. the matching-seed baseline, n_real=282): `λ=0.5 all`
is distinguishable from baseline at all three seeds (p=0.027, 0.025, 0.003); `λ=0.5 50:50` is
distinguishable at seeds 43/44 (p=0.026, p<0.001) but **not** at seed 42 (diff −3.3pp, 95% CI
[−7.8pp, +1.5pp], p=0.182) — that one seed's real-AUC cost is not statistically distinguishable from
noise, though the other five confirm a genuine, real-audio-specific accuracy cost from the
hold-aware training objective.

---

## 6. Data scaling: 16k → 32k → 64k

Tests whether more training data (fetched as whole additional parquet shards from
`pipecat-ai/smart-turn-data-v3.2-train`, decoded and feature-extracted via
`scripts/build_data_scale_tier.py`) closes the accuracy/hold-safety trade-off from Section 5, for
both the plain baseline and the chosen hold-aware objective (`λ=0.5 50:50`). Val/calib splits are
held fixed at every tier — only the train split grows.

| Experiment | Tier | Train clips | Overall AUC | Real AUC |
|---|---|---|---|---|
| [`baseline_no_pause_events_seed42`](../experiments/pause_event_sampling_comparison/baseline_no_pause_events_seed42/) (≈`mel_trajectory_1s_earlystopped_longrun`, Section 3) | 16k baseline | 12,797 | 0.8316 | 0.7436 |
| [`data_scale_32k_baseline`](../experiments/data_scale_32k_baseline/) | 32k baseline | — | **0.8522** (+2.06pp) | **0.7988** (+5.52pp) |
| [`data_scale_64k_baseline_seed42`](../experiments/data_scale_64k_baseline_seed42/) | 64k baseline, seed 42 | — | **0.8704** (+1.82pp vs 32k) | **0.8164** (+1.76pp vs 32k) |
| [`data_scale_64k_baseline_seed43`](../experiments/data_scale_64k_baseline_seed43/) | 64k baseline, seed 43 | — | 0.8763 | 0.7950 |
| [`data_scale_64k_baseline_seed44`](../experiments/data_scale_64k_baseline_seed44/) | 64k baseline, seed 44 | — | 0.8753 | 0.8204 |
| `pause_events_holdloss0.5_5050sampling_seed42` (Section 5) | 16k, λ=0.5 50:50 | 12,797 | 0.8261 | 0.7080 |
| [`data_scale_32k_holdloss0.5_5050sampling`](../experiments/data_scale_32k_holdloss0.5_5050sampling/) | 32k, λ=0.5 50:50 | 28,114 final clips | 0.8500 (−0.22pp vs 32k baseline) | 0.7391 (−5.97pp vs 32k baseline) |
| [`data_scale_64k_holdloss0.5_5050sampling_seed42`](../experiments/data_scale_64k_holdloss0.5_5050sampling_seed42/) | 64k, λ=0.5 50:50, seed 42 | 61,841 final clips | 0.8664 (−0.40pp vs 64k baseline) | 0.7736 (−4.28pp vs 64k baseline) |
| [`data_scale_64k_holdloss0.5_5050sampling_seed44`](../experiments/data_scale_64k_holdloss0.5_5050sampling_seed44/) | 64k, λ=0.5 50:50, seed 44 | 61,841 final clips | 0.8641 | 0.7595 |

The third seed at this tier, `data_scale_64k_holdloss0.5_5050sampling_seed43` (64k, λ=0.5 50:50, seed 43,
61,841 final clips), is the **frozen finalist checkpoint** — its numbers are not re-derived here;
see `docs/RESULTS.md` and the README.

**Reading it**: 32k is a decisive, large win for the plain baseline (+5.8pp real AUC — the single
largest gain of any experiment in this project) and the trade-off with the hold-aware objective
persists and widens at 32k in relative terms. At 64k the baseline gain is still decisive but smaller
in absolute terms, and the accuracy/hold-safety gap doesn't monotonically widen with scale
(−3.29pp at 16k → −5.97pp at 32k → −4.28pp at 64k) — more data helps both arms, it just doesn't
resolve the trade-off in one consistent direction. 128k was never run (out of scope for this pass).
Full matched-recall-audit detail confirming these 64k numbers hold up at matched complete-turn
recall (not just at each model's own threshold) is in `docs/RESULTS.md` Section 6c.

---

## 7. Distillation ablation (D0 / D1 / D2)

Despite A0 failing its own teacher-qualification gate (Section 4), a distillation attempt was run
anyway, offline, to see whether its signal helps at all. Student: B1@1s; teacher: `whisper_tiny_boundary_robust_retrain`
(teacher logits precomputed once via `scripts/precompute_teacher_logits.py`); fixed recipe (T=2,
α=0.5, teacher loss on final clips only, hard labels on internal holds), via
`scripts/train_distillation_ablation.py`.

| Experiment | Recipe | Overall AUC | Real AUC | Δ real vs. B1 baseline |
|---|---|---|---|---|
| `mel_trajectory_1s_speech_aligned_contract` (Section 3) | B1@1s baseline, no augmentation, no teacher | 0.8279 | 0.7402 | — |
| [`distillation_isolation_control`](../experiments/distillation_isolation_control/) | isolation control: boundary augmentation on, `alpha=1.0` (hard labels only, teacher term zeroed), via `scripts/train_distillation_isolation_control.py` | 0.8000 | 0.6626 | **−7.76pp** |
| [`distillation_canonical_boundary_teacher`](../experiments/distillation_canonical_boundary_teacher/) | boundary augmentation + canonical-boundary teacher logits | 0.8277 | 0.7168 | −2.34pp |
| [`distillation_mean3boundary_teacher`](../experiments/distillation_mean3boundary_teacher/) | boundary augmentation + mean-of-3-boundary teacher logits | 0.8269 | 0.7186 | −2.16pp |

**Reading it**: neither D1 nor D2 clears the project's own bar for keeping distillation (+0.5pp
overall AUC *or* +1pp real AUC) — both move backward on both axes vs. the untouched baseline, so
**distillation was dropped**. But the D0 isolation control reframes *why*: boundary augmentation
alone costs a large −7.76pp real AUC, far worse than D1/D2's −2.16 to −2.34pp — meaning the
teacher-logit signal was recovering roughly two-thirds of that self-inflicted loss (+5.0 to +5.4pp
vs. D0), just not enough to fully cancel it out and beat the plain baseline. Each distilled
checkpoint's own-threshold hold-FCR (`fcr_at_holds_distill` in each `metrics.json`) shows a similar
pattern to the P1 family — e.g. D0's hold FCR (all) is 8.5% vs. a same-threshold baseline reading of
64.6%, so the boundary-augmented objective does suppress false completions at holds, just at a real
accuracy cost that ultimately disqualified it. Full attribution analysis in `docs/RESULTS.md`
Section 2b.

---

## 8. Pairwise-ranking experiment

Proposed as a replacement candidate for the P1a/P1b hold-loss family: a within-utterance margin
loss pairing each completed clip's final score against one internal-hold score from the *same*
clip, `L = L_final_BCE + 0.1 · max(0, 0.2 − s_final + s_hold)`, via `scripts/train_pairwise_ranking.py`.
Main BCE and checkpoint selection stay final-clips-only; no boundary augmentation.

| Experiment | Seed / protocol | Overall AUC | Real AUC | Hold FCR, all (own threshold) |
|---|---|---|---|---|
| [`pairwise_ranking_singleseed`](../experiments/pairwise_ranking_singleseed/) | single-seed, fixed 5-epoch | 0.8285 | 0.7433 | 15.4% (vs. baseline's 31.3% at the same threshold — a **−15.8pp** first-look result) |
| [`pairwise_ranking_seed42`](../experiments/pairwise_ranking_seed42/) | seed 42, plateau protocol | 0.8285 | 0.7433 | 15.4% |
| [`pairwise_ranking_seed43`](../experiments/pairwise_ranking_seed43/) | seed 43, plateau protocol | 0.8309 | 0.7361 | 13.2% |
| [`pairwise_ranking_seed44`](../experiments/pairwise_ranking_seed44/) | seed 44, plateau protocol | 0.8257 | 0.7241 | 12.8% |

3-seed mean: overall AUC 0.8284±0.002, real AUC 0.7345±0.008 (a **−2.00pp** real-AUC cost vs. the
matching 3-seed baseline mean of 0.7545 — by far the smallest cost of any hold-aware objective
tried, vs. −5.47pp for `λ=0.5 all` and −5.83pp for `λ=0.5 50:50` over the same 3 seeds). Ranking
clears its own single-seed keep bar decisively, but at 3 seeds its hold-FCR is worse than both
λ=0.5 variants — a genuine Pareto trade-off, not a dominant win. It doesn't unseat either λ=0.5
finalist but stays documented as the lowest-accuracy-cost alternative if that trade-off is ever
revisited. Full comparison table and the matched-recall correction (which sharpens rather than
reverses this call) are in `docs/RESULTS.md` Sections 3 and 5b.

---

## 9. Final gating, finalist selection, and the official result

Everything from here — the metric audit that softened A0's disqualification, the finalist decision
between `λ=0.5 all` and `λ=0.5 50:50`, the 3-seed confirmation, temperature scaling, and the single
official evaluation against the 31,527-clip `smart-turn-data-v3.2-test` set — is written up in full
in **[`docs/RESULTS.md`](RESULTS.md)**. The frozen checkpoint
(`data_scale_64k_holdloss0.5_5050sampling_seed43`) and its headline test-set numbers are summarized in the
[README](../README.md).
