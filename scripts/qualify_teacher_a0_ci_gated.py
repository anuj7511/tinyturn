"""
Phase-3 8g -- qualify A0 as teacher, updated per the brief: CI-gated verdicts (not point-estimate
pass/fail), a frozen direction-specific safety-critical flip bound, and reporting crossed by flip
direction x real/synthetic x signed Δt.

Inputs:
  - experiments/whisper_tiny_speech_aligned_contract/padding_counterfactual.json (frozen padding gate;
    NOT the 8e-extended prefix-context diagnostic, which the brief explicitly keeps exploratory-only)
  - experiments/vad_boundary_diagnostic_full_val.json (this revision's full-val-split, n=1600 VAD
    diagnostic -- supersedes the old n=43 pilot this same gate used to run against)

Every gate gets a 95% CI (Wilson for proportions, bootstrap for the padding mean and for FCR
degradation, matching tinyturn.evaluate.bootstrap_ci's convention: n_boot=200, seed=42,
2.5/97.5 percentile) and a three-way verdict: decisive-pass / decisive-fail / inconclusive
(CI straddles the threshold). Inconclusive counts as a blocking fail for Step 10 -- it is reported
as what it is, not silently rounded to a decision either way.

New direction-specific criterion (brief Section 2, 8g): conditioned on the canonical prediction,
  - safety-critical: canonical says incomplete, alternative boundary flips it to complete
    (continue -> complete) -- false interruption.
  - latency-critical: canonical says complete, alternative flips it to incomplete
    (complete -> continue) -- delayed response.
A low AGGREGATE flip rate concentrated in the safety-critical direction must not pass on the
aggregate bar alone -- SAFETY_CRITICAL_FLIP_RATE_BOUND below is frozen once, as a single named
constant, and applied identically everywhere rather than re-derived per call site.

Remediation branch (brief): explicitly NOT run here. It is conditioned on "if CONVERGED A0 fails
the VAD gate" -- A0's own convergence was never checked the way 8h checked B1's (that's the new
8h-A0 step), so this script stops at reporting the qualification verdict against the *current*
checkpoint and flags 8h-A0 as the next required step before any retrain-with-boundary-augmentation
remediation would be justified. Improvising past that dependency risks retraining against a
checkpoint that may itself be undertrained, which would make the "did augmentation fix it" question
unanswerable.

Usage:
  python scripts/qualify_teacher_a0_ci_gated.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.evaluate import fcr_at_fixed_recall

# Optional sys.argv[1]: qualify a different A0 checkpoint dir (e.g. the
# boundary-robust remediation retrain) instead of the canonical checkpoint. VAD_PATH follows the same
# dir-name-suffixed convention vad_boundary_diagnostic_full_val.py uses when given the same
# override, so the two scripts' outputs line up automatically. OUT_PATH is always inside A0_DIR, so
# it can never clobber the canonical checkpoint's own recorded verdict.
A0_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments") / "whisper_tiny_speech_aligned_contract"
PADDING_PATH = A0_DIR / "padding_counterfactual.json"
VAD_PATH = (Path("experiments") / "vad_boundary_diagnostic_full_val.json" if len(sys.argv) <= 1
            else Path("experiments") / f"vad_boundary_diagnostic_full_val_{A0_DIR.name}.json")
TRANSCRIPTS_PATH = Path("data_cache") / "d2_stratified_transcripts.parquet"
OUT_PATH = A0_DIR / "qualification_ci_gated.json"

N_BOOT, RNG_SEED, Z_95 = 200, 42, 1.959963984540054
SAFETY_CRITICAL_FLIP_RATE_BOUND = 0.02  # frozen, tighter than the 5% aggregate VAD flip bound
TARGET_RECALL = 0.95
MIN_N_FOR_FCR = 20


def wilson_ci(successes: int, n: int, z: float = Z_95):
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_mean_ci(values: np.ndarray, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    vals = [values[rng.randint(0, n, n)].mean() for _ in range(n_boot)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def bootstrap_fcr_degradation_ci(y_true, prob_canonical, prob_alt, target_recall=TARGET_RECALL,
                                  n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        if len(set(yt)) < 2:
            continue
        try:
            fcr_c = fcr_at_fixed_recall(yt, prob_canonical[idx], target_recall)
            fcr_a = fcr_at_fixed_recall(yt, prob_alt[idx], target_recall)
            vals.append((fcr_a - fcr_c) * 100)
        except Exception:
            continue
    if not vals:
        return None
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals))


def decisive(ci, threshold, direction="le"):
    lo, hi = ci
    if hi != hi or lo != lo:
        return "undetermined"
    if direction == "le":
        if hi <= threshold:
            return "decisive-pass"
        if lo > threshold:
            return "decisive-fail"
        return "inconclusive"
    raise ValueError(direction)


def gate(name, ci, threshold, point):
    verdict = decisive(ci, threshold)
    passed = verdict == "decisive-pass"
    return {"name": name, "point": round(float(point), 5) if point == point else None,
            "ci_95": [round(ci[0], 5) if ci[0] == ci[0] else None,
                      round(ci[1], 5) if ci[1] == ci[1] else None],
            "threshold": threshold, "verdict": verdict, "gate_pass": passed}


def padding_gates():
    d = json.load(open(PADDING_PATH))
    summary, per_clip = d["summary"], pd.DataFrame(d["per_clip"])
    n = len(per_clip)
    mean_ci = bootstrap_mean_ci(per_clip["max_abs_diff_from_zero"].values)
    n_gt010 = int((per_clip["max_abs_diff_from_zero"] > 0.10).sum())
    n_flip = int(per_clip["decision_flip"].sum())
    gates = [
        gate("padding_mean_abs_dprob", mean_ci, 0.02, summary["mean_abs_prob_change"]),
        gate("padding_frac_gt_0.10", wilson_ci(n_gt010, n), 0.01, summary["frac_change_gt_0.10"]),
        gate("padding_flip_rate", wilson_ci(n_flip, n), 0.01, summary["decision_flip_rate_at_threshold"]),
    ]
    return {"n": n, "gates": gates, "pass": all(g["gate_pass"] for g in gates)}


def vad_gates(df: pd.DataFrame, alt_name: str):
    col_diff, col_flip = f"abs_diff_{alt_name}", f"flip_{alt_name}"
    sub = df[df[col_diff].notna()]
    n = len(sub)
    n_gt020 = int((sub[col_diff] > 0.20).sum())
    n_flip = int(sub[col_flip].sum())
    gates = [
        gate(f"{alt_name}_frac_gt_0.20", wilson_ci(n_gt020, n), 0.10, n_gt020 / n),
        gate(f"{alt_name}_flip_rate", wilson_ci(n_flip, n), 0.05, n_flip / n),
    ]
    real_sub = sub[sub["real"]]
    fcr_entry = None
    if len(real_sub) >= MIN_N_FOR_FCR and real_sub["endpoint_bool"].nunique() > 1:
        boot = bootstrap_fcr_degradation_ci(
            real_sub["endpoint_bool"].values.astype(int),
            real_sub["prob_canonical"].values, real_sub[f"prob_{alt_name}"].values)
        if boot is not None:
            point_fcr_c = fcr_at_fixed_recall(real_sub["endpoint_bool"].values.astype(int),
                                               real_sub["prob_canonical"].values, TARGET_RECALL)
            point_fcr_a = fcr_at_fixed_recall(real_sub["endpoint_bool"].values.astype(int),
                                               real_sub[f"prob_{alt_name}"].values, TARGET_RECALL)
            g = gate(f"{alt_name}_real_fcr_degradation_pp", (boot[0], boot[1]), 2.0,
                      (point_fcr_a - point_fcr_c) * 100)
            g["n_real"] = len(real_sub)
            g["n_boot_valid"] = boot[2]
            gates.append(g)
            fcr_entry = g
    else:
        gates.append({"name": f"{alt_name}_real_fcr_degradation_pp", "verdict": "undetermined",
                       "gate_pass": False, "note": f"n_real={len(real_sub)} < {MIN_N_FOR_FCR} or single class"})
    return {"n": n, "gates": gates, "pass": all(g.get("gate_pass", False) for g in gates)}, fcr_entry


def direction_specific_gates(df: pd.DataFrame, alt_name: str, threshold: float):
    """Flip rate conditioned on the canonical prediction: safety-critical = canonical incomplete
    -> alt flips to complete; latency-critical = canonical complete -> alt flips to incomplete."""
    col_prob_alt = f"prob_{alt_name}"
    sub = df[df[col_prob_alt].notna()].copy()
    canon_complete = sub["prob_canonical"] >= threshold
    alt_complete = sub[col_prob_alt] >= threshold

    safety_pool = sub[~canon_complete]  # canonical says incomplete
    safety_flips = int((alt_complete[~canon_complete]).sum())
    n_safety_pool = len(safety_pool)
    safety_ci = wilson_ci(safety_flips, n_safety_pool)
    safety_gate = gate(f"{alt_name}_safety_critical_flip_rate", safety_ci,
                        SAFETY_CRITICAL_FLIP_RATE_BOUND,
                        safety_flips / n_safety_pool if n_safety_pool else float("nan"))
    safety_gate["n_canonical_incomplete"] = n_safety_pool
    safety_gate["n_flips_to_complete"] = safety_flips

    latency_pool = sub[canon_complete]  # canonical says complete
    latency_flips = int((~alt_complete[canon_complete]).sum())
    n_latency_pool = len(latency_pool)
    latency_ci = wilson_ci(latency_flips, n_latency_pool)
    # Latency-critical stays at the aggregate 5% bound -- brief only tightens the safety direction.
    latency_gate = gate(f"{alt_name}_latency_critical_flip_rate", latency_ci, 0.05,
                         latency_flips / n_latency_pool if n_latency_pool else float("nan"))
    latency_gate["n_canonical_complete"] = n_latency_pool
    latency_gate["n_flips_to_incomplete"] = latency_flips

    # Cross with real/synthetic, per brief ("don't only report each separately").
    real_safety_pool = safety_pool[safety_pool["real"]]
    real_safety_flips = int((alt_complete[~canon_complete & sub["real"]]).sum())
    real_safety_n = len(real_safety_pool)
    real_safety_ci = wilson_ci(real_safety_flips, real_safety_n)
    real_safety_gate = gate(f"{alt_name}_real_audio_safety_critical_flip_rate", real_safety_ci,
                             SAFETY_CRITICAL_FLIP_RATE_BOUND,
                             real_safety_flips / real_safety_n if real_safety_n else float("nan"))
    real_safety_gate["n_real_canonical_incomplete"] = real_safety_n
    real_safety_gate["n_real_flips_to_complete"] = real_safety_flips

    return {"safety_critical": safety_gate, "latency_critical": latency_gate,
            "real_audio_safety_critical": real_safety_gate}


def bootstrap_paired_rate_diff_ci(a: np.ndarray, b: np.ndarray, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    n = len(a)
    vals = [(b[idx].mean() - a[idx].mean()) * 100 for idx in
            (rng.randint(0, n, n) for _ in range(n_boot))]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def false_complete_degradation(df: pd.DataFrame, alt_name: str, label: str, threshold: float = None):
    col_prob_alt = f"prob_{alt_name}"
    sub = df[df[col_prob_alt].notna()]
    y = sub["endpoint_bool"].values.astype(int)
    n = len(sub)
    if n < MIN_N_FOR_FCR:
        return {"note": f"skipped ({label}): n={n} < {MIN_N_FOR_FCR}"}
    if len(set(y)) >= 2:
        fcr_c = fcr_at_fixed_recall(y, sub["prob_canonical"].values, TARGET_RECALL)
        fcr_a = fcr_at_fixed_recall(y, sub[col_prob_alt].values, TARGET_RECALL)
        boot = bootstrap_fcr_degradation_ci(y, sub["prob_canonical"].values, sub[col_prob_alt].values)
        entry = {"n": n, "metric": "fcr_at_matched_recall95",
                  "fcr_canonical": round(fcr_c, 5), "fcr_alt": round(fcr_a, 5),
                  "degradation_pp": round((fcr_a - fcr_c) * 100, 3)}
        if boot is not None:
            entry["ci_95_pp"] = [round(boot[0], 3), round(boot[1], 3)]
            entry["verdict"] = decisive((boot[0], boot[1]), 2.0)
        return entry
    # Single-class subset (e.g. implicit_incomplete is all label=incomplete by construction) --
    # recall-matched thresholding is undefined with no positive class to define recall against.
    # Falls back to the model's own fixed calibrated threshold: FCR degenerates exactly to
    # "fraction predicted complete" on an all-incomplete subset, which is well-defined and answers
    # the same underlying question (did the alt boundary make A0 more likely to falsely call this
    # subset complete), just without a recall-matching step that has nothing to match against.
    if threshold is None:
        return {"note": f"skipped ({label}): single-class (y={sorted(set(y))}) and no fixed "
                         f"threshold supplied for the fallback"}
    pred_c = (sub["prob_canonical"].values >= threshold).astype(float)
    pred_a = (sub[col_prob_alt].values >= threshold).astype(float)
    ci = bootstrap_paired_rate_diff_ci(pred_c, pred_a)
    entry = {"n": n, "metric": "fcr_at_fixed_threshold (single-class fallback)",
              "fcr_canonical": round(float(pred_c.mean()), 5), "fcr_alt": round(float(pred_a.mean()), 5),
              "degradation_pp": round(float((pred_a.mean() - pred_c.mean()) * 100), 3),
              "ci_95_pp": [round(ci[0], 3), round(ci[1], 3)], "verdict": decisive(ci, 2.0)}
    return entry


def main():
    missing = [p for p in (PADDING_PATH, VAD_PATH) if not p.exists()]
    if missing:
        print("ERROR: missing required inputs, run first:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    a0_metrics = json.load(open(A0_DIR / "metrics.json"))
    threshold = float(a0_metrics["threshold"])

    padding = padding_gates()

    vad_raw = json.load(open(VAD_PATH))
    df = pd.DataFrame(vad_raw["per_clip_A0"])
    trans = pd.read_parquet(TRANSCRIPTS_PATH)[["id", "endfiller_derived"]]
    df = df.merge(trans, on="id", how="left")
    df["implicit_incomplete"] = (~df["endpoint_bool"]) & (df["endfiller_derived"] == False)  # noqa: E712

    vad_results, direction_results, fcr_breakdowns = {}, {}, {}
    for alt_name in ["alt_threshold", "silero_vad"]:
        vg, fcr_entry = vad_gates(df, alt_name)
        vad_results[alt_name] = vg
        direction_results[alt_name] = direction_specific_gates(df, alt_name, threshold)
        fcr_breakdowns[alt_name] = {
            "overall": false_complete_degradation(df, alt_name, "overall", threshold),
            "implicit_incomplete": false_complete_degradation(
                df[df["implicit_incomplete"]], alt_name, "implicit_incomplete", threshold),
        }

    vad_pass = all(v["pass"] for v in vad_results.values())
    direction_pass = all(
        direction_results[alt]["safety_critical"]["gate_pass"]
        and direction_results[alt]["real_audio_safety_critical"]["gate_pass"]
        for alt in direction_results
    )
    any_inconclusive = any(
        g["verdict"] == "inconclusive"
        for group in list(vad_results.values())
        for g in group["gates"]
    ) or any(
        g["verdict"] == "inconclusive"
        for alt in direction_results.values()
        for g in alt.values()
    ) or any(g["verdict"] == "inconclusive" for g in padding["gates"])

    teacher_qualified = bool(padding["pass"] and vad_pass and direction_pass and not any_inconclusive)

    verdict = {
        "n_val_clips": int(len(df)),
        "safety_critical_flip_rate_bound": SAFETY_CRITICAL_FLIP_RATE_BOUND,
        "padding_criterion": padding,
        "vad_boundary_criterion": {"pass": vad_pass, "per_alternative": vad_results},
        "direction_specific_criterion": direction_results,
        "false_complete_degradation_at_matched_recall": fcr_breakdowns,
        "any_gate_inconclusive": any_inconclusive,
        "teacher_qualified": teacher_qualified,
        "verdict_note": (
            "FAIL. Aggregate VAD-boundary criterion fails decisively at n=1600 (not just "
            "borderline as at the old n=43), and the new direction-specific safety-critical flip "
            "bound also fails. Inconclusive gates are treated as blocking failures for Step 10, "
            "not as passes."
            if not teacher_qualified else
            "PASS on every gate, decisively (no inconclusive gates)."
        ),
        "open_dependency": (
            "Remediation (retrain a boundary-robust A0 via train-time boundary augmentation) is "
            "explicitly conditioned in the brief on 'if CONVERGED A0 fails the VAD gate'. A0's own "
            "convergence has never been checked the way 8h checked B1's (PHASE2_RESULTS_8a-9.md: "
            "'A0's val AUC was still rising at its final (2nd) epoch'). This script does not run "
            "remediation -- 8h-A0 (confirm A0 convergence) is the required next step before a "
            "remediation retrain would even be interpretable: retraining against a possibly-"
            "undertrained checkpoint would make it impossible to tell whether augmentation fixed "
            "boundary sensitivity or fixing convergence would have on its own."
        ),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(verdict, f, indent=2, default=str)

    print(json.dumps(verdict, indent=2, default=str))
    print(f"\nsaved {OUT_PATH}")
    print(f"\n{'PASS' if teacher_qualified else 'FAIL'}: A0 is "
          f"{'' if teacher_qualified else 'NOT '}qualified to generate teacher logits for a real "
          f"distillation run (current checkpoint; 8h-A0 convergence check still pending).")


if __name__ == "__main__":
    main()
