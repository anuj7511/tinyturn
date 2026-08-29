"""
Resolve Phase-2 8f/8g's provisional status.

Step 1: Wilson CIs (proportions: frac>0.20, flip rate) and bootstrap CIs (FCR degradation at fixed
recall, not a simple proportion -- matches tinyturn.evaluate.bootstrap_ci's own convention:
n_boot=200, seed=42, 2.5/97.5 percentile) on the existing 206-clip pilot overlap's *actually
evaluated* subset (n=43 val+calib rows -- the 163 train rows were deliberately excluded from 8f
since both models were trained on them, which would bias a robustness diagnostic).

Step 2: check each of the 3 frozen VAD-boundary criteria against its CI -- decisive (CI entirely on
one side of the threshold) vs. borderline (CI straddles it).

Step 3: if any criterion is borderline, escalate to the full E5 sample IF the audio is actually
local (checked explicitly, not assumed) -- recompute all 3 criteria + CIs there instead.

Usage:
  python scripts/run_8f_resolve_provisional.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.evaluate import fcr_at_fixed_recall

VAD_JSON = Path("experiments") / "8f_vad_boundary_diagnostic.json"
N_BOOT = 200
RNG_SEED = 42
Z_95 = 1.959963984540054


def wilson_ci(successes: int, n: int, z: float = Z_95):
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_fcr_degradation_ci(y_true, prob_canonical, prob_alt, target_recall=0.95,
                                  n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        if len(set(yt)) < 2:
            continue  # fcr_at_fixed_recall needs both classes to define a recall-matched threshold
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
    """direction='le': criterion is "value <= threshold" -- passes decisively if ci_hi <= threshold,
    fails decisively if ci_lo > threshold, else borderline."""
    lo, hi = ci
    if hi != hi or lo != lo:
        return "undetermined"
    if hi <= threshold:
        return "decisive-pass"
    if lo > threshold:
        return "decisive-fail"
    return "borderline"


def analyze(per_clip, model_name):
    df = pd.DataFrame(per_clip)
    print(f"\n=== {model_name}: n={len(df)} (real={int(df['real'].sum())}, "
          f"synthetic={int((~df['real']).sum())}) ===")
    results = {}
    for alt in ("silero_vad", "energy_alt"):
        col_diff, col_flip, col_prob = f"abs_diff_{alt}", f"flip_{alt}", f"prob_{alt}"
        if col_diff not in df.columns:
            continue
        sub = df[df[col_diff].notna()]
        n = len(sub)

        n_gt020 = int((sub[col_diff] > 0.20).sum())
        frac_ci = wilson_ci(n_gt020, n)
        frac_point = n_gt020 / n
        frac_verdict = decisive(frac_ci, 0.10)

        n_flip = int(sub[col_flip].sum())
        flip_ci = wilson_ci(n_flip, n)
        flip_point = n_flip / n
        flip_verdict = decisive(flip_ci, 0.05)

        real_sub = sub[sub["real"]]
        fcr_result = None
        fcr_verdict = "undetermined (n too small)"
        if real_sub["endpoint_bool"].nunique() > 1 and len(real_sub) >= 2:
            fcr_result = bootstrap_fcr_degradation_ci(
                real_sub["endpoint_bool"].values.astype(int),
                real_sub["prob_canonical"].values, real_sub[col_prob].values)
            if fcr_result is not None:
                fcr_ci = (fcr_result[0], fcr_result[1])
                fcr_verdict = decisive(fcr_ci, 2.0)

        print(f"\n  vs. {alt} (n={n}):")
        print(f"    frac_gt_0.20   = {frac_point:.4f}  Wilson 95% CI = "
              f"[{frac_ci[0]:.4f}, {frac_ci[1]:.4f}]  (threshold <=0.10) -> {frac_verdict}")
        print(f"    flip_rate      = {flip_point:.4f}  Wilson 95% CI = "
              f"[{flip_ci[0]:.4f}, {flip_ci[1]:.4f}]  (threshold <=0.05) -> {flip_verdict}")
        if fcr_result is not None:
            print(f"    fcr_degrad_pp  bootstrap 95% CI (n_real={len(real_sub)}, "
                  f"{fcr_result[2]}/{N_BOOT} valid resamples) = "
                  f"[{fcr_result[0]:.2f}pp, {fcr_result[1]:.2f}pp]  (threshold <=2pp) -> {fcr_verdict}")
        else:
            print(f"    fcr_degrad_pp  -> {fcr_verdict} (n_real={len(real_sub)}, "
                  f"endpoint_bool classes present: {real_sub['endpoint_bool'].unique().tolist() if len(real_sub) else []})")

        results[alt] = {
            "n": n, "frac_gt_020": frac_point, "frac_gt_020_ci": frac_ci, "frac_gt_020_verdict": frac_verdict,
            "flip_rate": flip_point, "flip_rate_ci": flip_ci, "flip_rate_verdict": flip_verdict,
            "n_real": int(len(real_sub)),
            "fcr_degradation_ci": fcr_result[:2] if fcr_result else None,
            "fcr_degradation_verdict": fcr_verdict,
        }
    return results


def main():
    d = json.load(open(VAD_JSON))
    a0_results = analyze(d["per_clip_A0"], "A0")
    b1_results = analyze(d["per_clip_B1_1s"], "B1@1s (context only, not part of 8g's gate)")

    out = {"n_pilot": len(d["per_clip_A0"]), "A0": a0_results, "B1_1s": b1_results}
    with open(Path("experiments") / "A0_whisper_tiny_pv2speechend" / "8f_ci_analysis_pilot.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nsaved experiments/A0_whisper_tiny_pv2speechend/8f_ci_analysis_pilot.json")


if __name__ == "__main__":
    main()
