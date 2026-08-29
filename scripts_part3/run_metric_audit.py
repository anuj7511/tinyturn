"""
Step 10 planning, item 1 -- ground-truth-conditioned metric audit (mandatory, no training).

Motivation: 8g's "safety-critical flip rate" (run_8g_qualify_teacher_v2.py::direction_specific_gates)
conditions only on the CANONICAL PREDICTION (canonical says incomplete -> alt flips to complete). It
never checks whether canonical was actually right. A flip counted there can be a correction (truth is
really complete, canonical was wrong, the alt boundary fixed it) rather than a safety problem. That
inflates the flip-rate gate with events that aren't actually "introduced" errors.

This audit instead conditions on ground truth AND canonical correctness, so only genuinely
introduced errors count:

  introduced false completion: truth=incomplete, canonical=continue (correct), alternative=complete (now wrong)
  introduced delay:            truth=complete,   canonical=complete (correct), alternative=continue (now wrong)

It also reports each boundary variant's own FCR-at-fixed-recall (canonical / alt_threshold /
silero_vad) standalone, rather than only the alt-minus-canonical degradation 8g reports.

No new inference: reuses the per-clip probabilities already saved by run_8f_vad_boundary_diagnostic_v2.py
for A0 (original + boundary-robust remediation) and B1@1s.

Usage:
  python scripts_part3/run_metric_audit.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.evaluate import fcr_at_fixed_recall

Z_95 = 1.959963984540054
TARGET_RECALL = 0.95
MIN_N_FOR_FCR = 20

MODELS = [
    {
        "name": "A0_original",
        "exp_dir": Path("experiments") / "A0_whisper_tiny_pv2speechend",
        "vad_path": Path("experiments") / "8f_vad_boundary_diagnostic_v2.json",
        "per_clip_key": "per_clip_A0",
    },
    {
        "name": "A0_boundary_robust",
        "exp_dir": Path("experiments") / "A0_boundary_robust",
        "vad_path": Path("experiments") / "8f_vad_boundary_diagnostic_v2__A0_boundary_robust.json",
        "per_clip_key": "per_clip_A0",
    },
    {
        "name": "B1_1s",
        "exp_dir": Path("experiments") / "C1_B1_1s_pv2speechend",
        "vad_path": Path("experiments") / "8f_vad_boundary_diagnostic_v2.json",
        "per_clip_key": "per_clip_B1_1s",
    },
]

OUT_PATH = Path("experiments") / "metric_audit_ground_truth_conditioned.json"


def wilson_ci(successes: int, n: int, z: float = Z_95):
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def rate_entry(successes: int, n: int):
    if n == 0:
        return {"n": 0, "count": 0, "rate": None, "ci_95": [None, None]}
    lo, hi = wilson_ci(successes, n)
    return {"n": int(n), "count": int(successes), "rate": round(successes / n, 5),
            "ci_95": [round(lo, 5), round(hi, 5)]}


def introduced_errors(sub: pd.DataFrame, alt_name: str, threshold: float):
    """Ground-truth-conditioned introduced-error rates for one alt boundary, on one slice."""
    col_alt = f"prob_{alt_name}"
    d = sub[sub[col_alt].notna()]
    truth_complete = d["endpoint_bool"].astype(bool)
    canon_complete = d["prob_canonical"] >= threshold
    alt_complete = d[col_alt] >= threshold

    fc_pool = d[(~truth_complete) & (~canon_complete)]  # truth incomplete, canonical correctly says continue
    fc_flips = int((alt_complete[(~truth_complete) & (~canon_complete)]).sum())

    delay_pool = d[truth_complete & canon_complete]  # truth complete, canonical correctly says complete
    delay_flips = int((~alt_complete[truth_complete & canon_complete]).sum())

    return {
        "introduced_false_completion": rate_entry(fc_flips, len(fc_pool)),
        "introduced_delay": rate_entry(delay_flips, len(delay_pool)),
    }


def fcr_triplet(sub: pd.DataFrame, alt_name: str, label: str):
    col_alt = f"prob_{alt_name}"
    d = sub[sub[col_alt].notna()]
    y = d["endpoint_bool"].values.astype(int)
    if len(d) < MIN_N_FOR_FCR or len(set(y)) < 2:
        return {"note": f"skipped ({label}): n={len(d)} < {MIN_N_FOR_FCR} or single class"}
    return {
        "n": int(len(d)),
        "fcr_at_recall95_canonical": round(fcr_at_fixed_recall(y, d["prob_canonical"].values, TARGET_RECALL), 5),
        f"fcr_at_recall95_{alt_name}": round(fcr_at_fixed_recall(y, d[col_alt].values, TARGET_RECALL), 5),
    }


def audit_one(model_cfg: dict) -> dict:
    exp_dir, vad_path, per_clip_key = model_cfg["exp_dir"], model_cfg["vad_path"], model_cfg["per_clip_key"]
    metrics = json.load(open(exp_dir / "metrics.json"))
    threshold = float(metrics["threshold"])
    vad_raw = json.load(open(vad_path))
    df = pd.DataFrame(vad_raw[per_clip_key])

    result = {"threshold": threshold, "n_val_clips": int(len(df))}
    for alt_name in ["alt_threshold", "silero_vad"]:
        if f"prob_{alt_name}" not in df.columns:
            continue
        real_df = df[df["real"]]
        result[alt_name] = {
            "overall": introduced_errors(df, alt_name, threshold),
            "real_only": introduced_errors(real_df, alt_name, threshold),
            "fcr_at_recall95": {
                "overall": fcr_triplet(df, alt_name, "overall"),
                "real_only": fcr_triplet(real_df, alt_name, "real_only"),
            },
        }
    return result


def main():
    all_results = {}
    for cfg in MODELS:
        missing = [p for p in (cfg["exp_dir"] / "metrics.json", cfg["vad_path"]) if not p.exists()]
        if missing:
            print(f"SKIP {cfg['name']}: missing {[str(p) for p in missing]}")
            continue
        print(f"auditing {cfg['name']}...", flush=True)
        all_results[cfg["name"]] = audit_one(cfg)

    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(json.dumps(all_results, indent=2, default=str))
    print(f"\nsaved {OUT_PATH}")

    print("\n=== Ground-truth-conditioned safety summary ===")
    for name, r in all_results.items():
        for alt_name in ["alt_threshold", "silero_vad"]:
            if alt_name not in r:
                continue
            ie = r[alt_name]["overall"]["introduced_false_completion"]
            id_ = r[alt_name]["overall"]["introduced_delay"]
            print(f"{name} / {alt_name}: introduced_false_completion={ie['rate']} "
                  f"(n={ie['n']}, ci={ie['ci_95']})  introduced_delay={id_['rate']} "
                  f"(n={id_['n']}, ci={id_['ci_95']})")


if __name__ == "__main__":
    main()
