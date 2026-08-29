"""8h-A0 Step 1 (brief Section 2, "8h-A0"): a paired significance test between two checkpoints'
existing validation predictions, on the identical val set -- no new training needed. Resolves
whether a small AUC gap (e.g. the 0.16pp gap between A0@4s's original fixed-2-epoch checkpoint and
its Kaggle early-stopped retrain) is distinguishable from noise before spending a rerun on it.

Paired (not independent) resampling: the same bootstrap row-indices are applied to both models'
predictions each draw, since both were scored on the same val clips -- this correctly cancels
per-clip difficulty and only captures the correlated model-vs-model difference, unlike two separate
per-model CIs compared informally.
"""
from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score


def paired_bootstrap_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray,
                           metric_fn: Callable = roc_auc_score, n_boot: int = 2000,
                           seed: int = 42) -> dict:
    """Returns the observed metric(a) - metric(b) difference, its bootstrap 95% CI, and a two-sided
    bootstrap p-value (twice the smaller tail fraction crossing zero, capped at 1.0)."""
    y_true = np.asarray(y_true)
    prob_a = np.asarray(prob_a)
    prob_b = np.asarray(prob_b)
    assert len(y_true) == len(prob_a) == len(prob_b)
    n = len(y_true)

    observed_a = metric_fn(y_true, prob_a)
    observed_b = metric_fn(y_true, prob_b)
    observed_diff = observed_a - observed_b

    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        if len(set(yt.tolist())) < 2:
            continue
        try:
            diffs.append(metric_fn(yt, prob_a[idx]) - metric_fn(yt, prob_b[idx]))
        except Exception:
            continue

    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5]) if len(diffs) else (np.nan, np.nan)
    if len(diffs):
        frac_le_zero = float((diffs <= 0).mean())
        frac_ge_zero = float((diffs >= 0).mean())
        p_value = float(min(1.0, 2 * min(frac_le_zero, frac_ge_zero)))
    else:
        p_value = float("nan")

    return {
        "metric_a": float(observed_a),
        "metric_b": float(observed_b),
        "observed_diff": float(observed_diff),
        "ci_95": (float(ci_lo), float(ci_hi)),
        "p_value": p_value,
        "n_boot_used": int(len(diffs)),
        "n": int(n),
        "distinguishable_from_noise": bool(len(diffs) and not (ci_lo <= 0 <= ci_hi)),
    }
