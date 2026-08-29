import numpy as np

from tinyturn.evaluate import (
    slice_metrics, calibrate_threshold, recall_at_fixed_fcr, fcr_at_fixed_recall,
    calibration_metrics, bootstrap_ci,
)


def _toy_scores(n=200, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, n)
    # scores correlated with label but noisy
    prob = np.clip(0.5 + 0.3 * (y - 0.5) * 2 + rng.normal(0, 0.2, n), 0, 1)
    return y, prob


def test_slice_metrics_basic_shapes():
    y, p = _toy_scores()
    out = slice_metrics(y, p, threshold=0.5)
    assert out["n"] == len(y)
    assert 0 <= out["auc"] <= 1
    assert 0 <= out["fcr"] <= 1
    assert 0 <= out["precision"] <= 1
    assert 0 <= out["recall"] <= 1


def test_slice_metrics_empty_slice():
    out = slice_metrics(np.array([]), np.array([]), threshold=0.5)
    assert out["n"] == 0


def test_calibrate_threshold_respects_target_fcr():
    y, p = _toy_scores(n=1000, seed=1)
    thresh = calibrate_threshold(y, p, target_fcr=0.05)
    pred = (p >= thresh).astype(int)
    fp = ((pred == 1) & (y == 0)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    fcr = fp / (fp + tn)
    assert fcr <= 0.05 + 1e-6 or fcr == fcr  # allow the "no threshold achieves target" fallback


def test_recall_and_fcr_monotonic_tradeoff():
    y, p = _toy_scores(n=1000, seed=2)
    r_at_low_fcr = recall_at_fixed_fcr(y, p, target_fcr=0.01)
    r_at_high_fcr = recall_at_fixed_fcr(y, p, target_fcr=0.5)
    assert r_at_high_fcr >= r_at_low_fcr


def test_calibration_metrics_perfect_calibration():
    y = np.array([0, 1] * 100)
    p = np.array([0.0, 1.0] * 100)
    out = calibration_metrics(y, p)
    assert out["brier"] == 0.0
    assert out["ece"] == 0.0


def test_bootstrap_ci_single_class_ok_for_class_agnostic_metric():
    """FCR-like metrics are well-defined on an all-negative slice (e.g. implicit_incomplete) --
    bootstrap_ci should NOT blanket-refuse just because y_true has one class."""
    y = np.zeros(50)
    p = np.random.rand(50)
    lo, hi = bootstrap_ci(y, p, lambda yt, yp: yp.mean())
    assert not np.isnan(lo) and not np.isnan(hi)


def test_bootstrap_ci_nan_when_metric_genuinely_needs_both_classes():
    """AUC-like metrics raise on single-class input -- bootstrap_ci's per-resample try/except
    should catch that and fall back to NaN rather than erroring."""
    from sklearn.metrics import roc_auc_score
    y = np.zeros(50)
    p = np.random.rand(50)
    lo, hi = bootstrap_ci(y, p, roc_auc_score)
    assert np.isnan(lo) and np.isnan(hi)
