"""
Section 8 evaluation protocol -- computed identically for every experiment (C0 is the one
exception, since it isn't a trained model). Every slice reports n, AUC, F1/precision/recall (at a
calibrated threshold), FCR, recall-at-fixed-FCR, FCR-at-fixed-recall, plus overall parameter count/
model size/latency and calibration metrics at the top level. Bootstrap CIs (n_boot=200, matching
the convention used throughout the EDA) are reported per slice number.

Split discipline: threshold is calibrated on the `calib` split (never on val or the official test);
headline slice metrics are then reported on the `val` split (model-selection validation) using that
fixed threshold. The official 31,527-row HF test set is untouched here by design (Section 8: one
pass per finalist, not per experiment) -- Steps 1-5 never call this module against it.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score


def count_macs(model, *sample_inputs) -> int:
    """Batch-1 MACs for one forward pass (brief Section 8: "MACs if available")."""
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(*sample_inputs)
    return fc.get_total_flops() // 2  # FlopCounterMode counts multiply+add as 2 FLOPs

REAL_LANGUAGES = {"eng", "spa"}
N_BOOT = 200
RNG_SEED = 42


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn: Callable,
                  n_boot: int = N_BOOT, seed: int = RNG_SEED):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n < 20:
        return (np.nan, np.nan)
    # Note: some metrics (e.g. FCR) are well-defined on a single-class slice (implicit_incomplete
    # is all-negative by construction) -- only AUC-like metrics actually need both classes, and
    # those already fail fast per-resample via the try/except below, so no top-level class-count
    # gate here.
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        try:
            vals.append(metric_fn(yt, yp))
        except Exception:
            continue
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def calibrate_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_fcr: float = 0.05) -> float:
    """Smallest threshold such that FCR = FP/(FP+TN) <= target_fcr on the calib split (highest
    recall achievable while keeping false-completion rate at or below target)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ok = fpr <= target_fcr
    if not ok.any():
        return float(thresholds[np.argmin(fpr)])
    best = np.argmax(tpr * ok)
    return float(thresholds[best])


def recall_at_fixed_fcr(y_true, y_prob, target_fcr=0.05):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ok = fpr <= target_fcr
    return float(tpr[ok].max()) if ok.any() else float(tpr[np.argmin(fpr)])


def fcr_at_fixed_recall(y_true, y_prob, target_recall=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ok = tpr >= target_recall
    return float(fpr[ok].min()) if ok.any() else float(fpr[np.argmax(tpr)])


def calibration_metrics(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    brier = float(np.mean((y_prob - y_true) ** 2))
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if m.sum() == 0:
            continue
        conf = y_prob[m].mean()
        acc = y_true[m].mean()
        ece += (m.sum() / len(y_prob)) * abs(acc - conf)
    return {"brier": brier, "ece": float(ece)}


def slice_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    n = len(y_true)
    if n == 0:
        return {"n": 0}
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    out = {"n": int(n)}
    if len(set(y_true)) >= 2:
        auc = roc_auc_score(y_true, y_prob)
        out["auc"] = round(float(auc), 4)
        out["auc_ci"] = tuple(round(v, 4) for v in bootstrap_ci(y_true, y_prob, roc_auc_score))
    else:
        out["auc"] = None
        out["auc_ci"] = (np.nan, np.nan)

    def _f1(yt, yp):
        return f1_score(yt, (yp >= threshold).astype(int), zero_division=0)

    def _precision(yt, yp):
        return precision_score(yt, (yp >= threshold).astype(int), zero_division=0)

    def _recall(yt, yp):
        return recall_score(yt, (yp >= threshold).astype(int), zero_division=0)

    if (y_true == 1).any():
        # f1/precision/recall are only meaningful when the slice actually contains positives
        # (e.g. `implicit_incomplete` is by definition all-negative -- only FCR applies there).
        out["f1"] = round(float(_f1(y_true, y_prob)), 4)
        out["f1_ci"] = tuple(round(v, 4) if v == v else v for v in bootstrap_ci(y_true, y_prob, _f1))
        out["precision"] = round(float(_precision(y_true, y_prob)), 4)
        out["precision_ci"] = tuple(round(v, 4) if v == v else v for v in bootstrap_ci(y_true, y_prob, _precision))
        out["recall"] = round(float(_recall(y_true, y_prob)), 4)
        out["recall_ci"] = tuple(round(v, 4) if v == v else v for v in bootstrap_ci(y_true, y_prob, _recall))
    else:
        out["f1"] = out["precision"] = out["recall"] = None
        out["f1_ci"] = out["precision_ci"] = out["recall_ci"] = (np.nan, np.nan)

    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fcr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    out["fcr"] = round(fcr, 4)

    def _fcr(yt, yp):
        pred = (yp >= threshold).astype(int)
        fp_ = ((pred == 1) & (yt == 0)).sum()
        tn_ = ((pred == 0) & (yt == 0)).sum()
        return fp_ / (fp_ + tn_) if (fp_ + tn_) > 0 else np.nan

    out["fcr_ci"] = tuple(round(v, 4) if v == v else v for v in bootstrap_ci(y_true, y_prob, _fcr))

    if len(set(y_true)) >= 2:
        out["recall_at_fcr05"] = round(recall_at_fixed_fcr(y_true, y_prob, 0.05), 4)
        out["fcr_at_recall95"] = round(fcr_at_fixed_recall(y_true, y_prob, 0.95), 4)
    return out


@dataclass
class EvalOutputs:
    ids: list
    y_true: np.ndarray
    y_prob: np.ndarray
    language: list
    dataset: list
    synthetic: list
    implicit_incomplete: list


@torch.no_grad()
def run_inference(model, dataloader, device, use_trajectory: bool) -> EvalOutputs:
    model.eval()
    ids, y_true, y_prob = [], [], []
    language, dataset, synthetic, implicit_incomplete = [], [], [], []
    for batch in dataloader:
        log_mel = batch["log_mel"].to(device)
        mask = batch["valid_frame_mask"].to(device)
        traj = batch["trajectory"].to(device) if use_trajectory else None
        logits = model(log_mel, mask, traj)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(batch["label"].numpy().tolist())
        ids.extend(batch["id"])
        language.extend(batch["language"])
        dataset.extend(batch["dataset"])
        synthetic.extend(batch["synthetic"])
        implicit_incomplete.extend(batch["implicit_incomplete"])
    return EvalOutputs(ids=ids, y_true=np.array(y_true), y_prob=np.array(y_prob),
                        language=language, dataset=dataset, synthetic=synthetic,
                        implicit_incomplete=implicit_incomplete)


def full_report(outputs: EvalOutputs, threshold: float) -> dict:
    yt, yp = outputs.y_true, outputs.y_prob
    language = np.array(outputs.language)
    dataset = np.array(outputs.dataset)
    synthetic = np.array(outputs.synthetic, dtype=bool)
    implicit = np.array(outputs.implicit_incomplete, dtype=bool)
    real_mask = ~synthetic

    report = {"overall": slice_metrics(yt, yp, threshold)}
    report["implicit_incomplete"] = slice_metrics(yt[implicit], yp[implicit], threshold)
    report["real_all"] = slice_metrics(yt[real_mask], yp[real_mask], threshold)
    report["real_eng"] = slice_metrics(yt[real_mask & (language == "eng")],
                                        yp[real_mask & (language == "eng")], threshold)
    report["real_spa"] = slice_metrics(yt[real_mask & (language == "spa")],
                                        yp[real_mask & (language == "spa")], threshold)
    synth_only_lang_mask = ~np.isin(language, list(REAL_LANGUAGES))
    report["synthetic_only_languages"] = slice_metrics(yt[synth_only_lang_mask], yp[synth_only_lang_mask], threshold)

    per_source = {}
    for src in sorted(set(dataset.tolist())):
        m = dataset == src
        per_source[src] = slice_metrics(yt[m], yp[m], threshold)
    report["per_source"] = per_source
    aucs = [v["auc"] for v in per_source.values() if v.get("auc") is not None]
    report["per_source_macro_auc"] = round(float(np.mean(aucs)), 4) if aucs else None

    report["calibration"] = calibration_metrics(yt, yp)
    report["threshold"] = round(float(threshold), 4)
    return report
