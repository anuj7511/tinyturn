"""
Step 10 planning, correction 1 -- matched-recall hold-FCR audit, calibration-then-validation.

The 3-seed table in PHASE3_RESULTS_step10_distillation_ranking.md Section 3 reported hold FCR at
each arm's own independently-calibrated threshold (calibrated for target FCR=0.05 on final clips).
That's not comparable across arms with different calibration curves, and it isn't even the same
threshold-selection method the earlier keep/promotion rules used (matched complete-turn recall).
It can flip a ranking: ranking (real AUC 0.7345) and lambda=0.5-all (real AUC 0.6998) have nearly
tied real-hold FCR at their own thresholds (11.3% vs 11.0%) -- a much closer call than the headline
real-AUC gap suggests.

This script fixes BOTH problems in the existing `run_step9_controlled_rerun.py` precedent:
  1. Uses matched recall (90%/95% on the complete class) instead of each arm's own FCR-calibrated
     threshold, exactly like that script did.
  2. UNLIKE that script (which selects the threshold from `val`'s own recall curve, then reads
     hold-FCR from that same `val` split -- circular), this selects the threshold from the
     CALIBRATION split's recall curve, then evaluates recall/hold-FCR/extra-slice-recall on
     VALIDATION only. No threshold here is ever chosen and evaluated on the same split.

Runs against all 12 already-trained checkpoints (4 arms x 3 seeds) -- inference only, no training.

Usage:
  python scripts_part3/run_matched_recall_audit.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.dataset import TinyTurnDataset, collate
from tinyturn.pause_events import PauseEventDataset
from tinyturn.models import TinyTurnModel
from tinyturn.evaluate import run_inference
from tinyturn.train import TRAJECTORY_NAMES

CACHE_DIR = Path("data_cache")
OUT_PATH = Path("experiments") / "matched_recall_audit_calib_then_val.json"
CONTEXT_S = 1.0
TARGET_RECALLS = [0.90, 0.95]
CURVE_GRID = np.linspace(0.50, 0.99, 50)
SHORT_COMPLETE_MAX_WORDS = 3
RESPONSE_PARTICLES = {
    "okay", "ok", "yeah", "yes", "yep", "yup", "alright", "right", "sure",
    "haan", "bas", "mm", "mhm", "uh-huh", "uhhuh", "no", "nope",
}

CHECKPOINTS = {
    "B1_baseline": {
        42: "step9_results_updated/baseline_kaggle",
        43: "step9_results_updated/baseline_no_pause_events_seed43",
        44: "step9_results_updated/baseline_no_pause_events_seed44",
    },
    "ranking": {
        42: "experiments/B1_1s_ranking_seed42_plateau",
        43: "experiments/B1_1s_ranking_seed43_plateau",
        44: "experiments/B1_1s_ranking_seed44_plateau",
    },
    "lambda0.5_all": {
        42: "experiments/P1ab_lambda0.5_all_seed42_plateau",
        43: "step9_results_updated/P1ab_lambda0.5_all_seed43",
        44: "step9_results_updated/P1ab_lambda0.5_all_seed44",
    },
    "lambda0.5_5050": {
        42: "experiments/P1ab_lambda0.5_5050_seed42_plateau",
        43: "step9_results_updated/P1ab_lambda0.5_5050_seed43",
        44: "step9_results_updated/P1ab_lambda0.5_5050_seed44",
    },
}


def _load_model(ckpt_dir: Path):
    cfg = json.load(open(ckpt_dir / "config.json"))
    model = TinyTurnModel(n_mels=40, trajectory_dim=len(TRAJECTORY_NAMES),
                           mel_channels=cfg.get("mel_channels", 112), traj_channels=cfg.get("traj_channels", 24))
    model.load_state_dict(torch.load(ckpt_dir / "checkpoint.pt", map_location="cpu"))
    model.eval()
    return model


def _pause_probs_and_synth(model, pause_loader, device):
    y_prob, synthetic = [], []
    with torch.no_grad():
        for batch in pause_loader:
            log_mel = batch["log_mel"].to(device)
            mask = batch["valid_frame_mask"].to(device)
            traj = batch["trajectory"].to(device)
            logits = model(log_mel, mask, traj)
            y_prob.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            synthetic.extend(batch["synthetic"])
    return np.array(y_prob), np.array(synthetic, dtype=bool)


def threshold_at_recall(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float):
    """Highest (most conservative) threshold that still achieves >= target_recall on y_true==1,
    matching tinyturn.evaluate.fcr_at_fixed_recall's convention (min fpr among tpr>=target) but
    also returning the actual threshold value, not just the resulting fpr."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ok = tpr >= target_recall
    idx = int(np.argmin(np.where(ok, fpr, np.inf))) if ok.any() else int(np.argmax(tpr))
    return float(thresholds[idx]), float(tpr[idx]), float(fpr[idx])


def _has_response_particle(text) -> bool:
    if not isinstance(text, str):
        return False
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return any(w in RESPONSE_PARTICLES for w in words)


def _recall_at_threshold(probs, ids, complete, mask, threshold):
    sub = mask & complete
    n = int(sub.sum())
    if n == 0:
        return {"n": 0, "recall": None}
    return {"n": n, "recall": round(float((probs[sub] >= threshold).mean()), 4)}


def audit_one(ckpt_dir: Path, calib_loader, val_loader, pause_loader, val_ds, trans_df, device):
    model = _load_model(ckpt_dir)
    calib_out = run_inference(model, calib_loader, device, use_trajectory=True)
    val_out = run_inference(model, val_loader, device, use_trajectory=True)
    pause_probs, pause_synth = _pause_probs_and_synth(model, pause_loader, device)
    real_mask, synth_mask = ~pause_synth, pause_synth

    val_complete = val_out.y_true.astype(bool)
    val_implicit = np.array(val_out.implicit_incomplete, dtype=bool)
    ids = np.array(val_out.ids)
    meta = trans_df.set_index("id")
    n_words = np.array([meta["n_words"].get(i, np.nan) for i in ids])
    has_particle = np.array([_has_response_particle(meta["text"].get(i)) for i in ids])
    short_mask = n_words <= SHORT_COMPLETE_MAX_WORDS

    def _eval_at_threshold(t):
        pred_val = val_out.y_prob >= t
        return {
            "threshold_from_calib": t,
            "actual_recall_complete_val": (float(pred_val[val_complete].mean())
                                            if val_complete.any() else None),
            "fcr_implicit_incomplete_val": (float(pred_val[val_implicit].mean())
                                             if val_implicit.any() else None),
            "hold_fcr_all": float((pause_probs >= t).mean()) if len(pause_probs) else None,
            "hold_fcr_real": float((pause_probs[real_mask] >= t).mean()) if real_mask.any() else None,
            "hold_fcr_synthetic": float((pause_probs[synth_mask] >= t).mean()) if synth_mask.any() else None,
            "short_complete_recall": _recall_at_threshold(val_out.y_prob, ids, val_complete, short_mask, t),
            "response_particle_complete_recall": _recall_at_threshold(val_out.y_prob, ids, val_complete, has_particle, t),
        }

    matched = {}
    for target in TARGET_RECALLS:
        t, actual_recall_calib, _ = threshold_at_recall(calib_out.y_true, calib_out.y_prob, target)
        entry = {"target_recall_calib": target, "actual_recall_calib": actual_recall_calib}
        entry.update(_eval_at_threshold(t))
        matched[f"recall_{int(target*100)}"] = entry

    curve_thresholds = [threshold_at_recall(calib_out.y_true, calib_out.y_prob, g)[0] for g in CURVE_GRID]
    curve = {"target_recall_calib_grid": CURVE_GRID.tolist(), "thresholds": curve_thresholds,
             "recall_complete_val": [], "fcr_implicit_incomplete_val": [],
             "hold_fcr_all": [], "hold_fcr_real": [], "hold_fcr_synthetic": []}
    for t in curve_thresholds:
        e = _eval_at_threshold(t)
        curve["recall_complete_val"].append(e["actual_recall_complete_val"])
        curve["fcr_implicit_incomplete_val"].append(e["fcr_implicit_incomplete_val"])
        curve["hold_fcr_all"].append(e["hold_fcr_all"])
        curve["hold_fcr_real"].append(e["hold_fcr_real"])
        curve["hold_fcr_synthetic"].append(e["hold_fcr_synthetic"])

    return {"overall_auc": None, "matched": matched, "curve": curve,
            "n_val_pause_real": int(real_mask.sum()), "n_val_pause_synthetic": int(synth_mask.sum())}


def main():
    ds_kwargs = dict(context_s=CONTEXT_S, include_trajectory=True)
    calib_ds = TinyTurnDataset(split="calib", **ds_kwargs)
    val_ds = TinyTurnDataset(split="val", **ds_kwargs)
    val_pause_ds = PauseEventDataset(split="val", context_s=CONTEXT_S, include_trajectory=True)
    calib_loader = DataLoader(calib_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
    pause_loader = DataLoader(val_pause_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
    trans_df = pd.read_parquet(CACHE_DIR / "d2_stratified_transcripts.parquet")[["id", "text", "n_words"]]
    device = torch.device("cpu")

    results = {}
    for arm, seeds in CHECKPOINTS.items():
        results[arm] = {}
        for seed, path in seeds.items():
            d = Path(path)
            if not (d / "checkpoint.pt").exists():
                print(f"SKIP {arm} seed={seed}: {d} missing checkpoint.pt")
                continue
            print(f"auditing {arm} seed={seed} ({d})...", flush=True)
            results[arm][seed] = audit_one(d, calib_loader, val_loader, pause_loader, val_ds, trans_df, device)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved {OUT_PATH}")

    print("\n=== matched-recall (calib) -> hold-FCR (val) summary, mean +/- std across seeds ===")
    for arm, seeds in results.items():
        for target_key in ["recall_90", "recall_95"]:
            vals_all = [s["matched"][target_key]["hold_fcr_all"] for s in seeds.values()]
            vals_real = [s["matched"][target_key]["hold_fcr_real"] for s in seeds.values()]
            vals_syn = [s["matched"][target_key]["hold_fcr_synthetic"] for s in seeds.values()]
            actual_recall = [s["matched"][target_key]["actual_recall_complete_val"] for s in seeds.values()]
            print(f"{arm} @ {target_key}: actual_val_recall={np.mean(actual_recall):.4f} "
                  f"hold_fcr_all={np.mean(vals_all):.4f}+/-{np.std(vals_all):.4f} "
                  f"hold_fcr_real={np.mean(vals_real):.4f}+/-{np.std(vals_real):.4f} "
                  f"hold_fcr_synth={np.mean(vals_syn):.4f}+/-{np.std(vals_syn):.4f}")


if __name__ == "__main__":
    main()
