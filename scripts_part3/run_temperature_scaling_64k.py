"""
Step 10 planning, item 6 -- explicit temperature scaling + final threshold selection on calibration
data only, for the 6 64k checkpoints (baseline + lambda=0.5 50:50, seeds 42/43/44). Distinct from
each run's own per-checkpoint `calibrate_threshold` call (which already picks a threshold on calib
for target FCR<=0.05, but off *raw*, uncalibrated sigmoid outputs) -- this fits a single scalar
temperature T per checkpoint (Guo et al. 2017 style: minimize NLL over T on calib logits, labels held
fixed) BEFORE selecting the threshold, so the emitted probability is actually interpretable as a
confidence estimate, not just an internally-consistent ranking score. Motivated directly by Section
6c's finding that `lambda=0.5 50:50`'s raw ECE is very poor (up to 0.196) and much worse than the
baseline's.

Split discipline: T is fit on CALIB ONLY (never val, never test). The resulting threshold is also
selected on CALIB ONLY, off the temperature-scaled probabilities. Val is used only to report the
effect (calibration metrics before/after), never to select T or the threshold.

Expected/verified property, not assumed: temperature scaling is a strictly monotonic transform of
the logit (T > 0), so it cannot change any rank-based metric -- AUC, ROC-derived recall/FCR at any
matched operating point are identical before and after. The threshold VALUE changes (it's now a
threshold on a different, calibrated probability space), but the operating point it selects (recall/
FCR on calib) does not. This script verifies that explicitly on val rather than asserting it.

Usage:
  python scripts_part3/run_temperature_scaling_64k.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.dataset import TinyTurnDataset, collate
from tinyturn.models import TinyTurnModel
from tinyturn.evaluate import calibrate_threshold, calibration_metrics, recall_at_fixed_fcr
from tinyturn.train import TRAJECTORY_NAMES

OUT_PATH = Path("experiments") / "temperature_scaling_64k.json"
TARGET_FCR = 0.05

CHECKPOINTS = {
    "B1_64k_baseline": {
        42: "experiments/B1_1s_64k_baseline",
        43: "experiments/B1_1s_64k_baseline_seed43",
        44: "experiments/B1_1s_64k_baseline_seed44",
    },
    "B1_64k_lambda0.5_5050": {
        42: "experiments/B1_1s_64k_lambda0.5_5050",
        43: "experiments/B1_1s_64k_lambda0.5_5050_seed43",
        44: "experiments/B1_1s_64k_lambda0.5_5050_seed44",
    },
}


def _load_model(ckpt_dir: Path):
    cfg = json.load(open(ckpt_dir / "config.json"))
    model = TinyTurnModel(n_mels=40, trajectory_dim=len(TRAJECTORY_NAMES),
                           mel_channels=cfg.get("mel_channels", 112), traj_channels=cfg.get("traj_channels", 24))
    model.load_state_dict(torch.load(ckpt_dir / "checkpoint.pt", map_location="cpu"))
    model.eval()
    return model


@torch.no_grad()
def _raw_logits(model, loader) -> tuple:
    logits, labels = [], []
    for batch in loader:
        log_mel = batch["log_mel"]
        mask = batch["valid_frame_mask"]
        traj = batch["trajectory"]
        out = model(log_mel, mask, traj)
        logits.extend(out.numpy().tolist())
        labels.extend(batch["label"].numpy().tolist())
    return np.array(logits, dtype=np.float64), np.array(labels, dtype=np.float64)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Single-scalar temperature scaling (Guo et al. 2017): minimize BCE(logit/T, label) over T>0,
    fit on calib only. Parameterize as log(T) so the optimizer can't drive T<=0."""
    logits_t = torch.tensor(logits, dtype=torch.float64)
    labels_t = torch.tensor(labels, dtype=torch.float64)
    log_T = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        T = torch.exp(log_T)
        loss = F.binary_cross_entropy_with_logits(logits_t / T, labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_T).item())


def audit_one(ckpt_dir: Path, calib_loader, val_loader) -> dict:
    model = _load_model(ckpt_dir)
    calib_logits, calib_labels = _raw_logits(model, calib_loader)
    val_logits, val_labels = _raw_logits(model, val_loader)

    T = fit_temperature(calib_logits, calib_labels)

    calib_prob_raw = 1 / (1 + np.exp(-calib_logits))
    calib_prob_temp = 1 / (1 + np.exp(-calib_logits / T))
    val_prob_raw = 1 / (1 + np.exp(-val_logits))
    val_prob_temp = 1 / (1 + np.exp(-val_logits / T))

    threshold_raw = calibrate_threshold(calib_labels, calib_prob_raw, target_fcr=TARGET_FCR)
    threshold_temp = calibrate_threshold(calib_labels, calib_prob_temp, target_fcr=TARGET_FCR)

    # Verification: the operating point (recall at matched FCR) selected on CALIB must be identical
    # before/after temperature scaling, since T-scaling is a monotonic transform of the logit -- and
    # the recall this threshold achieves on VAL (out-of-sample) must therefore also match.
    recall_calib_raw = recall_at_fixed_fcr(calib_labels, calib_prob_raw, TARGET_FCR)
    recall_calib_temp = recall_at_fixed_fcr(calib_labels, calib_prob_temp, TARGET_FCR)
    val_recall_raw = float((val_prob_raw >= threshold_raw)[val_labels == 1].mean()) if (val_labels == 1).any() else None
    val_recall_temp = float((val_prob_temp >= threshold_temp)[val_labels == 1].mean()) if (val_labels == 1).any() else None

    return {
        "temperature": round(T, 4),
        "threshold_raw": round(float(threshold_raw), 4),
        "threshold_temp_scaled": round(float(threshold_temp), 4),
        "calibration_calib_raw": calibration_metrics(calib_labels, calib_prob_raw),
        "calibration_calib_temp_scaled": calibration_metrics(calib_labels, calib_prob_temp),
        "calibration_val_raw": calibration_metrics(val_labels, val_prob_raw),
        "calibration_val_temp_scaled": calibration_metrics(val_labels, val_prob_temp),
        "operating_point_check": {
            "recall_at_fcr05_calib_raw": round(recall_calib_raw, 4),
            "recall_at_fcr05_calib_temp_scaled": round(recall_calib_temp, 4),
            "val_recall_at_own_threshold_raw": round(val_recall_raw, 4) if val_recall_raw is not None else None,
            "val_recall_at_own_threshold_temp_scaled": round(val_recall_temp, 4) if val_recall_temp is not None else None,
            "note": "raw vs temp-scaled should match (monotonic transform) -- confirms T doesn't "
                    "change the decision rule, only the probability's calibration quality.",
        },
    }


def main():
    ds_kwargs = dict(context_s=1.0, include_trajectory=True)
    calib_ds = TinyTurnDataset(split="calib", **ds_kwargs)
    val_ds = TinyTurnDataset(split="val", **ds_kwargs)
    calib_loader = DataLoader(calib_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)

    results = {}
    for arm, seeds in CHECKPOINTS.items():
        results[arm] = {}
        for seed, path in seeds.items():
            d = Path(path)
            if not (d / "checkpoint.pt").exists():
                print(f"SKIP {arm} seed={seed}: {d} missing checkpoint.pt")
                continue
            print(f"temperature-scaling {arm} seed={seed} ({d})...", flush=True)
            results[arm][seed] = audit_one(d, calib_loader, val_loader)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved {OUT_PATH}")

    print("\n=== temperature scaling summary ===")
    for arm, seeds in results.items():
        for seed, r in seeds.items():
            print(f"{arm} seed={seed}: T={r['temperature']} "
                  f"threshold {r['threshold_raw']} -> {r['threshold_temp_scaled']}  "
                  f"val ECE {r['calibration_val_raw']['ece']:.4f} -> {r['calibration_val_temp_scaled']['ece']:.4f}  "
                  f"val Brier {r['calibration_val_raw']['brier']:.4f} -> {r['calibration_val_temp_scaled']['brier']:.4f}  "
                  f"op-point check: val_recall {r['operating_point_check']['val_recall_at_own_threshold_raw']} vs "
                  f"{r['operating_point_check']['val_recall_at_own_threshold_temp_scaled']}")


if __name__ == "__main__":
    main()
