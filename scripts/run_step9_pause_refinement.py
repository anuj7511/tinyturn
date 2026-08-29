"""
Phase-2 Step 9 -- pause-event refinement.

1. "Before anything else": retrain a fresh, contract-correct P1 (Step 7's original unweighted
   blend, `lambda_hold=None`) against the C1_B1_1s_pv2speechend baseline (Phase-2 8d), then plot
   the *matched-threshold* comparison the brief asks for: recall on complete final turns (x-axis)
   vs. FCR on internal holds (y-axis), sweeping threshold for each model, plus
   `implicit_incomplete` FCR at the same matched thresholds. This is the single-point comparison
   from Step 7 (STATUS_REPORT) generalized to a full curve, so an apparent P1 win can't just be an
   artifact of which threshold each model happened to be calibrated at.
2. P1a x P1b: three lambda_hold values (0.1, 0.25, 0.5), each with controlled_sampling=True
   (P1b), compared against the plain-P1 and baseline curves.

Usage:
  python scripts/run_step9_pause_refinement.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.dataset import TinyTurnDataset, collate
from tinyturn.pause_events import PauseEventDataset
from tinyturn.models import TinyTurnModel
from tinyturn.evaluate import run_inference
from tinyturn.train import TRAJECTORY_NAMES
from tinyturn.train_p1 import P1Config, train_p1
from torch.utils.data import DataLoader

BASELINE_DIR = Path("experiments") / "C1_B1_1s_pv2speechend"
PLAIN_P1_DIR = Path("experiments") / "P1_pause_events_pv2speechend"
OUT_DIR = Path("experiments") / "step9_pause_refinement"
CONTEXT_S = 1.0
TARGET_RECALLS = [0.90, 0.95]


def _load_model(ckpt_dir: Path, cfg_overrides=None):
    cfg = json.load(open(ckpt_dir / "config.json"))
    model = TinyTurnModel(n_mels=40, trajectory_dim=len(TRAJECTORY_NAMES),
                           mel_channels=cfg.get("mel_channels", 112), traj_channels=cfg.get("traj_channels", 24))
    model.load_state_dict(torch.load(ckpt_dir / "checkpoint.pt", map_location="cpu"))
    model.eval()
    return model


def _pause_probs(model, pause_loader, device):
    y_prob = []
    with torch.no_grad():
        for batch in pause_loader:
            log_mel = batch["log_mel"].to(device)
            mask = batch["valid_frame_mask"].to(device)
            traj = batch["trajectory"].to(device)
            logits = model(log_mel, mask, traj)
            y_prob.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.array(y_prob)


def matched_threshold_curve(model, val_loader, pause_loader, device, n_thresholds=201):
    val_out = run_inference(model, val_loader, device, use_trajectory=True)
    pause_probs = _pause_probs(model, pause_loader, device)

    complete_mask = val_out.y_true.astype(bool)
    implicit_mask = np.array(val_out.implicit_incomplete, dtype=bool)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    recall_complete, fcr_implicit, fcr_holds = [], [], []
    for t in thresholds:
        pred = val_out.y_prob >= t
        recall_complete.append(float(pred[complete_mask].mean()) if complete_mask.any() else float("nan"))
        fcr_implicit.append(float(pred[implicit_mask].mean()) if implicit_mask.any() else float("nan"))
        fcr_holds.append(float((pause_probs >= t).mean()) if len(pause_probs) else float("nan"))
    return {"thresholds": thresholds.tolist(), "recall_complete": recall_complete,
            "fcr_implicit_incomplete": fcr_implicit, "fcr_holds": fcr_holds,
            "n_val_complete": int(complete_mask.sum()), "n_val_implicit_incomplete": int(implicit_mask.sum()),
            "n_val_pause_holds": int(len(pause_probs))}


def at_matched_recall(curve: dict, target_recall: float) -> dict:
    recalls = np.array(curve["recall_complete"])
    idx = int(np.argmin(np.abs(recalls - target_recall)))
    return {"target_recall": target_recall, "threshold": curve["thresholds"][idx],
            "actual_recall_complete": recalls[idx], "fcr_holds": curve["fcr_holds"][idx],
            "fcr_implicit_incomplete": curve["fcr_implicit_incomplete"][idx]}


def plot_curves(curves: dict, out_path: Path):
    plt.figure(figsize=(7, 5.5))
    for name, curve in curves.items():
        plt.plot(curve["recall_complete"], curve["fcr_holds"], marker=".", markersize=2, label=name)
    plt.xlabel("Recall on complete final turns")
    plt.ylabel("FCR on internal holds")
    plt.title("Step 9: matched-threshold recall (final turns) vs. FCR (internal holds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_curve_for_dir(ckpt_dir: Path, val_loader, pause_loader, device):
    model = _load_model(ckpt_dir)
    return matched_threshold_curve(model, val_loader, pause_loader, device)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    if not (BASELINE_DIR / "checkpoint.pt").exists():
        print(f"ERROR: {BASELINE_DIR / 'checkpoint.pt'} not found -- run run_8d_retrain_b1_1s.py first.")
        sys.exit(1)

    print("=== Step 9, part 1: retrain plain P1 (Step 7 recipe) under the corrected contract ===", flush=True)
    if not (PLAIN_P1_DIR / "checkpoint.pt").exists():
        cfg = P1Config(exp_id="P1_pv2speechend", context_s=CONTEXT_S, epochs=5, batch_size=64,
                       num_workers=2, lambda_hold=None, controlled_sampling=False)
        train_p1(cfg, BASELINE_DIR / "checkpoint.pt", PLAIN_P1_DIR)
    else:
        print(f"{PLAIN_P1_DIR} already exists, skipping retrain", flush=True)

    print("\n=== Step 9, part 2: P1a x P1b sweep (lambda_hold in {0.1, 0.25, 0.5}, controlled sampling) ===",
          flush=True)
    p1ab_dirs = {}
    for lam in (0.1, 0.25, 0.5):
        tag = f"P1ab_lambda{lam}"
        d = Path("experiments") / f"step9_{tag}"
        p1ab_dirs[lam] = d
        if (d / "checkpoint.pt").exists():
            print(f"{d} already exists, skipping", flush=True)
            continue
        cfg = P1Config(exp_id=tag, context_s=CONTEXT_S, epochs=5, batch_size=64, num_workers=2,
                       lambda_hold=lam, controlled_sampling=True, real_synth_balance="proportional")
        train_p1(cfg, BASELINE_DIR / "checkpoint.pt", d)

    print("\n=== Step 9, part 3: matched-threshold curves for baseline / plain-P1 / P1a+P1b variants ===",
          flush=True)
    ds_kwargs = dict(context_s=CONTEXT_S, include_trajectory=True)
    val_ds = TinyTurnDataset(split="val", **ds_kwargs)
    val_pause_ds = PauseEventDataset(split="val", context_s=CONTEXT_S, include_trajectory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
    pause_loader = DataLoader(val_pause_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)

    curves = {}
    curves["baseline (no pause events)"] = build_curve_for_dir(BASELINE_DIR, val_loader, pause_loader, device)
    curves["P1 (plain, Step 7 recipe)"] = build_curve_for_dir(PLAIN_P1_DIR, val_loader, pause_loader, device)
    for lam, d in p1ab_dirs.items():
        curves[f"P1a+P1b (lambda_hold={lam})"] = build_curve_for_dir(d, val_loader, pause_loader, device)

    plot_path = OUT_DIR / "recall_vs_fcr_holds.png"
    plot_curves(curves, plot_path)
    print(f"saved plot to {plot_path}", flush=True)

    matched = {}
    for name, curve in curves.items():
        matched[name] = {f"recall_{int(r*100)}": at_matched_recall(curve, r) for r in TARGET_RECALLS}
        print(f"\n{name}:")
        for r in TARGET_RECALLS:
            m = matched[name][f"recall_{int(r*100)}"]
            print(f"  @ recall~{r:.0%} (actual {m['actual_recall_complete']:.3f}, "
                  f"threshold={m['threshold']:.3f}): fcr_holds={m['fcr_holds']:.4f} "
                  f"fcr_implicit_incomplete={m['fcr_implicit_incomplete']:.4f}")

    with open(OUT_DIR / "matched_threshold_results.json", "w") as f:
        json.dump({"curves": curves, "matched_recall_summary": matched}, f, indent=2, default=str)
    print(f"\nsaved {OUT_DIR / 'matched_threshold_results.json'}")


if __name__ == "__main__":
    main()
