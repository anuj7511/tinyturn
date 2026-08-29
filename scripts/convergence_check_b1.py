"""
Phase-2 8h -- confirm B1 has actually converged.

B0/B1/A0 shared the same epoch budget/schedule inherited from A0's protocol (fixed epoch count,
fixed LR, no early stopping) -- there's no reason a randomly-initialized model (B1) converges at the
same rate as a pretrained one (A0). This:

1. Reports best-epoch vs. final-epoch for the existing mel_trajectory_1s_speech_aligned_contract run (5 fixed epochs,
   Phase-2 8d's retrain) and states plainly whether val AUC was still rising at the final epoch.
2. Runs ONE longer B1@1s training with early stopping and its own LR schedule (ReduceLROnPlateau --
   distinct from A0's fixed-lr protocol), writing to a NEW directory so the qualifying
   mel_trajectory_1s_speech_aligned_contract checkpoint is untouched.
3. Compares the longer run's best val AUC against the original fixed-epoch run's.

Usage:
  python scripts/convergence_check_b1.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.train import ExperimentConfig, train_experiment

ORIGINAL_DIR = Path("experiments") / "mel_trajectory_1s_speech_aligned_contract"
LONGRUN_DIR = Path("experiments") / "mel_trajectory_1s_earlystopped_longrun"


def analyze_existing_run():
    metrics = json.load(open(ORIGINAL_DIR / "metrics.json"))
    history = metrics["history"]
    # Derived from history directly, not metrics["best_epoch"] etc. -- this run predates train.py's
    # 8h fields (best_epoch/final_epoch/best_val_auc/stopped_early), so those keys aren't present.
    best_h = max(history, key=lambda h: h["val_auc"])
    best_epoch = best_h["epoch"]
    best_val_auc = best_h["val_auc"]
    final_epoch = history[-1]["epoch"]
    print("=== Existing mel_trajectory_1s_speech_aligned_contract run (fixed 5-epoch protocol inherited from A0) ===")
    for h in history:
        marker = "  <- best" if h["epoch"] == best_epoch else ""
        print(f"  epoch {h['epoch']}: val_auc={h['val_auc']:.4f} loss={h['train_loss']:.4f}{marker}")
    print(f"best_epoch={best_epoch} (val_auc={best_val_auc:.4f}), final_epoch={final_epoch} "
          f"(val_auc={history[-1]['val_auc']:.4f})")
    print(f"val AUC was {'still rising' if best_epoch == final_epoch else 'NOT still rising'} "
          f"at the final epoch (best came {final_epoch - best_epoch} epoch(s) before the end).")
    metrics["best_epoch"] = best_epoch
    metrics["final_epoch"] = final_epoch
    metrics["best_val_auc"] = best_val_auc
    return metrics


def run_longer_b1(seed: int = 42):
    cfg = ExperimentConfig(
        exp_id="mel_trajectory_1s_earlystopped_longrun", context_s=1.0, use_trajectory=True, use_f0=False,
        epochs=40, early_stop_patience=6, lr_schedule="plateau",
        batch_size=64, num_workers=2, seed=seed,
    )
    return train_experiment(cfg, LONGRUN_DIR)


def main():
    if not (ORIGINAL_DIR / "metrics.json").exists():
        print(f"ERROR: {ORIGINAL_DIR / 'metrics.json'} not found -- run retrain_b1_1s_speech_aligned_contract.py first.")
        sys.exit(1)

    original = analyze_existing_run()
    print("\n=== Running longer B1@1s (early stopping, patience=6, ReduceLROnPlateau, max 40 epochs) ===")
    longrun_report = run_longer_b1()

    print("\n=== Comparison ===")
    orig_best = original["best_val_auc"]
    long_best = longrun_report["best_val_auc"]
    print(f"original (fixed 5 epochs): best_val_auc={orig_best:.4f} @ epoch {original['best_epoch']}")
    print(f"longer run (early-stopped): best_val_auc={long_best:.4f} @ epoch {longrun_report['best_epoch']}, "
          f"stopped at epoch {longrun_report['final_epoch']} "
          f"({'early-stopped' if longrun_report['stopped_early'] else 'hit max epochs'})")
    delta_pp = (long_best - orig_best) * 100
    print(f"delta: {delta_pp:+.2f}pp real/overall val AUC from training longer with its own schedule")

    summary = {
        "original_fixed_epochs": {"best_epoch": original["best_epoch"], "final_epoch": original["final_epoch"],
                                   "best_val_auc": orig_best, "history": original["history"]},
        "longer_early_stopped": {"best_epoch": longrun_report["best_epoch"],
                                  "final_epoch": longrun_report["final_epoch"],
                                  "stopped_early": longrun_report["stopped_early"],
                                  "best_val_auc": long_best, "history": longrun_report["history"]},
        "delta_pp": round(delta_pp, 3),
    }
    with open(Path("experiments") / "convergence_check_mel_trajectory.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nsaved experiments/convergence_check_mel_trajectory.json")


if __name__ == "__main__":
    main()
