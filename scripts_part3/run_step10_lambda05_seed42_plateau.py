"""
Step 10 planning, item 5 prerequisite -- fix a protocol mismatch discovered while assembling the
3-seed confirmation table.

step9_results_updated/P1ab_lambda0.5_{all,5050} (seed 42) were trained under the OLD fixed-5-epoch
protocol (epochs=5, no early stopping, no lr schedule) -- the same P1ab_lambda0.5_{all,5050}_seed{43,44}
arms, and the baseline at every seed (baseline_kaggle / baseline_no_pause_events_seed{43,44}), all use
the 8h-validated protocol instead (epochs=40, early_stop_patience=6, lr_schedule=plateau). The plan's
step 5 requires "identical training and early stopping" across seeds and explicitly gates reuse of an
existing seed-42 run on "if their manifests and code hashes match" -- they don't here, so seed 42's
lambda=0.5 arms need retraining under the matching protocol before a 3-seed table is meaningful.

Usage:
  python scripts_part3/run_step10_lambda05_seed42_plateau.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.train_p1 import P1Config, train_p1

BASELINE_CHECKPOINT = Path("step9_results_updated") / "baseline_kaggle" / "checkpoint.pt"


def main():
    for balance, suffix in [("proportional", "all"), ("50:50", "5050")]:
        cfg = P1Config(
            exp_id=f"P1ab_lambda0.5_{suffix}_seed42_plateau", context_s=1.0,
            epochs=40, early_stop_patience=6, lr_schedule="plateau",
            batch_size=64, lr=1e-3, num_workers=4, seed=42,
            lambda_hold=0.5, controlled_sampling=True, real_synth_balance=balance,
        )
        out_dir = Path("experiments") / cfg.exp_id
        train_p1(cfg, BASELINE_CHECKPOINT, out_dir)


if __name__ == "__main__":
    main()
