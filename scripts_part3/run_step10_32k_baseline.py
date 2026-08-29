"""
Step 10 planning, item 4 -- 32k scaling: B1 baseline, no pause events.
Exact same protocol as the 16k `baseline_kaggle` checkpoint (epochs<=40, early_stop_patience=6,
lr_schedule=plateau, batch_size=64, lr=1e-3, num_workers=4, seed=42) -- only the train-split size
differs (12,797 -> 28,114 clips, via scripts_part3/build_32k_scale_tier.py). Fixed val/calib splits
are untouched.

Usage:
  python scripts_part3/run_step10_32k_baseline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.train import ExperimentConfig, train_experiment


def main():
    cfg = ExperimentConfig(
        exp_id="B1_1s_32k_baseline", context_s=1.0, use_trajectory=True, use_f0=False,
        epochs=40, early_stop_patience=6, lr_schedule="plateau",
        batch_size=64, lr=1e-3, num_workers=4, seed=42,
    )
    out_dir = Path("experiments") / "B1_1s_32k_baseline"
    train_experiment(cfg, out_dir)


if __name__ == "__main__":
    main()
