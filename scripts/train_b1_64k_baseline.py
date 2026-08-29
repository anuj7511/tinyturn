"""
Step 10 planning -- 64k escalation: B1 baseline, no pause events. Originally seed 42 only, per the
user's instruction ("Run one 64k baseline, seed 42 only, with the same validation/calibration split
and protocol"); now generalized with --seed for the follow-up 3-seed confirmation at 64k (seeds
43/44), since that gate cleared (+1.82pp overall / +1.76pp real AUC over 32k). Exact same protocol as
the 32k/16k baseline checkpoints (epochs<=40, early_stop_patience=6, lr_schedule=plateau,
batch_size=64, lr=1e-3, num_workers=6, only seed differs) -- only the train-split size differs
(28,114 -> ~61k clips, via `build_data_scale_tier.py --shards 5,15,25,35,45,55,60,65,70,75,80`). Fixed
val/calib splits are untouched throughout.

Usage:
  python scripts/train_b1_64k_baseline.py               # seed 42 (already run)
  python scripts/train_b1_64k_baseline.py --seed 43
  python scripts/train_b1_64k_baseline.py --seed 44
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.train import ExperimentConfig, train_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    exp_id = "B1_1s_64k_baseline" if args.seed == 42 else f"B1_1s_64k_baseline_seed{args.seed}"
    cfg = ExperimentConfig(
        exp_id=exp_id, context_s=1.0, use_trajectory=True, use_f0=False,
        epochs=40, early_stop_patience=6, lr_schedule="plateau",
        batch_size=64, lr=1e-3, num_workers=6, seed=args.seed,
    )
    out_dir = Path("experiments") / exp_id
    train_experiment(cfg, out_dir)


if __name__ == "__main__":
    main()
