"""CLI entry point for B0 / B1 / B1-f0 training runs. Kept as a thin wrapper around
tinyturn.train.train_experiment so Windows' spawn-based multiprocessing DataLoader has a proper
`if __name__ == "__main__":` guard.

Usage:
  python scripts/train_mel_trajectory_model.py B0 --context-s 4.0
  python scripts/train_mel_trajectory_model.py B1 --context-s 4.0
  python scripts/train_mel_trajectory_model.py B1-f0 --context-s 4.0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.train import ExperimentConfig, train_experiment

EXPERIMENTS = {
    "B0": dict(use_trajectory=False, use_f0=False, dir_name="mel_only_baseline"),
    "B1": dict(use_trajectory=True, use_f0=False, dir_name="mel_trajectory_baseline"),
    "B1-f0": dict(use_trajectory=True, use_f0=True, dir_name="mel_trajectory_with_f0"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=list(EXPERIMENTS.keys()))
    p.add_argument("--context-s", type=float, default=4.0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()

    spec = EXPERIMENTS[args.experiment]
    cfg = ExperimentConfig(
        exp_id=args.experiment, context_s=args.context_s, use_trajectory=spec["use_trajectory"],
        use_f0=spec["use_f0"], epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        num_workers=args.num_workers,
    )
    out_dir = Path("experiments") / spec["dir_name"]
    train_experiment(cfg, out_dir)


if __name__ == "__main__":
    main()
