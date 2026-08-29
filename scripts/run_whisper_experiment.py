"""CLI entry point for A0 (Whisper-Tiny baseline) training. Separate from run_experiment.py because
it drives tinyturn.train_whisper rather than tinyturn.train (different dataset/model, same protocol).

Usage:
  python scripts/run_whisper_experiment.py --context-s 4.0 --epochs 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.train_whisper import WhisperExperimentConfig, train_whisper_experiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--context-s", type=float, default=4.0)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()

    cfg = WhisperExperimentConfig(
        exp_id="A0", context_s=args.context_s, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, num_workers=args.num_workers,
    )
    out_dir = Path("experiments") / "A0_whisper_tiny"
    train_whisper_experiment(cfg, out_dir)


if __name__ == "__main__":
    main()
