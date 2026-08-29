"""CLI entry point for Step 7 / P1 (final clips + internal-pause continuation events).

Superseded by scripts/train_pause_event_refinement.py for Phase-2 Step 9 (P1a/P1b) runs --
kept import-compatible with train_p1.py's P1Config (Phase-2 Step 9) since this script still runs
Step 7's original unweighted-blend recipe (lambda_hold=None, controlled_sampling=False) against
whatever baseline checkpoint you point it at.

Usage:
  python scripts/train_pause_events_p1.py --context-s 1.0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.train_p1 import P1Config, train_p1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--context-s", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--baseline-checkpoint", type=str,
                   default="experiments/mel_trajectory_1s_speech_aligned_contract/checkpoint.pt")
    args = p.parse_args()

    cfg = P1Config(
        exp_id="P1", context_s=args.context_s,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, num_workers=args.num_workers,
    )
    train_p1(cfg, Path(args.baseline_checkpoint), Path("experiments/pause_events_original_recipe"))


if __name__ == "__main__":
    main()
