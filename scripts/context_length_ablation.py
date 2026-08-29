"""
Step 6 -- context-length ablation on the finalists.

Tiny finalist: B1 (mel + trajectory fusion, no F0) -- chosen over B1-f0 because F0's accuracy gain
concentrated in the *synthetic* slice while its 10x latency cost bought nothing on the *real* slice
(Step 4's ablation), and over B0 because B1 wins on real-audio AUC and calibration.

Runs 1s/2s/4s for B1; reuses the existing B0-vs-B1-vs-A0 comparison's 4s runs (already trained in
Steps 3-5) as the "4s" point in each sweep rather than retraining them. Whisper gets 8s added only
if 4s still improves over 2s, per the brief's explicit instruction.

Usage:
  python scripts/context_length_ablation.py B1 --context-s 1.0
  python scripts/context_length_ablation.py B1 --context-s 2.0
  python scripts/context_length_ablation.py A0 --context-s 2.0
  python scripts/context_length_ablation.py A0 --context-s 8.0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["B1", "A0"])
    p.add_argument("--context-s", type=float, required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    ctx_tag = f"{args.context_s:g}s"
    if args.model == "B1":
        from tinyturn.train import ExperimentConfig, train_experiment
        cfg = ExperimentConfig(
            exp_id=f"C1_B1_{ctx_tag}", context_s=args.context_s, use_trajectory=True, use_f0=False,
            epochs=args.epochs or 5, batch_size=args.batch_size or 64, num_workers=args.num_workers,
        )
        out_dir = Path("experiments") / f"C1_B1_{ctx_tag}"
        train_experiment(cfg, out_dir)
    else:
        from tinyturn.train_whisper import WhisperExperimentConfig, train_whisper_experiment
        cfg = WhisperExperimentConfig(
            exp_id=f"C1_A0_{ctx_tag}", context_s=args.context_s,
            epochs=args.epochs or 2, batch_size=args.batch_size or 8, num_workers=args.num_workers,
        )
        out_dir = Path("experiments") / f"C1_A0_{ctx_tag}"
        train_whisper_experiment(cfg, out_dir)


if __name__ == "__main__":
    main()
