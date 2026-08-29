"""
Step 10 planning -- 64k escalation: chosen final objective (lambda=0.5, 50:50 real/synthetic
pause-event balance). Originally seed 42 only, run after train_b1_64k_baseline.py cleared the
user's gate (+1.82pp overall / +1.76pp real AUC over the 32k baseline); now generalized with --seed
for the follow-up 3-seed confirmation at 64k (seeds 43/44). Exact same protocol as the 32k/16k
`lambda0.5_5050` checkpoints (epochs<=40, early_stop_patience=6, lr_schedule=plateau, batch_size=64,
lr=1e-3, num_workers=6, lambda_hold=0.5, controlled_sampling=True, real_synth_balance="50:50") --
only the train-split size and seed differ. Compares FCR-at-holds against the SAME-SEED 64k baseline
(not seed 42's), so the "did the hold-aware objective's own baseline improve too" question stays
apples-to-apples per seed.

Usage:
  python scripts/train_b1_64k_lambda0.5_5050.py               # seed 42 (already run)
  python scripts/train_b1_64k_lambda0.5_5050.py --seed 43
  python scripts/train_b1_64k_lambda0.5_5050.py --seed 44
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.train_p1 import P1Config, train_p1

SEED_BASELINE_CHECKPOINTS = {
    42: Path("experiments") / "data_scale_64k_baseline_seed42" / "checkpoint.pt",
    43: Path("experiments") / "data_scale_64k_baseline_seed43" / "checkpoint.pt",
    44: Path("experiments") / "data_scale_64k_baseline_seed44" / "checkpoint.pt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    baseline_checkpoint = SEED_BASELINE_CHECKPOINTS[args.seed]
    if not baseline_checkpoint.exists():
        print(f"ERROR: {baseline_checkpoint} not found -- run train_b1_64k_baseline.py "
              f"--seed {args.seed} first.")
        sys.exit(1)
    exp_id = "data_scale_64k_holdloss0.5_5050sampling_seed42" if args.seed == 42 else f"data_scale_64k_holdloss0.5_5050sampling_seed42_seed{args.seed}"
    cfg = P1Config(
        exp_id=exp_id, context_s=1.0,
        epochs=40, early_stop_patience=6, lr_schedule="plateau",
        batch_size=64, lr=1e-3, num_workers=6, seed=args.seed,
        lambda_hold=0.5, controlled_sampling=True, real_synth_balance="50:50",
    )
    out_dir = Path("experiments") / exp_id
    train_p1(cfg, baseline_checkpoint, out_dir)


if __name__ == "__main__":
    main()
