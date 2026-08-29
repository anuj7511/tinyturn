"""
Step 10 planning, item 4 -- 32k scaling: selected hold-aware objective (lambda=0.5, 50:50 real/
synthetic pause-event balance) -- the strongest real-hold-FCR arm per the matched-recall audit.
Exact same protocol as the 16k `P1ab_lambda0.5_5050_seed42_plateau` checkpoint (epochs<=40,
early_stop_patience=6, lr_schedule=plateau, batch_size=64, lr=1e-3, num_workers=4, seed=42,
lambda_hold=0.5, controlled_sampling=True, real_synth_balance="50:50") -- only the train-split size
differs. Run after run_step10_32k_baseline.py so the FCR-at-holds comparison is against the
32k-trained baseline, not the 16k one.

Usage:
  python scripts/run_step10_32k_lambda0.5_5050.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.train_p1 import P1Config, train_p1

BASELINE_CHECKPOINT = Path("experiments") / "B1_1s_32k_baseline" / "checkpoint.pt"


def main():
    if not BASELINE_CHECKPOINT.exists():
        print(f"ERROR: {BASELINE_CHECKPOINT} not found -- run run_step10_32k_baseline.py first.")
        sys.exit(1)
    cfg = P1Config(
        exp_id="B1_1s_32k_lambda0.5_5050", context_s=1.0,
        epochs=40, early_stop_patience=6, lr_schedule="plateau",
        batch_size=64, lr=1e-3, num_workers=4, seed=42,
        lambda_hold=0.5, controlled_sampling=True, real_synth_balance="50:50",
    )
    out_dir = Path("experiments") / "B1_1s_32k_lambda0.5_5050"
    train_p1(cfg, BASELINE_CHECKPOINT, out_dir)


if __name__ == "__main__":
    main()
