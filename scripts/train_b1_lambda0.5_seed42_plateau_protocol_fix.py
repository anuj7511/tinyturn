"""
Fix a protocol mismatch discovered while assembling the 3-seed confirmation table for the
hold-loss objective.

The seed-42 checkpoints for the proportional and 50:50 sampling arms of
experiments/pause_event_sampling_comparison/ were trained under the OLD fixed-5-epoch protocol
(epochs=5, no early stopping, no lr schedule) -- the same arms at seeds 43/44, and the baseline at
every seed, all use the plateau/early-stopping protocol instead (epochs=40, early_stop_patience=6,
lr_schedule=plateau). A 3-seed average requires identical training and early-stopping across seeds,
so seed 42's two lambda=0.5 arms need retraining under the matching protocol before the 3-seed table
is meaningful.

Usage:
  python scripts/train_b1_lambda0.5_seed42_plateau_protocol_fix.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.train_p1 import P1Config, train_p1

BASELINE_CHECKPOINT = Path("experiments/pause_event_sampling_comparison") / "baseline_no_pause_events_seed42" / "checkpoint.pt"


def main():
    for balance, exp_name in [
        ("proportional", "pause_events_holdloss0.5_proportional_seed42"),
        ("50:50", "pause_events_holdloss0.5_5050sampling_seed42"),
    ]:
        cfg = P1Config(
            exp_id=exp_name, context_s=1.0,
            epochs=40, early_stop_patience=6, lr_schedule="plateau",
            batch_size=64, lr=1e-3, num_workers=4, seed=42,
            lambda_hold=0.5, controlled_sampling=True, real_synth_balance=balance,
        )
        out_dir = Path("experiments/pause_event_sampling_comparison") / cfg.exp_id
        train_p1(cfg, BASELINE_CHECKPOINT, out_dir)


if __name__ == "__main__":
    main()
