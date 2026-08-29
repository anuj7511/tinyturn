"""
Step 10 planning, correction 2 -- D0: distillation isolation control.

D1/D2 changed two things vs. the B1 baseline simultaneously: teacher logits AND student boundary
augmentation. D0 isolates the second variable alone -- hard labels only (alpha=1.0, so the soft/
teacher term in tinyturn.train_distill._distill_loss is multiplied by zero and contributes nothing),
same boundary augmentation, same exact D1/D2 protocol otherwise (T is irrelevant when alpha=1.0 but
kept for config-shape consistency). Compare D0 vs D1 vs D2 vs the untouched B1 baseline to attribute
D1/D2's real-AUC drop correctly before trusting "distillation failed" over "boundary augmentation on
B1 costs real AUC".

Usage:
  python scripts/train_distillation_isolation_control.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.train_distill import DistillConfig, train_distill


def main():
    cfg = DistillConfig(
        exp_id="distillation_isolation_control", teacher_target="d1", alpha=1.0, T=2.0,
        context_s=1.0, epochs=5, batch_size=64, lr=1e-3, num_workers=2, seed=42,
    )
    baseline_checkpoint = Path("experiments") / "mel_trajectory_1s_speech_aligned_contract" / "checkpoint.pt"
    out_dir = Path("experiments") / "distillation_isolation_control"
    train_distill(cfg, baseline_checkpoint, out_dir)


if __name__ == "__main__":
    main()
