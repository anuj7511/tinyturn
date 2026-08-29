"""
Run one of the two fixed distillation ablation configs: canonical-boundary teacher logits (d1)
or mean-of-3-boundary teacher logits (d2). See tinyturn/train_distill.py for the loss/recipe
definition.

Usage:
  python scripts/train_distillation_ablation.py d1
  python scripts/train_distillation_ablation.py d2
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.train_distill import DistillConfig, train_distill

EXP_NAMES = {
    "d1": "distillation_canonical_boundary_teacher",
    "d2": "distillation_mean3boundary_teacher",
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("d1", "d2"):
        print("usage: python scripts/train_distillation_ablation.py [d1|d2]")
        sys.exit(1)
    target = sys.argv[1]
    exp_name = EXP_NAMES[target]
    cfg = DistillConfig(
        exp_id=exp_name, teacher_target=target,
        context_s=1.0, epochs=5, batch_size=64, lr=1e-3, num_workers=2, seed=42,
        T=2.0, alpha=0.5,
    )
    baseline_checkpoint = Path("experiments") / "mel_trajectory_1s_speech_aligned_contract" / "checkpoint.pt"
    out_dir = Path("experiments") / exp_name
    train_distill(cfg, baseline_checkpoint, out_dir)


if __name__ == "__main__":
    main()
