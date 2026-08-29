"""
Step 10 planning, item 2 -- run one of the two fixed distillation ablation configs (D1 or D2).
See tinyturn/train_distill.py for the loss/recipe definition.

Usage:
  python scripts_part3/run_step10_distill.py d1
  python scripts_part3/run_step10_distill.py d2
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.train_distill import DistillConfig, train_distill


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("d1", "d2"):
        print("usage: python scripts_part3/run_step10_distill.py [d1|d2]")
        sys.exit(1)
    target = sys.argv[1]
    cfg = DistillConfig(
        exp_id=f"B1_1s_distill_{target}", teacher_target=target,
        context_s=1.0, epochs=5, batch_size=64, lr=1e-3, num_workers=2, seed=42,
        T=2.0, alpha=0.5,
    )
    baseline_checkpoint = Path("experiments") / "C1_B1_1s_pv2speechend" / "checkpoint.pt"
    out_dir = Path("experiments") / f"B1_1s_distill_{target}"
    train_distill(cfg, baseline_checkpoint, out_dir)


if __name__ == "__main__":
    main()
