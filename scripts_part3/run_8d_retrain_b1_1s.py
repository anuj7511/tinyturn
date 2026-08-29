"""One-off Phase-2 8d retrain: B1@1.0s (mel + trajectory fusion, tiny finalist, its own chosen
production context per Step 6) under the corrected speech-aligned input contract (Phase-2 8b/8c).

Writes to a NEW experiment directory (`C1_B1_1s_pv2speechend`) rather than overwriting
`experiments/C1_B1_1s/` -- the pre-8b run stays as the historical baseline (Phase-2 8d).

Same hyperparameters as the original C1_B1_1s run (context_s=1.0, 5 epochs, batch_size=64,
num_workers=2) so any AUC delta reflects the contract change, not a hyperparameter change.
Run only after run_8d_retrain_a0.py has finished (CPU-only, 16GB RAM, one experiment at a time).

Usage:
  python scripts_part3/run_8d_retrain_b1_1s.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.train import ExperimentConfig, train_experiment


def main():
    cfg = ExperimentConfig(
        exp_id="C1_B1_1s_pv2speechend", context_s=1.0, use_trajectory=True, use_f0=False,
        epochs=5, batch_size=64, num_workers=2,
    )
    out_dir = Path("experiments") / "C1_B1_1s_pv2speechend"
    train_experiment(cfg, out_dir)


if __name__ == "__main__":
    main()
