"""One-off Phase-2 8d retrain: A0 (Whisper-Tiny), full fine-tune, under the corrected speech-aligned
input contract (Phase-2 8b: N seconds ending at detected speech end, no baked-in post-roll; 8c:
canonicalized padding + real encoder attention masking).

Writes to a NEW experiment directory (`A0_whisper_tiny_pv2speechend`) rather than overwriting
`experiments/A0_whisper_tiny/` -- per Phase-2 8d, "the previous results for both remain historical
baselines only," and per the implementation rules, prior experiment directories are never
overwritten.

Same hyperparameters as the original A0_whisper_tiny run (context_s=4.0, 2 epochs, batch_size=8,
lr=1e-5, num_workers=0) so any AUC delta reflects the contract change, not a hyperparameter change.
CPU-only, 16GB RAM, one experiment at a time -- run this to completion before starting the B1@1s
retrain (scripts_part3/run_8d_retrain_b1_1s.py).

Usage:
  python scripts_part3/run_8d_retrain_a0.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.train_whisper import WhisperExperimentConfig, train_whisper_experiment


def main():
    cfg = WhisperExperimentConfig(
        exp_id="A0_pv2speechend", context_s=4.0, epochs=2, batch_size=8, lr=1e-5, num_workers=0,
    )
    out_dir = Path("experiments") / "A0_whisper_tiny_pv2speechend"
    train_whisper_experiment(cfg, out_dir)


if __name__ == "__main__":
    main()
