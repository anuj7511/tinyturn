"""
8g remediation (brief Section 2, "If converged A0 fails the VAD gate, follow this remediation
once"): retrain a boundary-robust A0@4s using train-time boundary augmentation.

Steps 1-3 of the brief's 5-step remediation:
  1. A/B/C boundaries for D2 training audio -- precomputed separately by
     scripts/precompute_train_boundary_augmentation.py into
     data_cache/d2_train_boundary_augmentation.parquet (this script requires that file to exist and
     cover the full train split; it does not compute boundaries itself).
  2. Retrain one boundary-robust A0 using randomly sampled plausible boundaries as augmentation,
     label-independent -- tinyturn.whisper_dataset.WhisperTurnDataset(augment_boundaries=True).
  3. Calibrate only on the canonical boundary -- val_ds/calib_ds below never set
     augment_boundaries, so calibration and model-selection AUC stay on the canonical boundary
     regardless of what train saw.
(Steps 4-5 -- rerun 8g exactly once on this checkpoint, stop if it fails or comes back inconclusive
rather than iterating -- are a separate manual decision after this script finishes; see the printed
next-steps at the end.)

GATED on 8h-A0 resolving first (open_dependency in experiments/A0_whisper_tiny_pv2speechend/
8g_qualification_v2.json): retraining against an ambiguously-converged base protocol would make "did
augmentation fix boundary sensitivity" and "would fixing convergence alone have done it" impossible
to tell apart. This script checks for experiments/8h_a0_step1_significance.json (produced by
run_8h_a0_step1_significance.py) before running, and refuses to proceed without it unless --force is
passed explicitly.

Epoch/schedule protocol is deliberately NOT hard-coded to a single "correct" value -- 8h-A0's own
open question is exactly what protocol A0 should train under (see EPOCHS/EARLY_STOP_PATIENCE/
LR_SCHEDULE/BATCH_SIZE below). Defaults mirror the Kaggle early-stopped protocol from the 8h-A0
longrun (plateau schedule, patience 3) as the safer of the two known options, but override these
once 8h-A0 actually resolves rather than trusting this default blindly.

Usage:
  python scripts/retrain_a0_boundary_robust.py [--force]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.train_whisper import WhisperExperimentConfig, train_whisper_experiment
from tinyturn.whisper_dataset import TRAIN_BOUNDARY_AUGMENTATION_PATH

OUT_DIR = Path("experiments") / "A0_whisper_tiny_boundary_robust"
STEP1_SIGNIFICANCE_PATH = Path("experiments") / "8h_a0_step1_significance.json"
FORCE = "--force" in sys.argv

CONTEXT_S = 4.0             # brief's "Model decision": A0@4s, not @2s, is the teacher candidate.
EPOCHS = 10
EARLY_STOP_PATIENCE = 3
LR_SCHEDULE = "plateau"
BATCH_SIZE = 16             # matches the original run's batch_size (8h-A0 Step 2 concern); raise if
                            # running on a GPU with headroom, as the Kaggle longrun cells do (32).
NUM_WORKERS = 0             # >0 needs tinyturn.whisper_dataset.worker_init_fn (already wired in
                            # train_whisper_experiment) so augmentation draws aren't correlated
                            # across forked workers -- safe either way, just slower at 0.
SEED = 42


def main():
    if not TRAIN_BOUNDARY_AUGMENTATION_PATH.exists():
        print(f"ERROR: {TRAIN_BOUNDARY_AUGMENTATION_PATH} not found -- run "
              f"scripts/precompute_train_boundary_augmentation.py first (full train split, "
              f"no --limit).")
        sys.exit(1)

    if not STEP1_SIGNIFICANCE_PATH.exists() and not FORCE:
        print(
            f"ERROR: {STEP1_SIGNIFICANCE_PATH} not found.\n"
            "This remediation retrain is explicitly gated on 8h-A0 resolving first (brief Section "
            "2 / 8g_qualification_v2.json's own open_dependency note): retraining against an "
            "ambiguously-converged base protocol would make it impossible to tell whether "
            "augmentation fixed boundary sensitivity, or fixing convergence alone would have.\n"
            "Run run_8h_a0_step1_significance.py first (needs the Kaggle-retrained A0@4s checkpoint "
            "downloaded locally), let 8h-A0 resolve which training protocol is correct, then update "
            "EPOCHS/EARLY_STOP_PATIENCE/LR_SCHEDULE/BATCH_SIZE at the top of this script to match.\n"
            "Pass --force to proceed anyway with this script's current (unvalidated) defaults."
        )
        sys.exit(1)

    if FORCE and not STEP1_SIGNIFICANCE_PATH.exists():
        print("--force: proceeding without 8h-A0 resolution. Results here may need to be redone "
              "once the correct training protocol is known.\n")

    cfg = WhisperExperimentConfig(
        exp_id="A0_boundary_robust", context_s=CONTEXT_S, epochs=EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE, lr_schedule=LR_SCHEDULE,
        batch_size=BATCH_SIZE, lr=1e-5, num_workers=NUM_WORKERS, seed=SEED,
        augment_boundaries=True,
    )
    print(f"training boundary-robust A0 -> {OUT_DIR}\nconfig: {cfg}\n")
    report = train_whisper_experiment(cfg, OUT_DIR)
    print(f"\ndone. best_epoch={report.get('best_epoch')} best_val_auc={report.get('best_val_auc')} "
          f"overall_auc={report['overall']['auc']} real_auc={report['real_all']['auc']}")

    print(
        "\nNext (brief steps 4-5, run manually):\n"
        f"  python scripts/padding_counterfactual_a0.py {OUT_DIR}\n"
        f"  python scripts/vad_boundary_diagnostic_full_val.py {OUT_DIR}\n"
        f"  python scripts/qualify_teacher_a0_ci_gated.py {OUT_DIR}\n"
        "Run 8g exactly once on this checkpoint. If it still fails or comes back inconclusive, "
        "stop -- per the brief, that's a real result (A0 isn't a viable teacher without more "
        "substantial rework), not a reason to iterate against the same held-out set."
    )


if __name__ == "__main__":
    main()
