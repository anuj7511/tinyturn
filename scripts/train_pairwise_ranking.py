"""
Step 10 planning, item 3 -- run the within-utterance pairwise-ranking experiment.
See tinyturn/train_ranking.py for the loss/pairing definition.

Usage:
  python scripts/train_pairwise_ranking.py                      # seed 42, fixed 5-epoch protocol
  python scripts/train_pairwise_ranking.py --seed 43 --plateau  # 3-seed confirmation (step 5),
                                                                   # matching the 8h-validated
                                                                   # protocol the other finalists use
                                                                   # at seeds 43/44.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.train_ranking import RankingConfig, train_ranking

SEED_BASELINE_CHECKPOINTS = {
    42: Path("step9_results_updated") / "baseline_kaggle" / "checkpoint.pt",
    43: Path("step9_results_updated") / "baseline_no_pause_events_seed43" / "checkpoint.pt",
    44: Path("step9_results_updated") / "baseline_no_pause_events_seed44" / "checkpoint.pt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plateau", action="store_true",
                     help="epochs=40/early_stop_patience=6/lr_schedule=plateau (8h-validated "
                          "protocol, matching the lambda=0.5 finalists at seeds 43/44) instead of "
                          "the original fixed 5-epoch protocol.")
    args = ap.parse_args()

    exp_id = f"B1_1s_ranking_seed{args.seed}" + ("_plateau" if args.plateau else "")
    cfg = RankingConfig(
        exp_id=exp_id, context_s=1.0,
        epochs=40 if args.plateau else 5,
        early_stop_patience=6 if args.plateau else None,
        lr_schedule="plateau" if args.plateau else None,
        batch_size=64, lr=1e-3, num_workers=0, seed=args.seed, margin=0.2, rank_weight=0.1,
    )
    baseline_checkpoint = (SEED_BASELINE_CHECKPOINTS[args.seed] if args.plateau
                            else Path("experiments") / "C1_B1_1s_pv2speechend" / "checkpoint.pt")
    out_dir = Path("experiments") / exp_id
    train_ranking(cfg, baseline_checkpoint, out_dir)


if __name__ == "__main__":
    main()
