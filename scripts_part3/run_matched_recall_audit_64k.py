"""
Step 10 planning, 64k confirmation -- matched-recall (calib->val) hold-FCR audit for the 64k models,
same method as run_matched_recall_audit.py (threshold selected on CALIB's recall curve, evaluated on
VAL only -- never selected and evaluated on the same split). Kept as a separate script/output rather
than folding into run_matched_recall_audit.py's CHECKPOINTS dict so re-running it doesn't re-audit
the 12 already-recorded 16k/32k-scale checkpoints.

Per the user's explicit instruction: do not use each 64k model's own-threshold hold FCR
(4.43% vs 69.25% as reported straight from metrics.json) to decide anything -- this audit is the
one that actually licenses a comparison, since both models are evaluated at the same matched
complete-turn recall on val, off a threshold chosen on calib.

Seeds 43/44 are added to CHECKPOINTS here (as they finish training) so the final report is a
3-seed mean+/-std, matching the existing 16k/32k precedent.

Usage:
  python scripts_part3/run_matched_recall_audit_64k.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.dataset import TinyTurnDataset, collate
from tinyturn.pause_events import PauseEventDataset
from scripts_part3.run_matched_recall_audit import audit_one, CACHE_DIR, CONTEXT_S

OUT_PATH = Path("experiments") / "matched_recall_audit_64k.json"

CHECKPOINTS = {
    "B1_64k_baseline": {
        42: "experiments/B1_1s_64k_baseline",
        43: "experiments/B1_1s_64k_baseline_seed43",
        44: "experiments/B1_1s_64k_baseline_seed44",
    },
    "B1_64k_lambda0.5_5050": {
        42: "experiments/B1_1s_64k_lambda0.5_5050",
        43: "experiments/B1_1s_64k_lambda0.5_5050_seed43",
        44: "experiments/B1_1s_64k_lambda0.5_5050_seed44",
    },
}


def main():
    ds_kwargs = dict(context_s=CONTEXT_S, include_trajectory=True)
    calib_ds = TinyTurnDataset(split="calib", **ds_kwargs)
    val_ds = TinyTurnDataset(split="val", **ds_kwargs)
    val_pause_ds = PauseEventDataset(split="val", context_s=CONTEXT_S, include_trajectory=True)
    calib_loader = DataLoader(calib_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
    pause_loader = DataLoader(val_pause_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
    trans_df = pd.read_parquet(CACHE_DIR / "d2_stratified_transcripts.parquet")[["id", "text", "n_words"]]
    device = torch.device("cpu")

    results = {}
    for arm, seeds in CHECKPOINTS.items():
        results[arm] = {}
        for seed, path in seeds.items():
            d = Path(path)
            if not (d / "checkpoint.pt").exists():
                print(f"SKIP {arm} seed={seed}: {d} missing checkpoint.pt")
                continue
            print(f"auditing {arm} seed={seed} ({d})...", flush=True)
            results[arm][seed] = audit_one(d, calib_loader, val_loader, pause_loader, val_ds, trans_df, device)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved {OUT_PATH}")

    print("\n=== 64k matched-recall (calib) -> hold-FCR (val) summary, mean +/- std across available seeds ===")
    for arm, seeds in results.items():
        if not seeds:
            continue
        for target_key in ["recall_90", "recall_95"]:
            vals_all = [s["matched"][target_key]["hold_fcr_all"] for s in seeds.values()]
            vals_real = [s["matched"][target_key]["hold_fcr_real"] for s in seeds.values()]
            vals_syn = [s["matched"][target_key]["hold_fcr_synthetic"] for s in seeds.values()]
            actual_recall = [s["matched"][target_key]["actual_recall_complete_val"] for s in seeds.values()]
            short_recall = [s["matched"][target_key]["short_complete_recall"]["recall"] for s in seeds.values()
                             if s["matched"][target_key]["short_complete_recall"]["recall"] is not None]
            print(f"{arm} @ {target_key} (n_seeds={len(seeds)}): actual_val_recall={np.mean(actual_recall):.4f} "
                  f"hold_fcr_all={np.mean(vals_all):.4f}+/-{np.std(vals_all):.4f} "
                  f"hold_fcr_real={np.mean(vals_real):.4f}+/-{np.std(vals_real):.4f} "
                  f"hold_fcr_synth={np.mean(vals_syn):.4f}+/-{np.std(vals_syn):.4f} "
                  f"short_complete_recall={np.mean(short_recall) if short_recall else float('nan'):.4f}")


if __name__ == "__main__":
    main()
