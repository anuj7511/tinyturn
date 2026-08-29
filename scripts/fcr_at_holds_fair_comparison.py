"""
Correction: train_p1.py's baseline-vs-P1 FCR-at-holds comparison evaluated the baseline model at
P1's calibrated threshold (0.5952) rather than the baseline's OWN calibrated threshold (0.7541,
from experiments/C1_B1_1s/metrics.json) -- an apples-to-oranges comparison, since a lower threshold
mechanically produces more "complete" predictions regardless of what the model learned. This
re-evaluates both models on the val pause-event set, each at its OWN calibrated threshold, i.e.
each model exactly as it would actually be deployed.

Kept import/call-signature-compatible with Phase-2 8b's dropped `postroll_s` parameter, but the
checkpoints it loads were trained under the old N+200ms-post-roll contract and are stale per
Phase-2 8d -- its output is historical (pre-8b), not a current result.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from tinyturn.pause_events import PauseEventDataset
from tinyturn.dataset import collate
from tinyturn.models import TinyTurnModel
from tinyturn.train_p1 import evaluate_fcr_at_holds
from tinyturn.train import TRAJECTORY_NAMES


def main():
    device = torch.device("cpu")
    val_pause_ds = PauseEventDataset(split="val", context_s=1.0, include_trajectory=True)
    val_pause_loader = DataLoader(val_pause_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)

    baseline_threshold = json.load(open("experiments/C1_B1_1s/metrics.json"))["threshold"]
    p1_threshold = json.load(open("experiments/P1_pause_events/metrics.json"))["threshold"]

    baseline = TinyTurnModel(n_mels=40, trajectory_dim=len(TRAJECTORY_NAMES))
    baseline.load_state_dict(torch.load("experiments/C1_B1_1s/checkpoint.pt", map_location=device))
    p1 = TinyTurnModel(n_mels=40, trajectory_dim=len(TRAJECTORY_NAMES))
    p1.load_state_dict(torch.load("experiments/P1_pause_events/checkpoint.pt", map_location=device))

    baseline_result = evaluate_fcr_at_holds(baseline, val_pause_loader, device, baseline_threshold)
    p1_result = evaluate_fcr_at_holds(p1, val_pause_loader, device, p1_threshold)

    print("baseline (B1@1.0s, threshold=%.4f):" % baseline_threshold, json.dumps(baseline_result, indent=2))
    print("P1 (threshold=%.4f):" % p1_threshold, json.dumps(p1_result, indent=2))

    out = {"baseline_at_own_threshold": baseline_result, "p1_at_own_threshold": p1_result}
    with open("experiments/P1_pause_events/fcr_at_holds_fair_compare.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved experiments/P1_pause_events/fcr_at_holds_fair_compare.json")


if __name__ == "__main__":
    main()
