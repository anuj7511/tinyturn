"""
8g remediation, Step 1 (brief Section 2, "Boundary-robust A0@4s retrain"): precompute
canonical/alt-threshold/Silero boundary estimates for D2 TRAIN-split audio.

Val already has alt/silero boundaries from 8f's full rerun (8f_vad_boundary_diagnostic_v2.json);
calib never needs them at all -- the brief is explicit that calibration stays on the canonical
boundary only, so drift in the augmented-training boundary set can't leak into where the deployed
threshold sits. Train is the only split that needs this precomputed.

Reuses the exact alt_threshold_speech_end / silero_speech_end functions from
run_8f_vad_boundary_diagnostic_v2.py unmodified -- these must stay byte-identical to what 8f/8g
already gate A0's val predictions against, or "robust to the same boundary variants 8g qualifies
against" wouldn't hold.

"Precompute all boundary estimates once, rather than running Silero inside the training data
loader" (brief) -- this script is that one-time precompute; tinyturn.whisper_dataset reads its
output at training time and never calls Silero live.

Usage:
  python scripts/precompute_train_boundary_augmentation.py [--limit N]
"""
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_8f_vad_boundary_diagnostic_v2 import (
    alt_threshold_speech_end, silero_speech_end, _load_wav,
)

CACHE_DIR = Path("data_cache")
OUT_PATH = CACHE_DIR / "d2_train_boundary_augmentation.parquet"


def main():
    limit = None
    if len(sys.argv) > 1 and sys.argv[1] == "--limit":
        limit = int(sys.argv[2])

    splits = pd.read_parquet(CACHE_DIR / "tinyturn_splits.parquet")
    sf_feat = pd.read_parquet(CACHE_DIR / "d2_stratified_signal_features.parquet")[["id", "last_active_t"]]
    df = splits[splits["split"] == "train"].merge(sf_feat, on="id", how="left")
    df = df[df["last_active_t"].notna()].reset_index(drop=True)
    if limit is not None:
        df = df.iloc[:limit].reset_index(drop=True)
    print(f"precomputing A/B/C boundaries for {len(df)} train clips", flush=True)

    print("loading Silero VAD (offline, from torch.hub cache)...", flush=True)
    silero_model, silero_utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', source='github',
                                                  trust_repo=True, onnx=False)
    get_speech_timestamps = silero_utils[0]

    alt_boundaries, silero_boundaries = [], []
    t0 = time.time()
    for i, r in df.iterrows():
        y, sr = _load_wav(r["id"])
        alt_boundaries.append(alt_threshold_speech_end(y, sr))
        y16k, _ = _load_wav(r["id"], target_sr=16000)
        silero_boundaries.append(silero_speech_end(y16k, get_speech_timestamps, silero_model))
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(df) - (i + 1)) / rate
            print(f"  {i + 1}/{len(df)} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

    out = df[["id", "last_active_t"]].rename(columns={"last_active_t": "canonical_boundary_s"})
    out["alt_threshold_boundary_s"] = alt_boundaries
    out["silero_boundary_s"] = silero_boundaries
    out.to_parquet(OUT_PATH, index=False)
    print(f"done in {time.time() - t0:.0f}s -- saved {len(out)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
