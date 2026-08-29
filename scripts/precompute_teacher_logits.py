"""
Step 10 planning, item 2 -- precompute A0_boundary_robust teacher logits for the D2 train split.

Per the plan: "Use A0_boundary_robust as an offline teacher despite its deployment-robustness
failure." Two distillation targets, both derived from the same three per-clip raw logits (no
sigmoid -- distillation operates on logits so temperature scaling is well-defined):
  D1: canonical-boundary A0 logit only.
  D2: mean of the canonical / alt-threshold / Silero logits (reuses the same three boundaries the
      8g-remediation retrain augmented on, from data_cache/d2_train_boundary_augmentation.parquet --
      no new boundary detection needed here, only new A0 forward passes).

No training here -- inference only, run once, cached to parquet so both D1 and D2 distillation runs
(and any rerun of either) reuse it without recomputing.

Usage:
  python scripts/precompute_teacher_logits.py [--limit N]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.preprocess import build_example
from tinyturn.whisper_model import WhisperEndpointModel, WHISPER_MODEL_NAME
from tinyturn.whisper_dataset import extract_whisper_features
from transformers import WhisperFeatureExtractor
import json

CACHE_DIR = Path("data_cache")
WAV_DIR = CACHE_DIR / "d2_stratified_wavs"
A0_DIR = Path("experiments") / "A0_boundary_robust"
BOUNDARY_AUG_PATH = CACHE_DIR / "d2_train_boundary_augmentation.parquet"
OUT_PATH = CACHE_DIR / "teacher_logits_a0_boundary_robust_train.parquet"
FRAME_LENGTH_S, HOP_LENGTH_S = 0.025, 0.010


def _load_wav(row_id):
    data, sr = sf.read(WAV_DIR / f"{row_id}.wav")
    y = data if data.ndim == 1 else data.mean(axis=1)
    return y.astype(np.float32), sr


def _a0_logit(model, feature_extractor, y, sr, speech_end_s, context_s):
    ex = build_example(y, sr, speech_end_s, context_s, frame_length_s=FRAME_LENGTH_S,
                        hop_length_s=HOP_LENGTH_S, label=False, row_id="teacher")
    input_features, vfm = extract_whisper_features(feature_extractor, ex.waveform, ex.valid_sample_mask, sr)
    with torch.no_grad():
        logit = model(torch.from_numpy(input_features).unsqueeze(0), torch.from_numpy(vfm).unsqueeze(0))
    return float(logit.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not (A0_DIR / "checkpoint.pt").exists():
        print(f"ERROR: {A0_DIR / 'checkpoint.pt'} not found.")
        sys.exit(1)

    splits = pd.read_parquet(CACHE_DIR / "tinyturn_splits.parquet")
    sf_feat = pd.read_parquet(CACHE_DIR / "d2_stratified_signal_features.parquet")[["id", "last_active_t"]]
    aug = pd.read_parquet(BOUNDARY_AUG_PATH)

    df = splits[splits["split"] == "train"][["id"]].merge(sf_feat, on="id", how="left")
    df = df.merge(aug[["id", "alt_threshold_boundary_s", "silero_boundary_s"]], on="id", how="left")
    df = df[df["last_active_t"].notna()].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    print(f"teacher-logit precompute: {len(df)} train clips", flush=True)

    a0_cfg = json.load(open(A0_DIR / "config.json"))
    context_s = float(a0_cfg["context_s"])
    model_name = a0_cfg.get("model_name", WHISPER_MODEL_NAME)
    model = WhisperEndpointModel(model_name=model_name)
    model.load_state_dict(torch.load(A0_DIR / "checkpoint.pt", map_location="cpu"))
    model.eval()
    fe = WhisperFeatureExtractor.from_pretrained(model_name)

    rows = []
    t0 = time.time()
    for i, r in df.iterrows():
        y, sr = _load_wav(r["id"])
        logit_canonical = _a0_logit(model, fe, y, sr, float(r["last_active_t"]), context_s)

        others = []
        for col in ("alt_threshold_boundary_s", "silero_boundary_s"):
            v = r[col]
            if pd.notna(v):
                others.append(_a0_logit(model, fe, y, sr, float(v), context_s))

        teacher_logit_d1 = logit_canonical
        teacher_logit_d2 = float(np.mean([logit_canonical] + others))
        rows.append({"id": r["id"], "logit_canonical": logit_canonical,
                     "n_alt_boundaries_used": len(others),
                     "teacher_logit_d1": teacher_logit_d1, "teacher_logit_d2": teacher_logit_d2})
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta_min = (len(df) - (i + 1)) / rate / 60
            print(f"  {i + 1}/{len(df)} ({elapsed:.0f}s, ETA {eta_min:.1f}min)", flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(OUT_PATH, index=False)
    print(f"done in {time.time() - t0:.0f}s -- saved {OUT_PATH} ({len(out)} rows, "
          f"{out['n_alt_boundaries_used'].eq(2).sum()} with both alt boundaries)")
    print(out[["teacher_logit_d1", "teacher_logit_d2"]].describe())


if __name__ == "__main__":
    main()
