"""
Step 10 planning -- ground-truth-conditioned VAD-boundary diagnostic for the 64k B1 checkpoints
(baseline + lambda=0.5 50:50, seeds 42/43/44), never run at this data scale/objective combination
before. Combines two existing pieces rather than re-deriving either:

  - Boundary computation (canonical / alt-threshold / Silero) on the full 1,600-clip val split,
    exactly as in `vad_boundary_diagnostic_full_val.py` (same alt-threshold formula, same Silero
    model). Cached to `experiments/8f_val_boundaries_cache.parquet` on first run so the expensive
    Silero + alt-threshold pass isn't repeated for each of up to 6 checkpoints, or on every re-run
    as seed43/44 checkpoints finish training.

  - Ground-truth-conditioned introduced-error metrics from `ground_truth_conditioned_metric_audit.py`
    (`introduced_false_completion` / `introduced_delay`, conditioned on truth AND canonical
    correctness) plus each variant's standalone FCR-at-recall95, instead of 8f-v2's plain
    canonical-conditioned flip-rate gate -- Section 1 of this project's own audit already
    established the flip-rate gate overcounts "introduced" errors with corrections, so this uses
    the corrected method directly rather than reproducing the superseded one.

Threshold discipline: each checkpoint's own calib-calibrated threshold (metrics.json["threshold"]),
same convention as 8f-v2 and ground_truth_conditioned_metric_audit.py -- never recomputed here.

Checkpoints that haven't finished training yet (seed43/44, while the 3-seed 64k confirmation runs)
are skipped with a printed note; re-run this script after they land to fill in the rest.

Usage:
  python scripts/vad_boundary_diagnostic_b1_64k.py
  python scripts/vad_boundary_diagnostic_b1_64k.py --refresh-boundaries  # force recompute
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.preprocess import build_example
from tinyturn.features import compute_trajectory_channels
from tinyturn.models import TinyTurnModel
from scripts.ground_truth_conditioned_metric_audit import (
    introduced_errors, fcr_triplet, rate_entry,
)

CACHE_DIR = Path("data_cache")
WAV_DIR = CACHE_DIR / "d2_stratified_wavs"
BOUNDARY_CACHE_PATH = Path("experiments") / "8f_val_boundaries_cache.parquet"
OUT_PATH = Path("experiments") / "vad_boundary_diagnostic_data_scale_64k.json"

N_MELS, N_FFT = 40, 512
FRAME_LENGTH_S, HOP_LENGTH_S = 0.025, 0.010
TRAJECTORY_NAMES = ["rel_energy", "pause_prob", "spectral_tilt", "spectral_flux", "envelope_activity"]

# Same constants as vad_boundary_diagnostic_full_val.py -- deliberately a different alt-threshold
# formula from tinyturn.boundary's canonical one, not a reproduction of it.
ALT_FRAME_LENGTH_S, ALT_HOP_LENGTH_S = 0.020, 0.010
ALT_NOISE_FLOOR_OFFSET_DB = -15.0
ALT_PEAK_OFFSET_DB = -30.0

CHECKPOINTS = {
    "B1_64k_baseline": {
        42: "experiments/data_scale_64k_baseline_seed42",
        43: "experiments/data_scale_64k_baseline_seed43",
        44: "experiments/data_scale_64k_baseline_seed44",
    },
    "B1_64k_lambda0.5_5050": {
        42: "experiments/data_scale_64k_holdloss0.5_5050sampling_seed42",
        43: "experiments/data_scale_64k_holdloss0.5_5050sampling_seed43",
        44: "experiments/data_scale_64k_holdloss0.5_5050sampling_seed44",
    },
}


def _load_wav(row_id, target_sr=None):
    data, sr = sf.read(WAV_DIR / f"{row_id}.wav")
    y = data if data.ndim == 1 else data.mean(axis=1)
    y = y.astype(np.float32)
    if target_sr is not None and sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y, sr


def alt_threshold_speech_end(y: np.ndarray, sr: int) -> float:
    duration = len(y) / sr
    frame_length = int(ALT_FRAME_LENGTH_S * sr)
    hop_length = int(ALT_HOP_LENGTH_S * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    if len(rms) == 0:
        return duration
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    rms_db = 20 * np.log10(np.maximum(rms, 1e-6))
    noise_floor_db = float(np.median(rms_db)) + ALT_NOISE_FLOOR_OFFSET_DB
    peak_db = float(rms_db.max())
    thresh_db = max(noise_floor_db, peak_db + ALT_PEAK_OFFSET_DB)
    active = rms_db > thresh_db
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return duration
    return float(times[idx[-1]])


def silero_speech_end(y16k: np.ndarray, get_speech_timestamps, model) -> float:
    duration = len(y16k) / 16000
    ts = get_speech_timestamps(torch.from_numpy(y16k), model, sampling_rate=16000, return_seconds=True)
    if not ts:
        return duration
    return float(min(ts[-1]["end"], duration))


def _log_mel(y, sr):
    frame_length = int(round(FRAME_LENGTH_S * sr))
    hop_length = int(round(HOP_LENGTH_S * sr))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=hop_length,
                                          win_length=frame_length, n_mels=N_MELS, center=False)
    return np.log(mel + 1e-6).T.astype(np.float32)


def _pad_mask_to(vfm, n_frames):
    vfm = vfm[:n_frames]
    if len(vfm) < n_frames:
        vfm = np.concatenate([vfm, np.zeros(n_frames - len(vfm), dtype=bool)])
    return vfm


def _b1_prob(model, y, sr, speech_end_s, context_s):
    ex = build_example(y, sr, speech_end_s, context_s, frame_length_s=FRAME_LENGTH_S,
                        hop_length_s=HOP_LENGTH_S, label=False, row_id="diag")
    log_mel = _log_mel(ex.waveform, sr)
    n_frames = log_mel.shape[0]
    vfm = _pad_mask_to(ex.valid_frame_mask, n_frames)
    chans = compute_trajectory_channels(ex.waveform, sr, ex.valid_sample_mask, FRAME_LENGTH_S, HOP_LENGTH_S)
    traj = np.stack([chans[n][:n_frames] for n in TRAJECTORY_NAMES], axis=-1).astype(np.float32)
    if traj.shape[0] < n_frames:
        traj = np.pad(traj, ((0, n_frames - traj.shape[0]), (0, 0)))
    with torch.no_grad():
        logit = model(torch.from_numpy(log_mel).unsqueeze(0), torch.from_numpy(vfm).unsqueeze(0),
                       torch.from_numpy(traj).unsqueeze(0))
    return float(torch.sigmoid(logit).item())


def build_or_load_boundaries(refresh: bool) -> pd.DataFrame:
    if BOUNDARY_CACHE_PATH.exists() and not refresh:
        print(f"loading cached val boundaries from {BOUNDARY_CACHE_PATH}", flush=True)
        return pd.read_parquet(BOUNDARY_CACHE_PATH)

    splits = pd.read_parquet(CACHE_DIR / "tinyturn_splits.parquet")
    sf_feat = pd.read_parquet(CACHE_DIR / "d2_stratified_signal_features.parquet")[
        ["id", "last_active_t", "endpoint_bool", "language", "dataset", "synthetic"]]
    df = splits[splits["split"] == "val"][["id"]].merge(sf_feat, on="id", how="left")
    df = df[df["last_active_t"].notna()].reset_index(drop=True)
    print(f"computing fresh alt-threshold + Silero boundaries for {len(df)} val clips...", flush=True)

    print("loading Silero VAD (offline, from torch.hub cache)...", flush=True)
    silero_model, silero_utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', source='github',
                                                  trust_repo=True, onnx=False)
    get_speech_timestamps = silero_utils[0]

    t0 = time.time()
    alt_boundaries, silero_boundaries = [], []
    for i, r in df.iterrows():
        y, sr = _load_wav(r["id"])
        alt_boundaries.append(alt_threshold_speech_end(y, sr))
        y16k, _ = _load_wav(r["id"], target_sr=16000)
        silero_boundaries.append(silero_speech_end(y16k, get_speech_timestamps, silero_model))
        if (i + 1) % 200 == 0:
            print(f"  boundaries {i + 1}/{len(df)} ({time.time() - t0:.0f}s)", flush=True)
    df["alt_threshold_boundary_s"] = alt_boundaries
    df["silero_boundary_s"] = silero_boundaries
    print(f"boundaries done in {time.time() - t0:.0f}s", flush=True)

    df.to_parquet(BOUNDARY_CACHE_PATH)
    print(f"cached boundaries to {BOUNDARY_CACHE_PATH}", flush=True)
    return df


def run_checkpoint(ckpt_dir: Path, boundaries_df: pd.DataFrame) -> dict:
    cfg = json.load(open(ckpt_dir / "config.json"))
    metrics = json.load(open(ckpt_dir / "metrics.json"))
    threshold = float(metrics["threshold"])
    context_s = float(cfg["context_s"])
    model = TinyTurnModel(n_mels=N_MELS, trajectory_dim=len(TRAJECTORY_NAMES),
                           mel_channels=cfg.get("mel_channels", 112), traj_channels=cfg.get("traj_channels", 24))
    model.load_state_dict(torch.load(ckpt_dir / "checkpoint.pt", map_location="cpu"))
    model.eval()

    rows = []
    t0 = time.time()
    for i, r in boundaries_df.iterrows():
        y, sr = _load_wav(r["id"])
        canonical = _b1_prob(model, y, sr, float(r["last_active_t"]), context_s)
        row = {"id": r["id"], "real": not bool(r["synthetic"]), "endpoint_bool": bool(r["endpoint_bool"]),
               "language": r["language"], "dataset": r["dataset"], "prob_canonical": canonical}
        for alt_name, boundary_col in [("alt_threshold", "alt_threshold_boundary_s"),
                                        ("silero_vad", "silero_boundary_s")]:
            se = r[boundary_col]
            if pd.isna(se):
                continue
            row[f"prob_{alt_name}"] = _b1_prob(model, y, sr, float(se), context_s)
        rows.append(row)
        if (i + 1) % 400 == 0:
            print(f"  [{ckpt_dir.name}] {i + 1}/{len(boundaries_df)} ({time.time() - t0:.0f}s)", flush=True)
    out = pd.DataFrame(rows)

    result = {"threshold": threshold, "n_val_clips": int(len(out))}
    real_df = out[out["real"]]
    for alt_name in ["alt_threshold", "silero_vad"]:
        if f"prob_{alt_name}" not in out.columns:
            continue
        result[alt_name] = {
            "overall": introduced_errors(out, alt_name, threshold),
            "real_only": introduced_errors(real_df, alt_name, threshold),
            "fcr_at_recall95": {
                "overall": fcr_triplet(out, alt_name, "overall"),
                "real_only": fcr_triplet(real_df, alt_name, "real_only"),
            },
        }
    return result, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-boundaries", action="store_true")
    args = ap.parse_args()

    boundaries_df = build_or_load_boundaries(args.refresh_boundaries)

    all_results = {}
    for arm, seeds in CHECKPOINTS.items():
        all_results[arm] = {}
        for seed, path in seeds.items():
            d = Path(path)
            if not (d / "checkpoint.pt").exists():
                print(f"SKIP {arm} seed={seed}: {d} missing checkpoint.pt (not trained yet)")
                continue
            print(f"running {arm} seed={seed} ({d})...", flush=True)
            result, per_clip = run_checkpoint(d, boundaries_df)
            result["per_clip"] = per_clip.to_dict(orient="records")
            all_results[arm][seed] = result

    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nsaved {OUT_PATH}")

    print("\n=== ground-truth-conditioned safety summary (64k B1 checkpoints) ===")
    for arm, seeds in all_results.items():
        for seed, r in seeds.items():
            for alt_name in ["alt_threshold", "silero_vad"]:
                if alt_name not in r:
                    continue
                ie = r[alt_name]["real_only"]["introduced_false_completion"]
                idl = r[alt_name]["real_only"]["introduced_delay"]
                print(f"{arm} seed={seed} / {alt_name} (real_only): "
                      f"introduced_false_completion={ie['rate']} (n={ie['n']}, ci={ie['ci_95']})  "
                      f"introduced_delay={idl['rate']} (n={idl['n']}, ci={idl['ci_95']})")


if __name__ == "__main__":
    main()
