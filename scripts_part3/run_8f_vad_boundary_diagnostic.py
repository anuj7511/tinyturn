"""
Phase-2 8f -- VAD-boundary diagnostic on the corrected A0 and B1@1s (feeds Section 8g's
VAD-boundary criterion).

Uses the 206-clip alternative-boundary pilot overlap already carried in tinyturn_splits.parquet
(`last_active_B_energy_alt`, `last_active_C_silero_vad` -- E5's per-clip alternative estimates),
restricted to the val+calib splits (neither model was trained on these rows). Per Phase-2 8f, this
existing 206-clip set is a development/pilot set only -- too small for a hard gate or subgroup
breakdown. This script's job is to say whether real sensitivity is visible at all; if it is, the
brief calls for recomputing alternative boundaries on the full 3,000-clip E5 sample before treating
this as a qualification result.

For each clip, for each model, builds one example under the canonical boundary (`last_active_t`)
and one under each alternative boundary, and compares predicted probability. Reports the fields
Section 8g's frozen VAD-boundary criterion needs:
  - fraction of examples changing by more than 0.20 (criterion: <= 10%)
  - decision-flip rate at each model's own calibrated threshold (criterion: <= 5%)
  - real-audio FCR-at-fixed-recall under each boundary, reported only if n is large enough to be
    meaningful (this pilot set's real-audio subset is small; see printed caveat if skipped)

Usage:
  python scripts_part3/run_8f_vad_boundary_diagnostic.py
"""
import json
import sys
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
from tinyturn.whisper_model import WhisperEndpointModel, WHISPER_MODEL_NAME
from tinyturn.whisper_dataset import extract_whisper_features
from tinyturn.evaluate import fcr_at_fixed_recall
from transformers import WhisperFeatureExtractor

CACHE_DIR = Path("data_cache")
WAV_DIR = CACHE_DIR / "d2_stratified_wavs"
A0_DIR = Path("experiments") / "A0_whisper_tiny_pv2speechend"
B1_DIR = Path("experiments") / "C1_B1_1s_pv2speechend"
N_MELS, N_FFT = 40, 512
FRAME_LENGTH_S, HOP_LENGTH_S = 0.025, 0.010
TRAJECTORY_NAMES = ["rel_energy", "pause_prob", "spectral_tilt", "spectral_flux", "envelope_activity"]
ALT_BOUNDARY_COLS = {"silero_vad": "last_active_C_silero_vad", "energy_alt": "last_active_B_energy_alt"}
MIN_N_FOR_FCR = 20  # below this, an ROC-based FCR-at-fixed-recall is not a meaningful number


def _load_wav(row_id, target_sr=None):
    data, sr = sf.read(WAV_DIR / f"{row_id}.wav")
    y = data if data.ndim == 1 else data.mean(axis=1)
    y = y.astype(np.float32)
    if target_sr is not None and sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y, sr


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


def _a0_prob(model, feature_extractor, y, sr, speech_end_s, context_s):
    ex = build_example(y, sr, speech_end_s, context_s, frame_length_s=FRAME_LENGTH_S,
                        hop_length_s=HOP_LENGTH_S, label=False, row_id="diag")
    input_features, vfm = extract_whisper_features(
        feature_extractor, ex.waveform, ex.valid_sample_mask, sr)
    with torch.no_grad():
        logit = model(torch.from_numpy(input_features).unsqueeze(0),
                       torch.from_numpy(vfm).unsqueeze(0))
    return float(torch.sigmoid(logit).item())


def _run_model(name, prob_fn, df, context_s, threshold, target_sr=None):
    rows = []
    for i, r in df.iterrows():
        y, sr = _load_wav(r["id"], target_sr=target_sr)
        canonical = prob_fn(y, sr, float(r["last_active_t"]), context_s)
        alt_probs = {}
        for alt_name, col in ALT_BOUNDARY_COLS.items():
            if pd.isna(r[col]):
                continue
            alt_probs[alt_name] = prob_fn(y, sr, float(r[col]), context_s)
        row = {"id": r["id"], "real": not bool(r["synthetic"]), "endpoint_bool": bool(r["endpoint_bool"]),
               "prob_canonical": canonical}
        for alt_name, p in alt_probs.items():
            row[f"prob_{alt_name}"] = p
            row[f"abs_diff_{alt_name}"] = abs(p - canonical)
            row[f"flip_{alt_name}"] = (p >= threshold) != (canonical >= threshold)
        rows.append(row)
    out = pd.DataFrame(rows)

    summary = {}
    for alt_name in ALT_BOUNDARY_COLS:
        col = f"abs_diff_{alt_name}"
        if col not in out.columns or out[col].notna().sum() == 0:
            continue
        sub = out[out[col].notna()]
        frac_gt_020 = (sub[col] > 0.20).mean()
        flip_rate = sub[f"flip_{alt_name}"].mean()
        entry = {
            "n": int(len(sub)),
            "frac_change_gt_0.20": round(float(frac_gt_020), 5),
            "decision_flip_rate_at_threshold": round(float(flip_rate), 5),
            "criterion_frac_gt_020_le_0.10": bool(frac_gt_020 <= 0.10),
            "criterion_flip_rate_le_0.05": bool(flip_rate <= 0.05),
        }
        real_sub = sub[sub["real"]]
        if len(real_sub) >= MIN_N_FOR_FCR and real_sub["endpoint_bool"].nunique() > 1:
            fcr_canon = fcr_at_fixed_recall(real_sub["endpoint_bool"].values, real_sub["prob_canonical"].values)
            fcr_alt = fcr_at_fixed_recall(real_sub["endpoint_bool"].values, real_sub[f"prob_{alt_name}"].values)
            entry["real_fcr_at_fixed_recall_canonical"] = round(fcr_canon, 5)
            entry["real_fcr_at_fixed_recall_alt"] = round(fcr_alt, 5)
            entry["real_fcr_degradation_pp"] = round((fcr_alt - fcr_canon) * 100, 3)
            entry["criterion_fcr_degradation_le_2pp"] = bool((fcr_alt - fcr_canon) * 100 <= 2.0)
        else:
            entry["real_fcr_note"] = (f"skipped: only {len(real_sub)} real-audio rows in this pilot "
                                       f"overlap (need >= {MIN_N_FOR_FCR} with both classes) -- not "
                                       f"a meaningful ROC-based estimate at this n")
        summary[alt_name] = entry
    return out, summary


def main():
    for d in (A0_DIR, B1_DIR):
        if not (d / "checkpoint.pt").exists():
            print(f"ERROR: {d / 'checkpoint.pt'} not found -- run the 8d retrain scripts first.")
            sys.exit(1)

    splits = pd.read_parquet(CACHE_DIR / "tinyturn_splits.parquet")
    # splits already carries dataset/language/synthetic/endpoint_bool -- only last_active_t is new
    # here, so only merge that in (merging the full column set would create _x/_y suffixed dupes).
    sf_feat = pd.read_parquet(CACHE_DIR / "d2_stratified_signal_features.parquet")[
        ["id", "last_active_t"]]
    df = splits[splits["split"].isin(["val", "calib"])].merge(sf_feat, on="id", how="left")
    df = df[df["last_active_C_silero_vad"].notna() | df["last_active_B_energy_alt"].notna()]
    df = df[df["last_active_t"].notna()].reset_index(drop=True)
    print(f"8f: {len(df)} val+calib clips with an alternative-boundary estimate available "
          f"(pilot overlap -- dev/pilot only, per Phase-2 8f)", flush=True)
    if len(df) == 0:
        print("No overlap rows found -- nothing to test.")
        sys.exit(1)

    a0_cfg = json.load(open(A0_DIR / "config.json"))
    a0_metrics = json.load(open(A0_DIR / "metrics.json"))
    a0_model = WhisperEndpointModel(model_name=a0_cfg.get("model_name", WHISPER_MODEL_NAME))
    a0_model.load_state_dict(torch.load(A0_DIR / "checkpoint.pt", map_location="cpu"))
    a0_model.eval()
    a0_fe = WhisperFeatureExtractor.from_pretrained(a0_cfg.get("model_name", WHISPER_MODEL_NAME))

    b1_cfg = json.load(open(B1_DIR / "config.json"))
    b1_metrics = json.load(open(B1_DIR / "metrics.json"))
    b1_model = TinyTurnModel(n_mels=N_MELS, trajectory_dim=len(TRAJECTORY_NAMES),
                             mel_channels=b1_cfg["mel_channels"], traj_channels=b1_cfg["traj_channels"])
    b1_model.load_state_dict(torch.load(B1_DIR / "checkpoint.pt", map_location="cpu"))
    b1_model.eval()

    print("running A0...", flush=True)
    a0_out, a0_summary = _run_model(
        "A0", lambda y, sr, se, cs: _a0_prob(a0_model, a0_fe, y, sr, se, cs),
        df, float(a0_cfg["context_s"]), float(a0_metrics["threshold"]))

    print("running B1@1s...", flush=True)
    b1_out, b1_summary = _run_model(
        "B1", lambda y, sr, se, cs: _b1_prob(b1_model, y, sr, se, cs),
        df, float(b1_cfg["context_s"]), float(b1_metrics["threshold"]), target_sr=16000)

    result = {"n_pilot_clips": len(df), "A0": a0_summary, "B1_1s": b1_summary}
    out_path = Path("experiments") / "8f_vad_boundary_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump({"summary": result,
                   "per_clip_A0": a0_out.to_dict(orient="records"),
                   "per_clip_B1_1s": b1_out.to_dict(orient="records")}, f, indent=2, default=str)

    print(json.dumps(result, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
