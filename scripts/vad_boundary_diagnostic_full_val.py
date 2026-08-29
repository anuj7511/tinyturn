"""
Phase-3 8f -- VAD-boundary diagnostic, recomputed fresh on full D2 val-split audio.

8f-0 (this revision's first question, per the brief): confirmed directly that
`data_cache/d2_stratified_wavs/` holds full, uncropped decoded audio, not pre-windowed training
examples -- all 15,998 files present, durations vary continuously (3.8s-11s+ observed on a sample),
not clustered at any single context_s (1s/2s/4s/8s) the way a pre-cropped cache would be. That
means Silero and an alt-threshold detector can be run fresh on the full 1,600-clip val split
directly -- no new HF fetch, and no single-shard representativeness problem (D2 spans all 83
shards by construction), superseding the old 206-clip/n=43 E5-pilot-based analysis rather than
needing to be reconciled with it.

Boundary estimators:
  - canonical: `last_active_t` from d2_stratified_signal_features.parquet (tinyturn.boundary's
    fixed-threshold formula -- the one the model input contract and training are built on).
  - alt_threshold: a genuinely different fixed-threshold energy detector (median-based noise floor
    instead of the canonical's 10th-percentile, and different offsets) computed fresh here. The
    original Part-2 alt-threshold formula that produced `last_active_B_energy_alt` isn't present in
    this working tree (that pilot's generating script lived outside scripts) -- this is a new,
    independently-reasonable alternative, not a reproduction, which is fine: this diagnostic's job is
    "does *a* plausible alternative energy detector disagree enough to matter," not "does this exact
    historical detector disagree."
  - silero_vad: snakers4/silero-vad (jit model), run fully offline from the already-populated
    torch.hub cache -- end of the last detected speech segment.

Threshold discipline (brief step 2): reuse each model's existing calibrated threshold from
metrics.json (calibrated on the `calib` split under the canonical boundary during 8d/8g) rather
than recomputing one here -- keeps the threshold fixed across every boundary variant, as required.

Evaluation (brief step 3): metrics are reported on the val split only (the model-selection
validation data 8g actually gates on), not calib.

Reporting (brief step 5): real vs. synthetic; complete vs. incomplete; signed Δt (positive = later
alt boundary, negative = earlier); |Δt| bins (0-50/50-100/100-200/>200ms) -- for both the frac>0.20
/ flip-rate proportions and, where n allows, real-audio FCR-at-fixed-recall degradation.

Usage:
  python scripts/vad_boundary_diagnostic_full_val.py
"""
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
from tinyturn.whisper_model import WhisperEndpointModel, WHISPER_MODEL_NAME
from tinyturn.whisper_dataset import extract_whisper_features
from tinyturn.evaluate import fcr_at_fixed_recall
from transformers import WhisperFeatureExtractor

CACHE_DIR = Path("data_cache")
WAV_DIR = CACHE_DIR / "d2_stratified_wavs"
# Optional sys.argv[1]: run against a different A0 checkpoint dir (e.g. the 8g-remediation
# boundary-robust retrain). OUT_PATH is suffixed by that dir's name so it never clobbers the
# canonical A0's own recorded 8f result -- 8g's qualification script reads OUT_PATH by name, so a
# remediation rerun must point it there explicitly too (see qualify_teacher_a0_ci_gated.py's own
# sys.argv[1] override).
A0_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments") / "A0_whisper_tiny_pv2speechend"
B1_DIR = Path("experiments") / "C1_B1_1s_pv2speechend"
OUT_PATH = (Path("experiments") / "8f_vad_boundary_diagnostic_v2.json" if len(sys.argv) <= 1
            else Path("experiments") / f"8f_vad_boundary_diagnostic_v2__{A0_DIR.name}.json")
N_MELS, N_FFT = 40, 512
FRAME_LENGTH_S, HOP_LENGTH_S = 0.025, 0.010
TRAJECTORY_NAMES = ["rel_energy", "pause_prob", "spectral_tilt", "spectral_flux", "envelope_activity"]
MIN_N_FOR_FCR = 20
DT_BIN_EDGES = [0, 0.05, 0.10, 0.20, np.inf]
DT_BIN_LABELS = ["0-50ms", "50-100ms", "100-200ms", ">200ms"]

# alt-threshold detector constants -- deliberately different from tinyturn.boundary's canonical
# formula (noise_floor = 10th-percentile RMS-dB, thresh = max(noise_floor+10, peak-40)).
ALT_FRAME_LENGTH_S, ALT_HOP_LENGTH_S = 0.020, 0.010
ALT_NOISE_FLOOR_OFFSET_DB = -15.0   # median(rms_db) - 15, not 10th-percentile + 10
ALT_PEAK_OFFSET_DB = -30.0          # peak_db - 30, not peak_db - 40


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


def _a0_prob(model, feature_extractor, y, sr, speech_end_s, context_s):
    ex = build_example(y, sr, speech_end_s, context_s, frame_length_s=FRAME_LENGTH_S,
                        hop_length_s=HOP_LENGTH_S, label=False, row_id="diag")
    input_features, vfm = extract_whisper_features(feature_extractor, ex.waveform, ex.valid_sample_mask, sr)
    with torch.no_grad():
        logit = model(torch.from_numpy(input_features).unsqueeze(0), torch.from_numpy(vfm).unsqueeze(0))
    return float(torch.sigmoid(logit).item())


def _dt_bin(abs_dt):
    idx = np.digitize([abs_dt], DT_BIN_EDGES[1:-1])[0]
    return DT_BIN_LABELS[idx]


def _slice_summary(sub: pd.DataFrame, alt_name: str, threshold: float):
    col_diff, col_flip = f"abs_diff_{alt_name}", f"flip_{alt_name}"
    n = len(sub)
    if n == 0:
        return {"n": 0}
    frac_gt_020 = float((sub[col_diff] > 0.20).mean())
    flip_rate = float(sub[col_flip].mean())
    entry = {
        "n": int(n),
        "frac_change_gt_0.20": round(frac_gt_020, 5),
        "decision_flip_rate": round(flip_rate, 5),
        "criterion_frac_gt_020_le_0.10": bool(frac_gt_020 <= 0.10),
        "criterion_flip_rate_le_0.05": bool(flip_rate <= 0.05),
    }
    real_sub = sub[sub["real"]]
    if len(real_sub) >= MIN_N_FOR_FCR and real_sub["endpoint_bool"].nunique() > 1:
        fcr_canon = fcr_at_fixed_recall(real_sub["endpoint_bool"].values, real_sub["prob_canonical"].values)
        fcr_alt = fcr_at_fixed_recall(real_sub["endpoint_bool"].values, real_sub[f"prob_{alt_name}"].values)
        entry["real_n"] = int(len(real_sub))
        entry["real_fcr_at_fixed_recall_canonical"] = round(fcr_canon, 5)
        entry["real_fcr_at_fixed_recall_alt"] = round(fcr_alt, 5)
        entry["real_fcr_degradation_pp"] = round((fcr_alt - fcr_canon) * 100, 3)
        entry["criterion_fcr_degradation_le_2pp"] = bool((fcr_alt - fcr_canon) * 100 <= 2.0)
    else:
        entry["real_fcr_note"] = f"skipped: only {len(real_sub)} real-audio rows (need >= {MIN_N_FOR_FCR} w/ both classes)"
    return entry


def _run_model(name, prob_fn, df, context_s, threshold):
    rows = []
    t0 = time.time()
    for i, r in df.iterrows():
        y, sr = _load_wav(r["id"])
        canonical = prob_fn(y, sr, float(r["last_active_t"]), context_s)
        row = {"id": r["id"], "real": not bool(r["synthetic"]), "endpoint_bool": bool(r["endpoint_bool"]),
               "language": r["language"], "dataset": r["dataset"], "prob_canonical": canonical}
        for alt_name, boundary_col in [("alt_threshold", "alt_threshold_boundary_s"),
                                        ("silero_vad", "silero_boundary_s")]:
            se = r[boundary_col]
            if pd.isna(se):
                continue
            p = prob_fn(y, sr, float(se), context_s)
            dt = float(se) - float(r["last_active_t"])  # signed: + = alt boundary later than canonical
            row[f"prob_{alt_name}"] = p
            row[f"abs_diff_{alt_name}"] = abs(p - canonical)
            row[f"flip_{alt_name}"] = (p >= threshold) != (canonical >= threshold)
            row[f"signed_dt_{alt_name}"] = dt
            row[f"dt_bin_{alt_name}"] = _dt_bin(abs(dt))
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  [{name}] {i + 1}/{len(df)} ({time.time() - t0:.0f}s)", flush=True)
    out = pd.DataFrame(rows)

    summary = {}
    for alt_name in ["alt_threshold", "silero_vad"]:
        if f"abs_diff_{alt_name}" not in out.columns:
            continue
        sub_all = out[out[f"abs_diff_{alt_name}"].notna()]
        entry = {"overall": _slice_summary(sub_all, alt_name, threshold)}
        entry["by_real_synthetic"] = {
            "real": _slice_summary(sub_all[sub_all["real"]], alt_name, threshold),
            "synthetic": _slice_summary(sub_all[~sub_all["real"]], alt_name, threshold),
        }
        entry["by_endpoint"] = {
            "complete": _slice_summary(sub_all[sub_all["endpoint_bool"]], alt_name, threshold),
            "incomplete": _slice_summary(sub_all[~sub_all["endpoint_bool"]], alt_name, threshold),
        }
        entry["by_signed_shift_direction"] = {
            "alt_later_than_canonical": _slice_summary(sub_all[sub_all[f"signed_dt_{alt_name}"] > 0], alt_name, threshold),
            "alt_earlier_than_canonical": _slice_summary(sub_all[sub_all[f"signed_dt_{alt_name}"] < 0], alt_name, threshold),
        }
        entry["by_abs_dt_bin"] = {
            lbl: _slice_summary(sub_all[sub_all[f"dt_bin_{alt_name}"] == lbl], alt_name, threshold)
            for lbl in DT_BIN_LABELS
        }
        summary[alt_name] = entry
    return out, summary


def main():
    for d in (A0_DIR, B1_DIR):
        if not (d / "checkpoint.pt").exists():
            print(f"ERROR: {d / 'checkpoint.pt'} not found -- run the 8d retrain scripts first.")
            sys.exit(1)

    splits = pd.read_parquet(CACHE_DIR / "tinyturn_splits.parquet")
    sf_feat = pd.read_parquet(CACHE_DIR / "d2_stratified_signal_features.parquet")[["id", "last_active_t"]]
    df = splits[splits["split"] == "val"].merge(sf_feat, on="id", how="left")
    df = df[df["last_active_t"].notna()].reset_index(drop=True)
    print(f"8f-v2: {len(df)} val clips (full D2 val split, all 83 shards) -- computing fresh "
          f"alt-threshold + Silero boundaries", flush=True)

    print("loading Silero VAD (offline, from torch.hub cache)...", flush=True)
    silero_model, silero_utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', source='github',
                                                  trust_repo=True, onnx=False)
    get_speech_timestamps = silero_utils[0]

    print("computing alt-threshold + Silero boundaries for all val clips...", flush=True)
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
        df, float(b1_cfg["context_s"]), float(b1_metrics["threshold"]))

    result = {
        "n_val_clips": len(df),
        "note": "recomputed on the full 1,600-clip val split spanning all 83 D2 shards -- "
                "supersedes the 206-clip/n=43 pilot overlap in PHASE2_RESULTS_8a-9.md's 8f section.",
        "alt_threshold_detector": {
            "frame_length_s": ALT_FRAME_LENGTH_S, "hop_length_s": ALT_HOP_LENGTH_S,
            "formula": "thresh_db = max(median(rms_db) - 15, peak_db - 30)",
            "note": "independently-reasonable alternative energy detector, not a reproduction of "
                    "the original Part-2 E5 alt-threshold formula (that generating script isn't in "
                    "this working tree).",
        },
        "silero_vad": {"source": "snakers4/silero-vad", "onnx": False},
        "A0": a0_summary, "B1_1s": b1_summary,
    }
    with open(OUT_PATH, "w") as f:
        json.dump({"summary": result,
                   "per_clip_A0": a0_out.to_dict(orient="records"),
                   "per_clip_B1_1s": b1_out.to_dict(orient="records")}, f, indent=2, default=str)

    print(json.dumps({k: v for k, v in result.items() if k not in ("A0", "B1_1s")}, indent=2))
    print("\nA0 overall vs alt_threshold:", json.dumps(a0_summary.get("alt_threshold", {}).get("overall", {}), indent=2))
    print("A0 overall vs silero_vad:", json.dumps(a0_summary.get("silero_vad", {}).get("overall", {}), indent=2))
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
