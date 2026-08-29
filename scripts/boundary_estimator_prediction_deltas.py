"""
E5 completion -- prediction deltas. Reuses e5_vad_sensitivity_results.csv's three per-clip
last_active_t estimates (A: fixed energy threshold, B: alt threshold, C: Silero VAD) on the
existing 3k local sample, recomputes the full PROBE_FEATURES set anchored to each estimator's
boundary, and applies a fixed trained probe (same family/training as D6/E4) to get a predicted
probability per estimator per clip -- completing the "and the resulting change in ... predictions"
half of E5 that was pending on D6.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.preprocessing import StandardScaler as Scaler

import sys
sys.path.insert(0, str(Path(__file__).parent))
from d6_context_probe import PROBE_FEATURES, segment_runs, slope_in_window

RNG_SEED = 42
CACHE_DIR = Path("data_cache")
WAV_DIR = CACHE_DIR / "d2_stratified_wavs"


def features_at_boundary(y, sr, last_active_t):
    y = y.astype(np.float32)
    duration = len(y) / sr
    frame_length, hop_length = int(0.032 * sr), int(0.010 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    rms_db = 20 * np.log10(np.maximum(rms, 1e-6))
    rms_mean_linear = float(np.mean(rms)) if len(rms) else np.nan
    noise_floor_db = float(np.percentile(rms_db, 10))
    peak_db = float(rms_db.max())
    thresh_db = max(noise_floor_db + 10, peak_db - 40)
    active = rms_db > thresh_db
    active_time_s = float(active.sum() * hop_length / sr)
    voiced_time_fraction = active_time_s / duration if duration > 0 else np.nan
    trailing_silence_s = max(duration - last_active_t, 0.0)

    runs = segment_runs(active, times)
    internal_pause_count = 0
    if len(runs) >= 3:
        for start, end, is_act in runs[1:-1]:
            if not is_act and (end - start) >= 0.1:
                internal_pause_count += 1

    energy_slope_1000ms = slope_in_window(times, rms_db, last_active_t, 1.0)
    rel_energy = rms / rms_mean_linear if rms_mean_linear and rms_mean_linear > 0 else np.full_like(rms, np.nan)
    energy_slope_relative = slope_in_window(times, rel_energy, last_active_t, 1.0)

    f0 = librosa.yin(y, fmin=70, fmax=400, sr=sr, frame_length=800, hop_length=200)
    f0_times = np.arange(len(f0)) * 200 / sr
    voiced = f0 > 0
    f0_slope_1000ms = np.nan
    if voiced.sum() >= 5:
        f0_ref = float(np.median(f0[voiced]))
        vt, vs = f0_times[voiced], 12 * np.log2(f0[voiced] / f0_ref)
        f0_slope_1000ms = slope_in_window(vt, vs, last_active_t, 1.0)

    n_fft = 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_e = S[(freqs >= 80) & (freqs < 1000)].mean(axis=0)
    high_e = S[(freqs >= 1000) & (freqs < 4000)].mean(axis=0)
    tilt_db = 20*np.log10(np.maximum(low_e,1e-6)) - 20*np.log10(np.maximum(high_e,1e-6))
    stft_times = librosa.frames_to_time(np.arange(len(tilt_db)), sr=sr, hop_length=hop_length)
    spectral_tilt_slope = slope_in_window(stft_times, tilt_db, last_active_t, 1.0)

    return dict(duration_s=duration, trailing_silence_s=trailing_silence_s,
                energy_slope_1000ms=energy_slope_1000ms, f0_slope_1000ms=f0_slope_1000ms,
                energy_slope_relative=energy_slope_relative, internal_pause_count=internal_pause_count,
                voiced_time_fraction=voiced_time_fraction, spectral_tilt_slope=spectral_tilt_slope)


def main():
    vad_df = pd.read_csv(CACHE_DIR.parent / "eda_outputs" / "tables" / "e5_vad_sensitivity_results.csv")
    sig = pd.read_parquet(CACHE_DIR / "signal_features_v3.parquet")
    id_to_file = sig.set_index("id")["file"].to_dict()

    d2feat = pd.read_parquet(CACHE_DIR / "d2_stratified_signal_features.parquet")
    d2feat = d2feat[d2feat["sr"].notna()]
    train_sample = d2feat.sample(n=min(5000, len(d2feat)), random_state=RNG_SEED)
    t0 = time.time()
    train_rows = []
    for _, r in train_sample.iterrows():
        try:
            data, sr = sf.read(Path("data_cache/d2_stratified_wavs") / f"{r['id']}.wav")
        except Exception:
            continue
        y = data if data.ndim == 1 else data.mean(axis=1)
        n_samp = int(8.0 * sr)
        y_ctx = y[-n_samp:] if len(y) > n_samp else y
        from d6_context_probe import compute_probe_features_fast
        feats = compute_probe_features_fast(y_ctx, sr)
        train_rows.append({"endpoint_bool": r["endpoint_bool"], **feats})
    train_df = pd.DataFrame(train_rows)
    X = train_df[PROBE_FEATURES].values.astype(float)
    y_lab = train_df["endpoint_bool"].astype(int).values
    missing_cols = np.where(np.isnan(X).any(axis=0))[0]
    for col in missing_cols:
        med = np.nanmedian(X[:, col])
        X[np.isnan(X[:, col]), col] = med if med == med else 0.0
    scaler = Scaler().fit(X)
    clf = LogReg(max_iter=1000).fit(scaler.transform(X), y_lab)
    print(f"trained fixed model in {time.time()-t0:.1f}s, n={len(train_df)}", flush=True)

    def predict_for_boundary(y, sr, last_active_t):
        feats = features_at_boundary(y, sr, last_active_t)
        x = np.array([[feats[f] for f in PROBE_FEATURES]], dtype=float)
        for col in missing_cols:
            if np.isnan(x[0, col]):
                med = np.nanmedian(X[:, col])
                x[0, col] = med if med == med else 0.0
        return clf.predict_proba(scaler.transform(x))[0, 1]

    t1 = time.time()
    rows = []
    for i, (_, r) in enumerate(vad_df.iterrows()):
        fpath = id_to_file.get(r["id"])
        if fpath is None:
            continue
        try:
            data, sr = sf.read(fpath)
        except Exception:
            continue
        y = data if data.ndim == 1 else data.mean(axis=1)
        preds = {}
        for label, col in [("A", "last_active_A_energy_fixed"), ("B", "last_active_B_energy_alt"),
                            ("C", "last_active_C_silero_vad")]:
            la_t = r[col]
            if pd.isna(la_t):
                preds[f"pred_{label}"] = np.nan
                continue
            try:
                preds[f"pred_{label}"] = predict_for_boundary(y, sr, la_t)
            except Exception:
                preds[f"pred_{label}"] = np.nan
        rows.append({"id": r["id"], **preds})
        if (i + 1) % 500 == 0:
            print(f"{i+1}/{len(vad_df)}, {time.time()-t1:.1f}s", flush=True)

    out = pd.DataFrame(rows)
    out["dpred_AB"] = (out["pred_A"] - out["pred_B"]).abs()
    out["dpred_AC"] = (out["pred_A"] - out["pred_C"]).abs()
    out["dpred_BC"] = (out["pred_B"] - out["pred_C"]).abs()
    out.to_csv("eda_outputs/tables/e5_prediction_deltas.csv", index=False)
    print(f"\nDONE, {time.time()-t1:.1f}s")
    for pair in ["AB", "AC", "BC"]:
        d = out[f"dpred_{pair}"].dropna()
        print(f"pred delta {pair}: n={len(d)} mean={d.mean():.4f} median={d.median():.4f} "
              f"frac>0.2={(d>0.2).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
