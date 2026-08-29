"""
C0 -- real-only context-length probe (brief Step 2).

D6 (`context_probe.py`) is population-wide (mixes real+synthetic), sweeps 1/2/4/6/8s, and
includes an F0 feature. This is deliberately different, per the brief's own Step 2: real-only (all
real, real English, real Spanish), context lengths 0.5/1/2/4s, and the reliable non-F0 feature
family only (energy, tilt, flux, envelope/pause -- no pyin/yin at all, so this also sidesteps the
speed problem D6's docstring flagged).

Reuses `segment_runs` / `slope_in_window` from context_probe.py verbatim (same threshold
convention as tinyturn.boundary) rather than reimplementing them a fourth time.

Output feeds the provisional `N` default for Step 1/3/4 (TinyTurnDataset's `context_s`) -- this is
an initial hypothesis only; Step 6 reruns the sweep on the trained model and makes the final call.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.preprocessing import StandardScaler as Scaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from d6_context_probe import segment_runs, slope_in_window, bootstrap_auc_ci  # noqa: E402

RNG_SEED = 42
CACHE_DIR = Path("data_cache")
WAV_DIR = CACHE_DIR / "d2_stratified_wavs"
OUT_DIR = Path("experiments/context_probe_handcrafted")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_LENGTHS_S = [0.5, 1, 2, 4]

# energy / tilt / flux / envelope-pause family only -- no F0, per the brief's Step 2 instruction.
PROBE_FEATURES = [
    "duration_s", "trailing_silence_s",
    "energy_slope_1000ms", "energy_slope_relative", "internal_pause_count", "voiced_time_fraction",
    "spectral_tilt_slope", "spectral_flux_mean",
    "speaking_rate_slope",
]


def compute_probe_features_no_f0(y, sr):
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

    runs = segment_runs(active, times)
    last_active_t = duration
    for start, end, is_act in runs[::-1]:
        if is_act:
            last_active_t = end
            break
    trailing_silence_s = max(duration - last_active_t, 0.0)

    internal_pause_count = 0
    if len(runs) >= 3:
        for start, end, is_act in runs[1:-1]:
            if not is_act and (end - start) >= 0.1:
                internal_pause_count += 1

    energy_slope_1000ms = slope_in_window(times, rms_db, last_active_t, 1.0)
    rel_energy = rms / rms_mean_linear if rms_mean_linear and rms_mean_linear > 0 else np.full_like(rms, np.nan)
    energy_slope_relative = slope_in_window(times, rel_energy, last_active_t, 1.0)

    n_fft = 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_e = S[(freqs >= 80) & (freqs < 1000)].mean(axis=0)
    high_e = S[(freqs >= 1000) & (freqs < 4000)].mean(axis=0)
    tilt_db = 20 * np.log10(np.maximum(low_e, 1e-6)) - 20 * np.log10(np.maximum(high_e, 1e-6))
    stft_times = librosa.frames_to_time(np.arange(len(tilt_db)), sr=sr, hop_length=hop_length)
    spectral_tilt_slope = slope_in_window(stft_times, tilt_db, last_active_t, 1.0)
    flux = np.concatenate([[0.0], np.sqrt(np.mean(np.diff(S, axis=1) ** 2, axis=0))]) if S.shape[1] > 0 else np.zeros(0)
    tail_mask = stft_times >= max(0, last_active_t - 1.0)
    spectral_flux_mean = float(flux[tail_mask].mean()) if tail_mask.sum() >= 3 else np.nan

    speaking_rate_slope = np.nan
    try:
        nyq = sr / 2
        b, a = butter(4, [300 / nyq, min(2000 / nyq, 0.99)], btype="band")
        y_band = filtfilt(b, a, y)
        from scipy.signal import hilbert
        envelope = np.abs(hilbert(y_band))
        win = max(int(0.02 * sr), 1)
        envelope_smooth = np.convolve(envelope, np.ones(win) / win, mode="same")
        active_samples = np.repeat(active, hop_length)[:len(envelope_smooth)]
        env_for_peaks = envelope_smooth.copy()
        if len(active_samples) == len(env_for_peaks):
            env_for_peaks[~active_samples] = 0
        peak_idx, _ = find_peaks(env_for_peaks, distance=int(0.1 * sr),
                                  prominence=0.1 * (env_for_peaks.max() + 1e-9))
        peak_times = peak_idx / sr
        if len(peak_times) >= 4:
            iois = np.diff(peak_times)
            speaking_rate_slope = float(np.polyfit(peak_times[1:], iois, 1)[0])
    except Exception:
        pass

    return dict(duration_s=duration, trailing_silence_s=trailing_silence_s,
                energy_slope_1000ms=energy_slope_1000ms, energy_slope_relative=energy_slope_relative,
                internal_pause_count=internal_pause_count, voiced_time_fraction=voiced_time_fraction,
                spectral_tilt_slope=spectral_tilt_slope, spectral_flux_mean=spectral_flux_mean,
                speaking_rate_slope=speaking_rate_slope)


def cv_probe_with_imputation(X, y, seed=RNG_SEED):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    oof_probs = np.full(len(y), np.nan)
    all_missing_cols = sorted(np.where(np.isnan(X).any(axis=0))[0].tolist())
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
        for col in all_missing_cols:
            med = np.nanmedian(X_train[:, col])
            med = med if med == med else 0.0
            X_train[np.isnan(X_train[:, col]), col] = med
            X_test[np.isnan(X_test[:, col]), col] = med
        scaler = Scaler().fit(X_train)
        clf = LogReg(max_iter=1000).fit(scaler.transform(X_train), y[train_idx])
        accs.append(clf.score(scaler.transform(X_test), y[test_idx]))
        oof_probs[test_idx] = clf.predict_proba(scaler.transform(X_test))[:, 1]
    return accs, oof_probs


def run_slice(slice_name, sample, ctx_s):
    rows = []
    t0 = time.time()
    for _, r in sample.iterrows():
        try:
            data, sr = sf.read(WAV_DIR / f"{r['id']}.wav")
        except Exception:
            continue
        y = data if data.ndim == 1 else data.mean(axis=1)
        n_samp = int(ctx_s * sr)
        y_ctx = y[-n_samp:] if len(y) > n_samp else y
        feats = compute_probe_features_no_f0(y_ctx, sr)
        rows.append({"id": r["id"], "endpoint_bool": r["endpoint_bool"], **feats})

    df = pd.DataFrame(rows)
    X = df[PROBE_FEATURES].values.astype(float)
    y_lab = df["endpoint_bool"].astype(int).values
    n = len(df)

    accs, oof_probs = cv_probe_with_imputation(X, y_lab)
    auc_raw = roc_auc_score(y_lab, oof_probs)
    auc = max(auc_raw, 1 - auc_raw)
    ci_lo, ci_hi = bootstrap_auc_ci(y_lab, oof_probs)
    majority = max(y_lab.mean(), 1 - y_lab.mean())

    result = {
        "slice": slice_name, "context_length_s": ctx_s, "n": n,
        "cv_accuracy": round(float(np.mean(accs)), 4), "cv_accuracy_std": round(float(np.std(accs)), 4),
        "majority_baseline": round(float(majority), 4),
        "auc_direction_normalized": round(float(auc), 4),
        "auc_ci_low": round(float(ci_lo), 4), "auc_ci_high": round(float(ci_hi), 4),
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"[{slice_name}] ctx={ctx_s}s n={n} acc={result['cv_accuracy']:.3f} "
          f"auc={auc:.3f} [{ci_lo:.3f},{ci_hi:.3f}]  ({result['elapsed_s']}s)", flush=True)
    return result


def main():
    feat_full = pd.read_parquet(CACHE_DIR / "d2_stratified_signal_features.parquet")
    feat_full = feat_full[feat_full["sr"].notna()]
    real = feat_full[~feat_full["synthetic"]]

    slices = {
        "real_all": real,
        "real_eng": real[real["language"] == "eng"],
        "real_spa": real[real["language"] == "spa"],
    }
    for name, df in slices.items():
        print(f"slice {name}: n={len(df)}")

    results = []
    for ctx_s in CONTEXT_LENGTHS_S:
        for slice_name, df in slices.items():
            results.append(run_slice(slice_name, df, ctx_s))
        pd.DataFrame(results).to_csv(OUT_DIR / "context_probe_results.csv", index=False)

    out = pd.DataFrame(results)
    print("\n" + out.to_string(index=False))

    # provisional N: highest-AUC context length on the real_all slice (the largest-n, most stable
    # estimate); real_eng/real_spa are reported for transparency but real_spa's n=236 makes its own
    # per-length CV noisy on its own -- it should inform, not solely decide, N.
    real_all = out[out["slice"] == "real_all"].sort_values("auc_direction_normalized", ascending=False)
    provisional_n = float(real_all.iloc[0]["context_length_s"])
    print(f"\nprovisional N (real_all, best AUC): {provisional_n}s")

    with open(OUT_DIR / "config.json", "w") as f:
        json.dump({
            "provisional_context_s": provisional_n,
            "postroll_s": 0.2,
            "note": ("Handcrafted real-only probe, Step 2/C0. Initial hypothesis for N -- Step 6 "
                     "reruns the sweep on the trained model and makes the final decision, matching "
                     "D6-vs-E1's relationship in the EDA."),
        }, f, indent=2)
    print(f"saved {OUT_DIR / 'config.json'} and {OUT_DIR / 'context_probe_results.csv'}")


if __name__ == "__main__":
    main()
