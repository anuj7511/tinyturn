"""
D6 -- controlled context-length probes (trained), using D2's stratified sample.

Speed fix vs. the first version: profiling showed compute_signal_features_v3's `pyin` call takes
0.54s/clip (70% of total runtime) -- fine for a one-time full feature pass (D2), but running it x5
context lengths x 15,998 clips would take ~17 hours. This version computes ONLY the 8 features
PROBE_FEATURES actually uses, and swaps `pyin` for `yin` (~3.3x faster, same choice C2 already made
for the same reason) since pitch-tracking robustness isn't the point here -- holding the model and
procedure fixed across context lengths is. Sample size reduced to n=5000 (same precedent as C2/D5),
still >>800 and >>the original context-window analysis.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.preprocessing import StandardScaler as Scaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RNG_SEED = 42
CACHE_DIR = Path("data_cache")
WAV_DIR = CACHE_DIR / "d2_stratified_wavs"
CONTEXT_LENGTHS_S = [1, 2, 4, 6, 8]
N_SAMPLE = 5000

PROBE_FEATURES = ["duration_s", "trailing_silence_s", "energy_slope_1000ms", "f0_slope_1000ms",
                   "energy_slope_relative", "internal_pause_count", "voiced_time_fraction",
                   "spectral_tilt_slope"]
MISSING_PRONE = ["f0_slope_1000ms"]


def segment_runs(active, times):
    runs = []
    if len(active) == 0:
        return runs
    run_start = 0
    cur = bool(active[0])
    for i in range(1, len(active)):
        if bool(active[i]) != cur:
            runs.append((float(times[run_start]), float(times[i - 1]), cur))
            run_start = i
            cur = bool(active[i])
    runs.append((float(times[run_start]), float(times[-1]), cur))
    return runs


def slope_in_window(times, values, window_end, window_size, min_frac=0.4, min_pts=3):
    mask = (times >= max(0, window_end - window_size)) & (times <= window_end)
    n = int(mask.sum())
    if n < min_pts:
        return np.nan
    if len(times) >= 2:
        med_dt = np.median(np.diff(np.sort(times)))
        expected = window_size / med_dt if med_dt > 0 else n
        if n < min_frac * expected:
            return np.nan
    return float(np.polyfit(times[mask], values[mask], 1)[0])


def compute_probe_features_fast(y, sr):
    """Only the 8 PROBE_FEATURES, using `yin` instead of `pyin` for speed (see module docstring)."""
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


def bootstrap_auc_ci(y_true, scores, n_boot=200, seed=42):
    rng = np.random.RandomState(seed)
    valid = (~pd.isna(scores)) & (~pd.isna(y_true))
    yt, sc = np.asarray(y_true)[valid].astype(bool), np.asarray(scores)[valid]
    if len(yt) < 20 or len(set(yt)) < 2:
        return np.nan, np.nan
    aucs = []
    n = len(yt)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb, sb = yt[idx], sc[idx]
        if len(set(yb)) < 2:
            continue
        a = roc_auc_score(yb, sb)
        aucs.append(max(a, 1 - a))
    return (np.nan, np.nan) if not aucs else (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5))


def cv_probe_with_imputation(X, missing_mask_cols, y, seed=RNG_SEED):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    oof_probs = np.full(len(y), np.nan)
    # impute every column with any NaN (short context windows can leave several features
    # undefined, e.g. slope_in_window's min_pts requirement), not just the ones we expected --
    # train-fold-only median per D12's decision either way.
    all_missing_cols = sorted(set(missing_mask_cols) | set(np.where(np.isnan(X).any(axis=0))[0].tolist()))
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
        for col in all_missing_cols:
            med = np.nanmedian(X_train[:, col])
            med = med if med == med else 0.0  # entire train fold NaN for this col -> fall back to 0
            X_train[np.isnan(X_train[:, col]), col] = med
            X_test[np.isnan(X_test[:, col]), col] = med
        scaler = Scaler().fit(X_train)
        clf = LogReg(max_iter=1000).fit(scaler.transform(X_train), y[train_idx])
        accs.append(clf.score(scaler.transform(X_test), y[test_idx]))
        oof_probs[test_idx] = clf.predict_proba(scaler.transform(X_test))[:, 1]
    return accs, oof_probs


def main():
    feat_full = pd.read_parquet(CACHE_DIR / "d2_stratified_signal_features.parquet")
    feat_full = feat_full[feat_full["sr"].notna()]
    sample = feat_full.sample(n=min(N_SAMPLE, len(feat_full)), random_state=RNG_SEED)
    print(f"D6: n={len(sample)} sample, {len(CONTEXT_LENGTHS_S)} context lengths", flush=True)

    results = []
    for ctx_s in CONTEXT_LENGTHS_S:
        rows = []
        t0 = time.time()
        for i, (_, r) in enumerate(sample.iterrows()):
            try:
                data, sr = sf.read(WAV_DIR / f"{r['id']}.wav")
            except Exception:
                continue
            y = data if data.ndim == 1 else data.mean(axis=1)
            n_samp = int(ctx_s * sr)
            y_ctx = y[-n_samp:] if len(y) > n_samp else y
            feats = compute_probe_features_fast(y_ctx, sr)
            rows.append({"id": r["id"], "endpoint_bool": r["endpoint_bool"], **feats})
            if (i + 1) % 1000 == 0:
                print(f"  ctx={ctx_s}s: {i+1}/{len(sample)}, {time.time()-t0:.1f}s", flush=True)

        df = pd.DataFrame(rows)
        X = df[PROBE_FEATURES].values.astype(float)
        missing_cols = [PROBE_FEATURES.index(c) for c in MISSING_PRONE]
        y_lab = df["endpoint_bool"].astype(int).values
        n = len(df)

        accs, oof_probs = cv_probe_with_imputation(X, missing_cols, y_lab)
        auc_raw = roc_auc_score(y_lab, oof_probs)
        auc = max(auc_raw, 1 - auc_raw)
        ci_lo, ci_hi = bootstrap_auc_ci(y_lab, oof_probs)
        majority = max(y_lab.mean(), 1 - y_lab.mean())

        results.append({
            "context_length_s": ctx_s, "n": n, "cv_accuracy": round(np.mean(accs), 4),
            "cv_accuracy_std": round(np.std(accs), 4), "majority_baseline": round(majority, 4),
            "auc_direction_normalized": round(auc, 4), "auc_ci_low": round(ci_lo, 4),
            "auc_ci_high": round(ci_hi, 4),
        })
        print(f"context={ctx_s}s: n={n} acc={np.mean(accs):.3f} auc={auc:.3f} "
              f"[{ci_lo:.3f},{ci_hi:.3f}]  ({time.time()-t0:.1f}s)", flush=True)

        out = pd.DataFrame(results)
        out.to_csv("eda_outputs/tables/d6_context_probe_results.csv", index=False)

    print("\nsaved d6_context_probe_results.csv")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
