"""
D5 -- full temporal trajectories over the final 2s, aligned to last_active_t, using D2's decoded
sample. n=4000 subsample of D2's 15,998 (compute reasons, documented -- not on the brief's
"must finish before architecture" list).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG_SEED = 42
WAV_DIR = Path("data_cache/d2_stratified_wavs")
N_SAMPLE = 4000
WINDOW_S = 2.0
N_POINTS = 100  # 20ms resolution over the 2s window
GRID = np.linspace(-WINDOW_S, 0, N_POINTS)  # t=0 is last_active_t


def extract_trajectory(y, sr, last_active_t):
    y = y.astype(np.float32)
    frame_length, hop_length = int(0.032 * sr), int(0.010 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    rms_db = 20 * np.log10(np.maximum(rms, 1e-6))
    rms_mean_db = rms_db.mean()
    rel_energy = rms_db - rms_mean_db

    noise_floor_db = np.percentile(rms_db, 10)
    peak_db = rms_db.max()
    thresh_db = max(noise_floor_db + 10, peak_db - 40)
    active = (rms_db > thresh_db).astype(float)
    pause_prob = 1 - active

    n_fft = 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_e = S[(freqs >= 80) & (freqs < 1000)].mean(axis=0)
    high_e = S[(freqs >= 1000) & (freqs < 4000)].mean(axis=0)
    tilt_db = 20*np.log10(np.maximum(low_e,1e-6)) - 20*np.log10(np.maximum(high_e,1e-6))
    stft_times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)
    flux = np.concatenate([[0], np.sqrt(np.mean(np.diff(S, axis=1)**2, axis=0))])

    # yin instead of pyin: ~3.3x faster (0.165s vs 0.537s/clip, profiled), same tradeoff C2 already
    # made for the same reason. voiced_prob is approximated as a binary voiced/unvoiced indicator
    # (yin doesn't give a soft probability) -- acceptable since this is a trajectory-shape probe,
    # not a pitch-tracking-accuracy benchmark.
    f_hop = 200
    f0 = librosa.yin(y, fmin=70, fmax=400, sr=sr, frame_length=800, hop_length=f_hop)
    f0_times = np.arange(len(f0)) * f_hop / sr
    voiced = f0 > 0
    voiced_prob = voiced.astype(float)
    f0_ref = float(np.median(f0[voiced])) if voiced.sum() >= 5 else np.nan
    semitones = 12*np.log2(np.maximum(f0,1e-6)/f0_ref) if f0_ref == f0_ref else np.full_like(f0, np.nan)
    semitones = np.where(voiced, semitones, np.nan)

    def resample_to_grid(src_times, src_vals, rel_grid):
        query_t = last_active_t + rel_grid
        valid_src = ~np.isnan(src_vals)
        if valid_src.sum() < 2:
            return np.full(len(rel_grid), np.nan)
        return np.interp(query_t, src_times[valid_src], src_vals[valid_src], left=np.nan, right=np.nan)

    return {
        "rel_energy": resample_to_grid(times, rel_energy, GRID),
        "pause_prob": resample_to_grid(times, pause_prob, GRID),
        "spectral_tilt": resample_to_grid(stft_times, tilt_db, GRID),
        "spectral_flux": resample_to_grid(stft_times, flux, GRID),
        "pitch_semitones": resample_to_grid(f0_times, semitones, GRID),
        "voicing_prob": resample_to_grid(f0_times, voiced_prob.astype(float), GRID),
    }


def main():
    feat = pd.read_parquet("data_cache/d2_stratified_signal_features.parquet")
    feat = feat[feat["sr"].notna() & feat["last_active_t"].notna()].copy()
    trans = pd.read_parquet("data_cache/d2_stratified_transcripts.parquet")
    feat = feat.merge(trans[["id", "endfiller_derived"]], on="id", how="left")

    sample = feat.sample(n=min(N_SAMPLE, len(feat)), random_state=RNG_SEED)
    print(f"D5: extracting trajectories for {len(sample)} clips")

    channels = ["rel_energy", "pause_prob", "spectral_tilt", "spectral_flux", "pitch_semitones", "voicing_prob"]
    arrays = {c: [] for c in channels}
    meta_rows = []
    t0 = time.time()
    for i, (_, row) in enumerate(sample.iterrows()):
        try:
            data, sr = sf.read(WAV_DIR / f"{row['id']}.wav")
        except Exception:
            continue
        y = data if data.ndim == 1 else data.mean(axis=1)
        try:
            traj = extract_trajectory(y, sr, row["last_active_t"])
        except Exception:
            continue
        for c in channels:
            arrays[c].append(traj[c])
        meta_rows.append({
            "id": row["id"], "endpoint_bool": row["endpoint_bool"], "synthetic": row["synthetic"],
            "language": row["language"], "dataset": row["dataset"], "endfiller_derived": row["endfiller_derived"],
        })
        if (i+1) % 500 == 0:
            print(f"{i+1}/{len(sample)} done, {time.time()-t0:.1f}s elapsed", flush=True)

    meta_df = pd.DataFrame(meta_rows)
    print(f"\nextracted {len(meta_df)} trajectories in {time.time()-t0:.1f}s")

    out = {"grid": GRID, "meta": meta_df.to_dict(orient="list")}
    for c in channels:
        out[c] = np.stack(arrays[c])
    np.savez("data_cache/d5_trajectory_arrays.npz", **out)
    meta_df.to_parquet("data_cache/d5_trajectory_meta.parquet")
    print("saved data_cache/d5_trajectory_arrays.npz + d5_trajectory_meta.parquet")

    # plot grid: rows = channels, cols = split dimensions
    splits = {
        "endpoint_bool": [("True", meta_df["endpoint_bool"] == True), ("False", meta_df["endpoint_bool"] == False)],
        "real_vs_synth": [("real", meta_df["synthetic"] == False), ("synthetic", meta_df["synthetic"] == True)],
        "language": [("hin", meta_df["language"] == "hin"), ("eng", meta_df["language"] == "eng"),
                     ("spa", meta_df["language"] == "spa")],
        "filler_state": [("filler=True", meta_df["endfiller_derived"] == True),
                          ("filler=False", meta_df["endfiller_derived"] == False)],
    }
    fig, axes = plt.subplots(len(channels), len(splits), figsize=(20, 3*len(channels)))
    for row_i, c in enumerate(channels):
        arr = out[c]
        for col_j, (split_name, groups) in enumerate(splits.items()):
            ax = axes[row_i, col_j]
            for label, mask in groups:
                mask = mask.values
                if mask.sum() < 5:
                    continue
                sub = arr[mask]
                mean_traj = np.nanmean(sub, axis=0)
                std_traj = np.nanstd(sub, axis=0)
                ax.plot(GRID, mean_traj, label=f"{label} (n={mask.sum()})")
                ax.fill_between(GRID, mean_traj-std_traj, mean_traj+std_traj, alpha=0.15)
            if row_i == 0:
                ax.set_title(split_name)
            if col_j == 0:
                ax.set_ylabel(c)
            ax.legend(fontsize=7)
            ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    plt.savefig("eda_outputs/plots/d5_terminal_trajectories.png", dpi=110)
    print("saved eda_outputs/plots/d5_terminal_trajectories.png")


if __name__ == "__main__":
    main()
