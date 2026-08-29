"""
Step 10 planning, item 4 -- 32k data-scaling tier: whole-shard fetch + decode + feature extraction,
WITHOUT ASR transcription.

Per the correction request: plain B1/pause/ranking training needs waveform, endpoint label,
metadata, speech boundaries, internal-pause events, and log-mel/trajectory features -- not GPT
transcription. `endfiller` ground truth already ships natively on synthetic rows in the raw HF
dataset (confirmed: real rows have it null 100% of the time, synthetic rows ~97% populated -- same
convention D2 already relies on), so skipping ASR loses nothing plain training needs.

Whole-shard fetching (8i's re-scoping, brief Section "8i"): `pipecat-ai/smart-turn-data-v3.2-train`
is stored as 83 individually-downloadable parquet shard files (confirmed via HfApi.list_repo_files),
~3265 rows / ~500MB each. Grabbing 5 whole shards directly avoids the 3.6-hour full-dataset stream
the original D2 build needed (it had to scan every one of 270,933 rows to find 15,998 IDs scattered
across all 83 shards) -- this instead downloads only the ~2.5GB these 5 shards contain.

Shards used: 10, 20, 30, 40, 50 -- checked individually (metadata columns only, no audio) against
the population-level language/synthetic/dataset composition in eda_complete_results.md Section B2
before committing (all five landed within ~1-2pp of the full 270,946-row population on every
column checked: language eng/spa, synthetic rate, dataset chirp3_1 share).

Appends ONLY -- never touches the existing 15,998 D2 rows or their split assignments. New rows are
always split="train" (never val/calib), per the brief's requirement that fixed validation/
calibration IDs never enter a larger training tier.

Generalized (Step 10 planning, item on 64k escalation) to accept any shard list via --shards, so the
same script builds any tier -- pass whichever new, not-yet-used, pre-verified-representative shard
indices are needed to reach the next target size.

Usage:
  python scripts/build_32k_scale_tier.py [--limit-per-shard N] [--shards 10,20,30,40,50]
"""
import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import pyarrow.parquet as pq
from scipy.signal import butter, filtfilt, find_peaks
from huggingface_hub import hf_hub_download
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.pause_events import extract_pause_events_for_clip

REPO = "pipecat-ai/smart-turn-data-v3.2-train"
DEFAULT_SHARDS = [10, 20, 30, 40, 50]
CACHE_DIR = Path("data_cache")
WAV_DIR = CACHE_DIR / "d2_stratified_wavs"
FEATURES_PATH = CACHE_DIR / "d2_stratified_signal_features.parquet"
SPLITS_PATH = CACHE_DIR / "tinyturn_splits.parquet"
PAUSE_EVENTS_PATH = CACHE_DIR / "tinyturn_pause_events.parquet"
CKPT_FEAT = CACHE_DIR / "_32k_checkpoint_features.parquet"
CKPT_SPLITS = CACHE_DIR / "_32k_checkpoint_splits.parquet"
CKPT_PAUSE = CACHE_DIR / "_32k_checkpoint_pause.parquet"
CHECKPOINT_EVERY = 1000
N_WORKERS = 14  # of 16 cores -- leave headroom for the main process/OS
TAIL_WINDOWS = [0.2, 0.3, 0.5, 1.0, 2.0]


# ---- verbatim compute_signal_features_v3 (copied from scripts/d2_decode_transcribe.py,
# not imported -- that module creates a live OpenAI client at import time, which this ASR-free
# pipeline has no business depending on). ----

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


def compute_signal_features_v3(y, sr):
    y = y.astype(np.float32)
    duration = len(y) / sr

    frame_length = int(0.032 * sr)
    hop_length = int(0.010 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    rms_db = 20 * np.log10(np.maximum(rms, 1e-6))
    rms_mean_linear = float(np.mean(rms)) if len(rms) else np.nan

    noise_floor_db = float(np.percentile(rms_db, 10))
    peak_db = float(rms_db.max())
    thresh_db = max(noise_floor_db + 10, peak_db - 40)
    active = rms_db > thresh_db
    MIN_PAUSE_S = 0.1

    active_time_s = float(active.sum() * hop_length / sr)
    voiced_time_fraction = active_time_s / duration if duration > 0 else np.nan

    runs = segment_runs(active, times)
    last_active_t = duration
    for start, end, is_act in runs[::-1]:
        if is_act:
            last_active_t = end
            break
    trailing_silence_s = max(duration - last_active_t, 0.0)

    internal_pauses = []
    if len(runs) >= 3:
        for start, end, is_act in runs[1:-1]:
            if not is_act and (end - start) >= MIN_PAUSE_S:
                internal_pauses.append(end - start)
    internal_pause_count = len(internal_pauses)
    internal_pause_total_s = float(sum(internal_pauses))
    internal_pause_max_s = float(max(internal_pauses)) if internal_pauses else 0.0

    energy_slopes = {}
    for w in TAIL_WINDOWS:
        energy_slopes[f"energy_slope_{int(w*1000)}ms"] = slope_in_window(times, rms_db, last_active_t, w)

    rel_energy = rms / rms_mean_linear if rms_mean_linear and rms_mean_linear > 0 else np.full_like(rms, np.nan)
    energy_slope_relative = slope_in_window(times, rel_energy, last_active_t, 1.0)

    f0_slopes = {f"f0_slope_{int(w*1000)}ms": np.nan for w in TAIL_WINDOWS}
    creaky_score = np.nan
    pitch_reset_after_pause = np.nan
    pitch_range_compression = np.nan
    try:
        f_len = int(0.05 * sr)
        f_hop = max(f_len // 4, 1)
        f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=70, fmax=400, sr=sr, frame_length=f_len, hop_length=f_hop)
        f0 = np.nan_to_num(f0, nan=0.0)
        f0_times = np.arange(len(f0)) * f_hop / sr
        voiced = f0 > 0

        if voiced.sum() >= 5:
            f0_ref = float(np.median(f0[voiced]))
            voiced_times = f0_times[voiced]
            voiced_semitones = 12 * np.log2(f0[voiced] / f0_ref)
            for w in TAIL_WINDOWS:
                f0_slopes[f"f0_slope_{int(w*1000)}ms"] = slope_in_window(voiced_times, voiced_semitones, last_active_t, w)

            tail_voiced_mask = (voiced_times >= max(0, last_active_t - 1.0)) & (voiced_times <= last_active_t)
            std_whole = float(np.std(voiced_semitones))
            if tail_voiced_mask.sum() >= 3 and std_whole > 0:
                std_tail = float(np.std(voiced_semitones[tail_voiced_mask]))
                pitch_range_compression = std_tail / std_whole

            if internal_pauses:
                last_pause = None
                for start, end, is_act in runs[1:-1]:
                    if not is_act and (end - start) >= MIN_PAUSE_S:
                        last_pause = (start, end)
                if last_pause:
                    p_start, p_end = last_pause
                    before_mask = voiced & (f0_times >= max(0, p_start - 0.5)) & (f0_times < p_start)
                    after_mask = voiced & (f0_times > p_end) & (f0_times <= p_end + 0.5)
                    if before_mask.sum() >= 2 and after_mask.sum() >= 2:
                        before_semi = 12 * np.log2(f0[before_mask] / f0_ref)
                        after_semi = 12 * np.log2(f0[after_mask] / f0_ref)
                        pitch_reset_after_pause = float(np.median(after_semi) - np.median(before_semi))

        tail_mask_rms = (times >= max(0, last_active_t - 0.5)) & (times <= last_active_t) & active
        if tail_mask_rms.sum() >= 3:
            tail_times = times[tail_mask_rms]
            nearest_idx = np.clip((tail_times / (f_hop / sr)).astype(int), 0, len(voiced_prob) - 1)
            low_conf = voiced_prob[nearest_idx] < 0.5
            creaky_score = float(low_conf.mean())
    except Exception:
        pass

    spectral_tilt_slope = np.nan
    try:
        n_fft = 512
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        low_energy = S[(freqs >= 80) & (freqs < 1000)].mean(axis=0)
        high_energy = S[(freqs >= 1000) & (freqs < 4000)].mean(axis=0)
        tilt_db = 20 * np.log10(np.maximum(low_energy, 1e-6)) - 20 * np.log10(np.maximum(high_energy, 1e-6))
        stft_times = librosa.frames_to_time(np.arange(len(tilt_db)), sr=sr, hop_length=hop_length)
        spectral_tilt_slope = slope_in_window(stft_times, tilt_db, last_active_t, 1.0)
    except Exception:
        pass

    breath_flatness_max = np.nan
    breath_energy_above_floor = np.nan
    try:
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
        tail_silence_mask = times > last_active_t
        if tail_silence_mask.sum() >= 2:
            breath_flatness_max = float(flatness[tail_silence_mask].max())
            breath_energy_above_floor = float(rms_db[tail_silence_mask].max() - noise_floor_db)
    except Exception:
        pass

    speaking_rate_slope = np.nan
    final_syllable_ratio = np.nan
    try:
        nyq = sr / 2
        b, a = butter(4, [300 / nyq, min(2000 / nyq, 0.99)], btype="band")
        y_band = filtfilt(b, a, y)
        from scipy.signal import hilbert
        envelope = np.abs(hilbert(y_band))
        win = max(int(0.02 * sr), 1)
        envelope_smooth = np.convolve(envelope, np.ones(win) / win, mode="same")
        active_samples = np.repeat(active, hop_length)[: len(envelope_smooth)]
        env_for_peaks = envelope_smooth.copy()
        if len(active_samples) == len(env_for_peaks):
            env_for_peaks[~active_samples] = 0
        peak_idx, _ = find_peaks(env_for_peaks, distance=int(0.1 * sr),
                                  prominence=0.1 * (env_for_peaks.max() + 1e-9))
        peak_times = peak_idx / sr
        if len(peak_times) >= 4:
            iois = np.diff(peak_times)
            speaking_rate_slope = float(np.polyfit(peak_times[1:], iois, 1)[0])
            final_syllable_ratio = float(iois[-1] / np.mean(iois)) if np.mean(iois) > 0 else np.nan
    except Exception:
        pass

    out = dict(
        duration_s=duration, last_active_t=last_active_t, trailing_silence_s=trailing_silence_s,
        voiced_time_fraction=voiced_time_fraction,
        internal_pause_count=internal_pause_count, internal_pause_total_s=internal_pause_total_s,
        internal_pause_max_s=internal_pause_max_s,
        energy_slope_relative=energy_slope_relative,
        spectral_tilt_slope=spectral_tilt_slope,
        breath_flatness_max=breath_flatness_max, breath_energy_above_floor=breath_energy_above_floor,
        creaky_score=creaky_score, pitch_reset_after_pause=pitch_reset_after_pause,
        pitch_range_compression=pitch_range_compression,
        speaking_rate_slope=speaking_rate_slope, final_syllable_ratio=final_syllable_ratio,
        rms_mean_db=float(rms_db.mean()), noise_floor_db=noise_floor_db,
        clipping_frac=float(np.mean(np.abs(y) >= 0.999)),
    )
    out.update(energy_slopes)
    out.update(f0_slopes)
    return out


def process_row(row_dict, shard_idx, row_in_shard):
    ex_id = row_dict["id"]
    audio_bytes = row_dict["audio"]["bytes"]
    data, sr = sf.read(io.BytesIO(audio_bytes))
    channels = 1 if data.ndim == 1 else data.shape[1]
    y = data if data.ndim == 1 else data.mean(axis=1)
    sf.write(WAV_DIR / f"{ex_id}.wav", data, sr)
    feats = compute_signal_features_v3(y.astype(np.float32), sr)

    feat_row = {
        "id": ex_id, "language": row_dict["language"], "endpoint_bool": bool(row_dict["endpoint_bool"]),
        "midfiller": row_dict["midfiller"], "endfiller": row_dict["endfiller"],
        "synthetic": bool(row_dict["synthetic"]), "spoken_text": row_dict.get("spoken_text"),
        "dataset": row_dict["dataset"], "shard": shard_idx, "row_in_shard": row_in_shard,
        "duration_bin": None, "_strat_key": None, "sr": sr, "channels": channels, **feats,
    }
    events = extract_pause_events_for_clip(ex_id, y.astype(np.float32), sr)
    split_row = {
        "id": ex_id, "split": "train", "dataset": row_dict["dataset"], "language": row_dict["language"],
        "synthetic": bool(row_dict["synthetic"]), "endpoint_bool": bool(row_dict["endpoint_bool"]),
        "group": f"singleton_{ex_id}", "last_active_B_energy_alt": np.nan, "last_active_C_silero_vad": np.nan,
    }
    return feat_row, split_row, events


def _checkpoint(all_feat_rows, all_split_rows, all_pause_rows):
    pd.DataFrame(all_feat_rows).to_parquet(CKPT_FEAT, index=False)
    pd.DataFrame(all_split_rows).to_parquet(CKPT_SPLITS, index=False)
    pd.DataFrame(all_pause_rows).to_parquet(CKPT_PAUSE, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-shard", type=int, default=None)
    ap.add_argument("--shards", type=str, default=None,
                     help="comma-separated shard indices, e.g. 5,15,25 (default: the original 32k-tier set)")
    args = ap.parse_args()
    shards = [int(s) for s in args.shards.split(",")] if args.shards else DEFAULT_SHARDS

    existing_ids = set(pd.read_parquet(SPLITS_PATH)["id"])
    print(f"existing D2 ids: {len(existing_ids):,}", flush=True)

    # Resume from a prior interrupted run's checkpoint, if present -- process-pool work is
    # expensive enough (pyin per clip) that losing it to a crash/interrupt would be wasteful.
    all_feat_rows, all_split_rows, all_pause_rows = [], [], []
    if CKPT_FEAT.exists():
        all_feat_rows = pd.read_parquet(CKPT_FEAT).to_dict(orient="records")
        all_split_rows = pd.read_parquet(CKPT_SPLITS).to_dict(orient="records")
        all_pause_rows = pd.read_parquet(CKPT_PAUSE).to_dict(orient="records")
        existing_ids |= {r["id"] for r in all_feat_rows}
        print(f"resuming from checkpoint: {len(all_feat_rows):,} clips already done this run", flush=True)

    t0 = time.time()
    since_checkpoint = 0
    for shard_idx in shards:
        fname = f"data/train-{shard_idx:05d}-of-00083.parquet"
        print(f"fetching shard {shard_idx} ({fname})...", flush=True)
        path = hf_hub_download(REPO, fname, repo_type="dataset")
        table = pq.read_table(path)
        df = table.to_pandas()
        df = df[~df["id"].isin(existing_ids)].reset_index(drop=True)
        if args.limit_per_shard:
            df = df.head(args.limit_per_shard)
        print(f"  shard {shard_idx}: {len(df):,} new rows (of {table.num_rows:,} total) to process "
              f"with {N_WORKERS} worker processes", flush=True)

        n_done = 0
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = [pool.submit(process_row, row.to_dict(), shard_idx, i) for i, row in df.iterrows()]
            for fut in as_completed(futures):
                feat_row, split_row, events = fut.result()
                all_feat_rows.append(feat_row)
                all_split_rows.append(split_row)
                all_pause_rows.extend(events)
                existing_ids.add(feat_row["id"])
                n_done += 1
                since_checkpoint += 1
                if n_done % 500 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{shard_idx}] {n_done}/{len(df)} ({elapsed:.0f}s elapsed total)", flush=True)
                if since_checkpoint >= CHECKPOINT_EVERY:
                    _checkpoint(all_feat_rows, all_split_rows, all_pause_rows)
                    since_checkpoint = 0
                    print(f"  [checkpoint] {len(all_feat_rows):,} clips saved to staging", flush=True)
        print(f"  shard {shard_idx} done ({len(df):,} rows) at {time.time()-t0:.0f}s total elapsed", flush=True)

    _checkpoint(all_feat_rows, all_split_rows, all_pause_rows)
    print(f"\ntotal new clips: {len(all_feat_rows):,}", flush=True)

    old_feat = pd.read_parquet(FEATURES_PATH)
    new_feat = pd.DataFrame(all_feat_rows)
    combined_feat = pd.concat([old_feat, new_feat], ignore_index=True)
    combined_feat.to_parquet(FEATURES_PATH, index=False)
    print(f"saved {FEATURES_PATH}: {len(old_feat):,} -> {len(combined_feat):,} rows", flush=True)

    old_splits = pd.read_parquet(SPLITS_PATH)
    new_splits = pd.DataFrame(all_split_rows)
    combined_splits = pd.concat([old_splits, new_splits], ignore_index=True)
    combined_splits.to_parquet(SPLITS_PATH, index=False)
    print(f"saved {SPLITS_PATH}: {len(old_splits):,} -> {len(combined_splits):,} rows "
          f"(split counts: {combined_splits['split'].value_counts().to_dict()})", flush=True)

    old_pause = pd.read_parquet(PAUSE_EVENTS_PATH)
    new_pause = pd.DataFrame(all_pause_rows)
    combined_pause = pd.concat([old_pause, new_pause], ignore_index=True)
    combined_pause.to_parquet(PAUSE_EVENTS_PATH, index=False)
    print(f"saved {PAUSE_EVENTS_PATH}: {len(old_pause):,} -> {len(combined_pause):,} rows", flush=True)

    for p in (CKPT_FEAT, CKPT_SPLITS, CKPT_PAUSE):
        p.unlink(missing_ok=True)

    print(f"\nDONE in {time.time()-t0:.0f}s total.")


if __name__ == "__main__":
    main()
