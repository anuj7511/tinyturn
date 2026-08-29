"""
D2 -- decode + transcribe the 15,998-row stratified sample (data_cache/stratified_signal_manifest.csv).

Single streaming pass through the full pipecat-ai/smart-turn-data-v3.2-train dataset (all 83
shards / 270,946 rows), because the stratified ids are scattered across every shard (confirmed:
m["shard"].nunique() == 83) -- there is no cheaper way to fetch just these 15,998 rows' audio
bytes than to stream past all of them. Measured steady-state throughput ~23-24 rows/s once warmed
up => full pass is ~3-3.5 hours, not the "decode is incidental" framing in the brief (which did
not anticipate that a stratified id set touches every shard). ASR transcription runs concurrently
in a thread pool as matches are found, so it does not add meaningfully to the wall-clock time.

Resumable: skips ids already present in the checkpoint parquet files on restart.
"""
import io
import json
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings("ignore")

REPO = "pipecat-ai/smart-turn-data-v3.2-train"
CACHE_DIR = Path("data_cache")
WAV_DIR = CACHE_DIR / "d2_stratified_wavs"
WAV_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_OUT = CACHE_DIR / "d2_stratified_signal_features.parquet"
TRANSCRIPTS_OUT = CACHE_DIR / "d2_stratified_transcripts.parquet"
PROGRESS_LOG = CACHE_DIR / "d2_decode_progress.json"

CHECKPOINT_EVERY = 500  # matches, not stream rows

TAIL_WINDOWS = [0.2, 0.3, 0.5, 1.0, 2.0]


# ---- ASR setup (same as cell 32 in eda.ipynb) ----
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

ASR_MODEL = "gpt-4o-transcribe"
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found in environment/.env")
client = OpenAI()


def transcribe_bytes_openai(audio_bytes: bytes, filename: str = "clip.flac", retries: int = 2) -> str:
    last_err = None
    for attempt in range(retries + 1):
        try:
            file_obj = io.BytesIO(audio_bytes)
            file_obj.name = filename
            resp = client.audio.transcriptions.create(model=ASR_MODEL, file=file_obj)
            return resp.text
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1) + np.random.uniform(0, 1))  # jitter added for this ~40x volume run
    raise last_err


import unicodedata

def clean_word(w: str) -> str:
    return "".join(c for c in w.strip().lower() if not unicodedata.category(c).startswith("P")).strip()


FILLER_TOKENS = {
    "um", "umm", "uh", "uhh", "er", "erm", "hmm", "hm", "like",
    "actually", "basically", "and", "but", "so", "because",
    "na", "yaar", "yani", "toh", "aur", "lekin", "par", "ki", "matlab", "kyonki",
    "हं", "अरे", "मतलब",
}


def derive_filler_labels_from_text(text: str, n_tail_words: int = 3) -> dict:
    tokens = [t for t in (clean_word(w) for w in (text or "").strip().split()) if t]
    if not tokens:
        return {"endfiller_derived": False, "midfiller_derived": False, "n_words": 0}
    tail, head = tokens[-n_tail_words:], tokens[:-n_tail_words]
    return {
        "endfiller_derived": any(t in FILLER_TOKENS for t in tail),
        "midfiller_derived": any(t in FILLER_TOKENS for t in head),
        "n_words": len(tokens),
    }


# ---- DSP feature extraction (verbatim compute_signal_features_v3 from eda.ipynb cell 20) ----

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


def process_match(ex_id, audio_bytes, meta_row):
    result = {"id": ex_id, **meta_row}
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        channels = 1 if data.ndim == 1 else data.shape[1]
        y = data if data.ndim == 1 else data.mean(axis=1)
        sf.write(WAV_DIR / f"{ex_id}.wav", data, sr)
        feats = compute_signal_features_v3(y, sr)
        feat_row = {**result, "sr": sr, "channels": channels, **feats}
    except Exception as e:
        feat_row = {**result, "sr": None, "channels": None, "decode_error": str(e)}

    try:
        text = transcribe_bytes_openai(audio_bytes, filename=f"{ex_id}.flac")
        derived = derive_filler_labels_from_text(text)
        transcript_row = {"id": ex_id, "text": text, **derived}
    except Exception as e:
        transcript_row = {"id": ex_id, "text": None, "endfiller_derived": None,
                           "midfiller_derived": None, "n_words": None, "asr_error": str(e)}

    return feat_row, transcript_row


def main():
    manifest = pd.read_csv(CACHE_DIR / "stratified_signal_manifest.csv")
    target_ids = set(manifest["id"])
    meta_by_id = manifest.set_index("id").to_dict(orient="index")
    print(f"target: {len(target_ids):,} stratified ids across all 83 shards")

    done_ids = set()
    feat_records, transcript_records = [], []
    if FEATURES_OUT.exists():
        prev = pd.read_parquet(FEATURES_OUT)
        feat_records = prev.to_dict(orient="records")
        done_ids |= set(prev["id"])
    if TRANSCRIPTS_OUT.exists():
        prevt = pd.read_parquet(TRANSCRIPTS_OUT)
        transcript_records = prevt.to_dict(orient="records")
        done_ids &= set(prevt["id"]) if done_ids else set(prevt["id"])

    remaining = target_ids - done_ids
    print(f"already done: {len(done_ids):,}  |  remaining: {len(remaining):,}")
    if not remaining:
        print("nothing left to do.")
        return

    from datasets import load_dataset, Audio
    ds = load_dataset(REPO, split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    t0 = time.time()
    n_seen = 0
    n_matched = 0
    pending = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for ex in ds:
            n_seen += 1
            if ex["id"] in remaining:
                fut = pool.submit(process_match, ex["id"], ex["audio"]["bytes"], meta_by_id[ex["id"]])
                futures.append(fut)
                n_matched += 1

            if n_seen % 20000 == 0:
                elapsed = time.time() - t0
                print(f"[stream] seen={n_seen:,} matched={n_matched:,}/{len(remaining):,} "
                      f"elapsed={elapsed/60:.1f}min rate={n_seen/elapsed:.1f} rows/s", flush=True)

            # drain completed futures periodically, checkpoint every CHECKPOINT_EVERY matches
            if len(futures) >= CHECKPOINT_EVERY or n_matched >= len(remaining):
                for fut in futures:
                    feat_row, transcript_row = fut.result()
                    feat_records.append(feat_row)
                    transcript_records.append(transcript_row)
                futures = []
                pd.DataFrame(feat_records).to_parquet(FEATURES_OUT)
                pd.DataFrame(transcript_records).to_parquet(TRANSCRIPTS_OUT)
                elapsed = time.time() - t0
                print(f"[checkpoint] {len(feat_records):,} total done, "
                      f"{elapsed/60:.1f}min elapsed, {n_seen:,} stream rows seen", flush=True)
                PROGRESS_LOG.write_text(json.dumps({
                    "n_done": len(feat_records), "n_target": len(target_ids),
                    "n_stream_seen": n_seen, "elapsed_s": elapsed,
                }))

            if len(done_ids) + n_matched >= len(target_ids) and not futures:
                break

        # drain any stragglers
        for fut in futures:
            feat_row, transcript_row = fut.result()
            feat_records.append(feat_row)
            transcript_records.append(transcript_row)

    pd.DataFrame(feat_records).to_parquet(FEATURES_OUT)
    pd.DataFrame(transcript_records).to_parquet(TRANSCRIPTS_OUT)
    elapsed = time.time() - t0
    print(f"DONE: {len(feat_records):,} clips decoded+featurized+transcribed in {elapsed/60:.1f} min "
          f"(streamed {n_seen:,} rows total)")


if __name__ == "__main__":
    main()
