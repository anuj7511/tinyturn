"""
D1 (part 1) -- fetch the official smart-turn-data-v3.2-test set: metadata + audio hashes/fingerprints
for every row, so it can be compared against train's already-hashed clips.

Unlike D2, there's no scattered-id problem here -- we want every test row, so this is a plain full
streaming pass over the (much smaller, ~4.84GB / 31,527-row) test repo. Runs at reduced concurrency
(no ASR calls needed) so it competes as little as possible with the D2 job's network usage.
"""
import io
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import hashlib

warnings.filterwarnings("ignore")

TEST_REPO = "pipecat-ai/smart-turn-data-v3.2-test"
CACHE_DIR = Path("data_cache")
OUT = CACHE_DIR / "d1_test_hashes_fingerprints.parquet"
CHECKPOINT_EVERY = 2000


def waveform_hash(data, sr, target_sr=16000):
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    int16_data = np.clip(np.round(data * 32767), -32768, 32767).astype(np.int16)
    return hashlib.sha256(int16_data.tobytes()).hexdigest()


def mfcc_fingerprint(data, sr):
    if data.ndim > 1:
        data = data.mean(axis=1)
    mfcc = librosa.feature.mfcc(y=data.astype(np.float32), sr=sr, n_mfcc=20)
    vec = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def main():
    records = []
    fps = []
    done_ids = set()
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        records = prev.drop(columns=["mfcc_fp"]).to_dict(orient="records")
        fps = list(np.stack(prev["mfcc_fp"].values))
        done_ids = set(prev["id"])
        print(f"resuming: {len(done_ids)} test rows already hashed")

    from datasets import load_dataset, Audio
    ds = load_dataset(TEST_REPO, split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    t0 = time.time()
    n_seen = 0
    for ex in ds:
        n_seen += 1
        if ex["id"] in done_ids:
            continue
        try:
            data, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]))
            wh = waveform_hash(data, sr)
            fp = mfcc_fingerprint(data, sr)
            records.append({
                "id": ex["id"], "language": ex["language"], "endpoint_bool": ex["endpoint_bool"],
                "midfiller": ex["midfiller"], "endfiller": ex["endfiller"], "synthetic": ex["synthetic"],
                "dataset": ex["dataset"], "waveform_sha256": wh,
                "duration_s": len(data) / sr,
            })
            fps.append(fp)
        except Exception as e:
            records.append({"id": ex["id"], "language": ex.get("language"), "decode_error": str(e)})
            fps.append(np.full(40, np.nan))

        if n_seen % CHECKPOINT_EVERY == 0:
            out_df = pd.DataFrame(records)
            out_df["mfcc_fp"] = list(fps)
            out_df.to_parquet(OUT)
            elapsed = time.time() - t0
            print(f"[checkpoint] {n_seen:,} rows processed, {elapsed/60:.1f}min elapsed, "
                  f"rate={n_seen/elapsed:.1f} rows/s", flush=True)

    out_df = pd.DataFrame(records)
    out_df["mfcc_fp"] = list(fps)
    out_df.to_parquet(OUT)
    print(f"DONE: {len(out_df):,} test rows hashed+fingerprinted in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
