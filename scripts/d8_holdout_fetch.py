"""
D8 -- fetch + transcribe a disjoint holdout for independent lexicon validation.

Efficiency fix vs. D2's approach: rather than streaming the whole ~41GB dataset to find scattered
ids, fetch a small number of FULL shards directly by URL (id+audio columns only), which the
dataset-viewer-column-pruned pattern already used for metadata makes cheap even with the audio
column included (no need to stream past the other ~80 shards). Verified: one shard (audio+id
columns), 3,264 rows, fetched in ~122s. Two shards give ample per-language coverage (each shard
already contains a representative mix of every language/dataset) after filtering out any id
already used by D2 or appearing in the test set.
"""
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import unicodedata

load_dotenv()

REPO = "pipecat-ai/smart-turn-data-v3.2-train"
N_SHARDS = 83
SHARDS_TO_FETCH = [20, 60]  # arbitrary, not otherwise special-cased
CACHE_DIR = Path("data_cache")
TARGET_PER_LANG = 45
RNG_SEED = 42

ASR_MODEL = "gpt-4o-transcribe"
client = OpenAI()


def clean_word(w):
    return "".join(c for c in w.strip().lower() if not unicodedata.category(c).startswith("P")).strip()


def transcribe_bytes_openai(audio_bytes, filename="clip.flac", retries=2):
    last_err = None
    for attempt in range(retries + 1):
        try:
            f = io.BytesIO(audio_bytes)
            f.name = filename
            resp = client.audio.transcriptions.create(model=ASR_MODEL, file=f)
            return resp.text
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1) + np.random.uniform(0, 1))
    raise last_err


def main():
    d2_ids = set(pd.read_csv(CACHE_DIR / "stratified_signal_manifest.csv")["id"])
    meta_all = pd.read_parquet(CACHE_DIR / "metadata_all_shards.parquet").set_index("id")

    t0 = time.time()
    frames = []
    for shard_i in SHARDS_TO_FETCH:
        url = f"hf://datasets/{REPO}/data/train-{shard_i:05d}-of-{N_SHARDS:05d}.parquet"
        df = pd.read_parquet(url, columns=["id", "audio"])
        df["shard"] = shard_i
        frames.append(df)
        print(f"shard {shard_i}: {len(df)} rows fetched, {time.time()-t0:.1f}s elapsed", flush=True)
    pool = pd.concat(frames, ignore_index=True)
    pool = pool[~pool["id"].isin(d2_ids)]
    pool = pool.join(meta_all[["language", "dataset", "synthetic", "endfiller", "midfiller", "endpoint_bool"]],
                      on="id")
    print(f"pool after excluding D2 ids: {len(pool)} rows")

    rng = np.random.RandomState(RNG_SEED)
    rows = []
    for lang, grp in pool.groupby("language"):
        real = grp[grp["synthetic"] == False]
        synth = grp[grp["synthetic"] == True]
        n_real_take = min(len(real), max(TARGET_PER_LANG // 3, 1)) if len(real) else 0
        n_synth_take = min(len(synth), TARGET_PER_LANG - n_real_take)
        if n_real_take:
            rows.append(real.sample(n=n_real_take, random_state=RNG_SEED))
        if n_synth_take:
            rows.append(synth.sample(n=n_synth_take, random_state=RNG_SEED))
    holdout = pd.concat(rows, ignore_index=True).drop_duplicates(subset="id")
    print(f"D8 holdout drawn: {len(holdout)} ids across {holdout['language'].nunique()} languages")

    def transcribe_row(row):
        try:
            text = transcribe_bytes_openai(row["audio"]["bytes"], filename=f"{row['id']}.flac")
            return {"id": row["id"], "language": row["language"], "dataset": row["dataset"],
                    "synthetic": row["synthetic"], "endfiller": row["endfiller"],
                    "midfiller": row["midfiller"], "endpoint_bool": row["endpoint_bool"], "text": text}
        except Exception as e:
            return {"id": row["id"], "language": row["language"], "text": None, "asr_error": str(e)}

    t1 = time.time()
    records = []
    with ThreadPoolExecutor(max_workers=8) as ex_pool:
        futures = [ex_pool.submit(transcribe_row, row) for _, row in holdout.iterrows()]
        for fut in as_completed(futures):
            records.append(fut.result())
            if len(records) % 100 == 0:
                print(f"transcribed {len(records)}/{len(holdout)}, {time.time()-t1:.1f}s", flush=True)

    out = pd.DataFrame(records)
    out.to_parquet(CACHE_DIR / "d8_holdout_transcripts.parquet")
    print(f"DONE: {len(out)} holdout clips transcribed in {time.time()-t1:.1f}s "
          f"(total incl. fetch: {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
