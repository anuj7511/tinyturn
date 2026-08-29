"""
Step 7 precompute -- extract internal-pause event spans (>= 200ms, up to 2 longest per clip) for
every D2-cache clip whose cached `internal_pause_max_s` suggests it might have one, using the
canonical boundary estimator on the actual waveform (the cached aggregate stats don't include exact
span positions). Saves data_cache/tinyturn_pause_events.parquet.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import soundfile as sf

from tinyturn.pause_events import extract_pause_events_for_clip, EVENTS_PATH, MIN_PAUSE_S
from tinyturn.dataset import WAV_DIR

CACHE_DIR = Path("data_cache")


def main():
    sf_feat = pd.read_parquet(CACHE_DIR / "d2_stratified_signal_features.parquet")
    sf_feat = sf_feat[sf_feat["sr"].notna() & sf_feat["last_active_t"].notna()]
    candidates = sf_feat[sf_feat["internal_pause_max_s"] >= MIN_PAUSE_S]
    print(f"scanning {len(candidates)} candidate clips (of {len(sf_feat)} total) for pause events "
          f">= {MIN_PAUSE_S}s", flush=True)

    rows = []
    t0 = time.time()
    n_ok, n_err = 0, 0
    for i, (_, r) in enumerate(candidates.iterrows()):
        try:
            data, sr = sf.read(WAV_DIR / f"{r['id']}.wav")
        except Exception:
            n_err += 1
            continue
        y = data if data.ndim == 1 else data.mean(axis=1)
        try:
            events = extract_pause_events_for_clip(r["id"], y, sr)
        except Exception:
            n_err += 1
            continue
        rows.extend(events)
        n_ok += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(candidates)} clips scanned, {len(rows)} events so far, "
                  f"{time.time()-t0:.1f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(EVENTS_PATH, index=False)
    print(f"\nscanned {n_ok} clips ok, {n_err} errors, produced {len(out)} pause events "
          f"in {time.time()-t0:.1f}s", flush=True)
    print(out.groupby("event_idx")["pause_duration_s"].describe())
    print(f"saved {EVENTS_PATH}")


if __name__ == "__main__":
    main()
