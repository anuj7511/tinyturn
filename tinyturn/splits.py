"""
Step 1 -- train / model-selection-validation / threshold-calibration split over the D2 stratified
cache (15,998 real+synthetic clips with decoded audio in data_cache/d2_stratified_wavs/).

No split existed anywhere in the repo before this (confirmed against eda_part3_report.md, which
flags this as still open). The **official** test set is a separate, untouched HF repo
(`pipecat-ai/smart-turn-data-v3.2-test`, 31,527 rows, hashed but not cached locally) -- per the
brief's Section 8 discipline it is touched once per finalist, not during this stepwise development,
so it plays no part in this module.

Design:
  - group key = D11 voice_cluster where available (206/15,998 rows overlap voice_cluster's 3,000-id
    subsample -- limited coverage, but real leakage protection is free where it exists), else each
    id is its own singleton group -- so no group ever spans a split boundary.
  - stratified (approximately) by dataset x endpoint_bool via StratifiedGroupKFold, so every split
    keeps roughly the source/label composition of the whole cache.
  - 80% train / 10% model-selection val / 10% threshold-calibration.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RNG_SEED = 42
CACHE_DIR = Path("data_cache")
OUT_PATH = CACHE_DIR / "tinyturn_splits.parquet"


def _load_alt_boundaries() -> pd.DataFrame:
    """E5's per-clip alternative boundary estimates (A: fixed energy threshold -- our canonical
    v0 -- vs. B: alt energy threshold vs. C: Silero VAD), on its own 3,000-clip sample (206 of
    which overlap the D2 cache used here). Kept alongside the splits/metadata, unused by Steps 1-5,
    specifically so Step 9's boundary augmentation doesn't need to recompute Silero VAD from
    scratch for clips it's already available for."""
    path = Path("eda_outputs") / "tables" / "e5_vad_sensitivity_results.csv"
    if not path.exists():
        return pd.DataFrame(columns=["id", "last_active_B_energy_alt", "last_active_C_silero_vad"])
    v = pd.read_csv(path)
    return v[["id", "last_active_B_energy_alt", "last_active_C_silero_vad"]]


def build_splits(seed: int = RNG_SEED) -> pd.DataFrame:
    sf = pd.read_parquet(CACHE_DIR / "d2_stratified_signal_features.parquet")
    sf = sf[sf["sr"].notna() & sf["last_active_t"].notna()].reset_index(drop=True)

    vc = pd.read_parquet(CACHE_DIR / "d11_voice_clusters.parquet")[["id", "voice_cluster"]]
    df = sf.merge(vc, on="id", how="left")
    df = df.merge(_load_alt_boundaries(), on="id", how="left")
    df["group"] = df["voice_cluster"].apply(
        lambda c: f"cluster_{int(c)}" if pd.notna(c) else None
    )
    missing = df["group"].isna()
    df.loc[missing, "group"] = "singleton_" + df.loc[missing, "id"]

    df["strata"] = df["dataset"].astype(str) + "|" + df["endpoint_bool"].astype(str)

    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1, dtype=int)
    for fold_id, (_, test_idx) in enumerate(sgkf.split(df, df["strata"], df["group"])):
        fold[test_idx] = fold_id
    df["_fold"] = fold

    split = np.where(df["_fold"] == 0, "val",
             np.where(df["_fold"] == 1, "calib", "train"))
    df["split"] = split

    out = df[["id", "split", "dataset", "language", "synthetic", "endpoint_bool", "group",
              "last_active_B_energy_alt", "last_active_C_silero_vad"]].copy()
    return out


def load_splits() -> pd.DataFrame:
    if not OUT_PATH.exists():
        raise FileNotFoundError(f"{OUT_PATH} not found -- run `python -m tinyturn.splits` first")
    return pd.read_parquet(OUT_PATH)


def _assert_no_group_leakage(df: pd.DataFrame):
    g = df.groupby("group")["split"].nunique()
    leaked = g[g > 1]
    assert len(leaked) == 0, f"{len(leaked)} groups span multiple splits: {leaked.index.tolist()[:5]}"


def main():
    df = build_splits()
    _assert_no_group_leakage(df)
    counts = df["split"].value_counts()
    print("split sizes:\n", counts)
    print("\nendpoint_bool balance by split:\n", df.groupby("split")["endpoint_bool"].mean())
    print("\nsynthetic balance by split:\n", df.groupby("split")["synthetic"].mean())
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nsaved {OUT_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
