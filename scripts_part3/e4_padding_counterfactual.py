"""
E4 -- padding counterfactual. Reuses D6's fast feature extraction (compute_probe_features_fast)
and the same 8 PROBE_FEATURES / same simple logistic-regression probe family. Trains ONE fixed
model on the 8s-context features (all of D2's sample, matching D6's longest window), then applies
that fixed model to four different padding schemes of a held-out set of short clips, comparing
predicted probability across schemes for the same underlying speech.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.preprocessing import StandardScaler as Scaler

import sys
sys.path.insert(0, str(Path(__file__).parent))
from d6_context_probe import compute_probe_features_fast, PROBE_FEATURES

RNG_SEED = 42
CACHE_DIR = Path("data_cache")
WAV_DIR = CACHE_DIR / "d2_stratified_wavs"
TARGET_LEN_S = 8.0
N_TRAIN = 5000
N_TEST_SHORT_CLIPS = 400


def build_padded_variant(y, sr, target_len_s, scheme, rng):
    target_n = int(target_len_s * sr)
    if len(y) >= target_n:
        return y[-target_n:]  # already long enough -- truncate to tail, same as D6/reference conv.
    pad_n = target_n - len(y)
    if scheme == "left_zero":
        return np.concatenate([np.zeros(pad_n, dtype=y.dtype), y])
    elif scheme == "noise":
        noise_level = np.std(y) * 0.02 + 1e-5
        pad = rng.normal(0, noise_level, pad_n).astype(y.dtype)
        return np.concatenate([pad, y])
    elif scheme == "repeat":
        reps = int(np.ceil(target_n / len(y)))
        tiled = np.tile(y, reps)
        return tiled[-target_n:]
    elif scheme == "unpadded":
        return y  # natural length, no padding -- feature extraction handles variable length fine
    raise ValueError(scheme)


def main():
    feat_full = pd.read_parquet(CACHE_DIR / "d2_stratified_signal_features.parquet")
    feat_full = feat_full[feat_full["sr"].notna()]

    rng = np.random.RandomState(RNG_SEED)
    train_sample = feat_full.sample(n=min(N_TRAIN, len(feat_full)), random_state=RNG_SEED)
    print(f"E4: training fixed 8s probe on n={len(train_sample)}", flush=True)

    t0 = time.time()
    train_rows = []
    for i, (_, r) in enumerate(train_sample.iterrows()):
        try:
            data, sr = sf.read(WAV_DIR / f"{r['id']}.wav")
        except Exception:
            continue
        y = data if data.ndim == 1 else data.mean(axis=1)
        n_samp = int(TARGET_LEN_S * sr)
        y_ctx = y[-n_samp:] if len(y) > n_samp else y
        feats = compute_probe_features_fast(y_ctx, sr)
        train_rows.append({"endpoint_bool": r["endpoint_bool"], **feats})
        if (i + 1) % 2000 == 0:
            print(f"  train feat {i+1}/{len(train_sample)}, {time.time()-t0:.1f}s", flush=True)

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

    # held-out short clips (duration < 8s, so padding is actually needed), excluded from training
    train_ids = set(train_sample["id"])
    short_pool = feat_full[(~feat_full["id"].isin(train_ids)) & (feat_full["duration_s"] < 6.0)]
    test_sample = short_pool.sample(n=min(N_TEST_SHORT_CLIPS, len(short_pool)), random_state=RNG_SEED)
    print(f"padding counterfactual on {len(test_sample)} short (<6s) held-out clips", flush=True)

    schemes = ["left_zero", "noise", "repeat", "unpadded"]
    rows = []
    t1 = time.time()
    for i, (_, r) in enumerate(test_sample.iterrows()):
        try:
            data, sr = sf.read(WAV_DIR / f"{r['id']}.wav")
        except Exception:
            continue
        y = data if data.ndim == 1 else data.mean(axis=1)
        probs = {}
        for scheme in schemes:
            variant = build_padded_variant(y, sr, TARGET_LEN_S, scheme, rng)
            feats = compute_probe_features_fast(variant, sr)
            x = np.array([[feats[f] for f in PROBE_FEATURES]], dtype=float)
            for col in missing_cols:
                if np.isnan(x[0, col]):
                    med = np.nanmedian(X[:, col])
                    x[0, col] = med if med == med else 0.0
            prob = clf.predict_proba(scaler.transform(x))[0, 1]
            probs[f"prob_{scheme}"] = prob
        variance = float(np.var(list(probs.values())))
        max_abs_diff = float(max(probs.values()) - min(probs.values()))
        rows.append({"id": r["id"], "duration_s": r["duration_s"], "endpoint_bool": r["endpoint_bool"],
                     **probs, "prob_variance_across_schemes": variance, "max_abs_diff": max_abs_diff})
        if (i + 1) % 100 == 0:
            print(f"  counterfactual {i+1}/{len(test_sample)}, {time.time()-t1:.1f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("eda_outputs/tables/e4_padding_counterfactual_results.csv", index=False)
    print(f"\nDONE: {len(out)} clips, {time.time()-t1:.1f}s")
    print(f"mean prob variance across schemes: {out['prob_variance_across_schemes'].mean():.5f}")
    print(f"mean max abs diff across schemes: {out['max_abs_diff'].mean():.4f}")
    print(f"fraction of clips with max_abs_diff > 0.2 (prediction flips a lot depending on scheme): "
          f"{(out['max_abs_diff'] > 0.2).mean()*100:.1f}%")
    print(out[[c for c in out.columns if c.startswith('prob_')] + ['max_abs_diff']].describe().to_string())


if __name__ == "__main__":
    main()
