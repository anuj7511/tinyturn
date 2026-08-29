"""Integration tests for TinyTurnDataset against the real D2 cache -- boundary_source parity,
disk-cache correctness, and mono/16kHz enforcement."""
import numpy as np
import pytest

from tinyturn.dataset import TinyTurnDataset, FEATURE_CACHE_DIR
from tinyturn.boundary import estimate_speech_end


def _tiny_val_dataset(**kwargs):
    ds = TinyTurnDataset(split="val", context_s=1.0, **kwargs)
    ds.df = ds.df.iloc[:8].reset_index(drop=True)
    return ds


def test_boundary_source_recompute_matches_cached_closely():
    """`recompute` re-derives the boundary with tinyturn.boundary.estimate_speech_end -- same
    formula as the cached D2 value, so they should agree almost exactly (float rounding only)."""
    ds_cached = _tiny_val_dataset(boundary_source="cached", use_disk_cache=False)
    ds_recompute = _tiny_val_dataset(boundary_source="recompute", use_disk_cache=False)
    for i in range(len(ds_cached)):
        row = ds_cached.df.iloc[i]
        y, sr = ds_cached._load_wav(row["id"])
        cached_val = float(row["last_active_t"])
        recomputed_val = estimate_speech_end(y, sr).speech_end_s
        assert abs(cached_val - recomputed_val) < 0.02, (row["id"], cached_val, recomputed_val)


def test_disk_cache_roundtrip_matches_uncached(tmp_path, monkeypatch):
    import tinyturn.dataset as dmod
    cache_dir = tmp_path / "feature_cache"
    monkeypatch.setattr(dmod, "FEATURE_CACHE_DIR", cache_dir)

    ds_nocache = _tiny_val_dataset(include_trajectory=True, use_disk_cache=False)
    ds_cache1 = _tiny_val_dataset(include_trajectory=True, use_disk_cache=True)
    ds_cache2 = _tiny_val_dataset(include_trajectory=True, use_disk_cache=True)

    item_nocache = ds_nocache[0]
    item_cache_cold = ds_cache1[0]   # writes cache
    assert any(cache_dir.glob("*.npz")), "expected cache files to be written"
    item_cache_warm = ds_cache2[0]   # reads cache

    np.testing.assert_allclose(item_nocache["log_mel"].numpy(), item_cache_cold["log_mel"].numpy())
    np.testing.assert_allclose(item_cache_cold["log_mel"].numpy(), item_cache_warm["log_mel"].numpy())
    np.testing.assert_allclose(item_nocache["trajectory"].numpy(), item_cache_warm["trajectory"].numpy())


def test_include_f0_appends_extra_channel():
    ds_no_f0 = _tiny_val_dataset(include_trajectory=True, include_f0=False, use_disk_cache=False)
    ds_f0 = _tiny_val_dataset(include_trajectory=True, include_f0=True, use_disk_cache=False)
    item_no_f0 = ds_no_f0[0]
    item_f0 = ds_f0[0]
    assert item_no_f0["trajectory"].shape[-1] == 5
    assert item_f0["trajectory"].shape[-1] == 6
    np.testing.assert_allclose(item_no_f0["trajectory"].numpy(), item_f0["trajectory"].numpy()[:, :5])


def test_endfiller_resolved_prefers_ground_truth_for_synthetic():
    ds = TinyTurnDataset(split="val", context_s=1.0, use_disk_cache=False)
    synth_rows = ds.df[ds.df["synthetic"]]
    assert len(synth_rows) > 0
    # ground-truth `endfiller` is defined (non-null) for at least most synthetic rows, and where it
    # is, endfiller_resolved must equal it exactly (not silently overridden by the ASR-derived guess)
    has_gt = synth_rows["endfiller"].notna()
    assert has_gt.mean() > 0.9
    pd_eq = (synth_rows.loc[has_gt, "endfiller_resolved"] == synth_rows.loc[has_gt, "endfiller"])
    assert pd_eq.all()
