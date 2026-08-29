"""Dataset wrapper producing Whisper-convention log-mel features (via WhisperFeatureExtractor) from
the same build_example windowed waveform used by B0/B1 -- identical context/masks/splits (Phase-2
8b: speech-aligned, no baked-in post-roll), per the brief's Section 5 requirement that A0 differ
from B0/B1 only in encoder choice."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor

from tinyturn.preprocess import build_example, frame_valid_mask
from tinyturn.whisper_model import WHISPER_MODEL_NAME

CACHE_DIR = Path("data_cache")
WAV_DIR = CACHE_DIR / "d2_stratified_wavs"
FRAME_LENGTH_S = 0.025
HOP_LENGTH_S = 0.010
TRAIN_BOUNDARY_AUGMENTATION_PATH = CACHE_DIR / "d2_train_boundary_augmentation.parquet"
BOUNDARY_AUGMENTATION_COLUMNS = ["canonical_boundary_s", "alt_threshold_boundary_s", "silero_boundary_s"]


def worker_init_fn(worker_id):
    """Reseed both `random` and `numpy.random` per DataLoader worker. Without this, workers spawned
    via fork (the default on Linux, incl. Kaggle) inherit an identical RNG state from the parent
    process, so boundary-augmentation draws would be correlated in lockstep across workers instead
    of independent. Derives each worker's seed from `torch.initial_seed()`, which PyTorch already
    varies per worker, so this doesn't need its own base-seed plumbing."""
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def silence_invalid_samples(waveform: np.ndarray, valid_sample_mask: np.ndarray) -> np.ndarray:
    """Force the invalid (padded) region of a build_example waveform to exact silence before
    feature extraction (Phase-2 8c, extended). Necessary because WhisperFeatureExtractor's own
    per-utterance log-mel normalization clamps every frame to `log_spec.max() - 8.0`, computed over
    the *whole* waveform -- so whatever raw content happens to sit in the padded region shifts the
    normalization applied to the valid frames too, before Section 8c's mel-level canonicalization
    (which runs downstream, inside the model) ever gets a chance to intervene. Verified directly:
    without this, two waveforms with byte-identical valid content but different padding-region
    content produce measurably different mel values on the *valid* frames (see
    tests/test_whisper_dataset.py). build_example already only ever emits zero-padded waveforms in
    normal use, so this is a no-op there; it exists so the invariance holds regardless of what
    happens to a waveform between build_example and feature extraction."""
    return np.where(valid_sample_mask, waveform, 0.0).astype(waveform.dtype)


def extract_whisper_features(feature_extractor, waveform: np.ndarray, valid_sample_mask: np.ndarray,
                              sr: int):
    """Shared feature-extraction path: silence the padded region, run WhisperFeatureExtractor, and
    derive the frame mask directly from valid_sample_mask using center=True framing -- matching
    WhisperFeatureExtractor's own internal convention (torch.stft's default center=True,
    reflect-padded), not build_example's center=False valid_frame_mask, which places frame i's
    window at a different sample offset and would mislabel the valid/padding boundary frame by about
    one frame. One place so this can't drift out of sync between WhisperTurnDataset, latency
    measurement, and the Section 8e/8f diagnostics."""
    safe_waveform = silence_invalid_samples(waveform, valid_sample_mask)
    feats = feature_extractor(safe_waveform, sampling_rate=sr, padding=False, return_tensors="np")
    input_features = feats["input_features"][0]  # (n_mels, T)
    n_frames = input_features.shape[1]
    frame_length = int(round(FRAME_LENGTH_S * sr))
    hop_length = int(round(HOP_LENGTH_S * sr))
    vfm = frame_valid_mask(valid_sample_mask, frame_length, hop_length, center=True)[:n_frames]
    if len(vfm) < n_frames:
        vfm = np.concatenate([vfm, np.zeros(n_frames - len(vfm), dtype=bool)])
    return input_features.astype(np.float32), vfm


class WhisperTurnDataset(Dataset):
    def __init__(self, split: str, context_s: float = 4.0,
                 splits_path: Path = CACHE_DIR / "tinyturn_splits.parquet",
                 signal_features_path: Path = CACHE_DIR / "d2_stratified_signal_features.parquet",
                 transcripts_path: Path = CACHE_DIR / "d2_stratified_transcripts.parquet",
                 wav_dir: Path = WAV_DIR, model_name: str = WHISPER_MODEL_NAME,
                 augment_boundaries: bool = False,
                 boundary_augmentation_path: Path = TRAIN_BOUNDARY_AUGMENTATION_PATH):
        """`augment_boundaries` (8g remediation, "boundary-robust A0 retrain"): at __getitem__ time,
        randomly pick among the precomputed canonical/alt-threshold/Silero boundary estimates for
        this clip to build the windowed example, independent of the clip's label -- teaches
        robustness to exactly the boundary disagreement 8g's VAD-boundary gate measures at
        qualification time. The label itself always stays the ground-truth `endpoint_bool`; only
        which timestamp anchors the window changes. Intended for the train split only -- val/calib
        should keep using the canonical boundary (calib in particular: "calibrate only on the
        canonical boundary," brief Section 2), so callers should simply not pass
        `augment_boundaries=True` for those splits rather than this class special-casing `split`.
        Missing precomputed rows (a clip not covered by the boundary-augmentation parquet) silently
        fall back to canonical-only, so partially-populated caches still work."""
        self.context_s = context_s
        self.wav_dir = wav_dir
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.augment_boundaries = augment_boundaries

        splits = pd.read_parquet(splits_path)
        sf_feat = pd.read_parquet(signal_features_path)[
            ["id", "language", "dataset", "synthetic", "endpoint_bool", "last_active_t"]
        ]
        trans = pd.read_parquet(transcripts_path)[["id", "endfiller_derived"]]
        df = splits[splits["split"] == split][["id"]].merge(sf_feat, on="id", how="left")
        df = df.merge(trans, on="id", how="left")
        df["implicit_incomplete"] = (~df["endpoint_bool"]) & (df["endfiller_derived"] == False)  # noqa: E712

        if augment_boundaries:
            aug = pd.read_parquet(boundary_augmentation_path)[
                ["id", "alt_threshold_boundary_s", "silero_boundary_s"]
            ]
            df = df.merge(aug, on="id", how="left")
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_wav(self, row_id):
        data, sr = sf.read(self.wav_dir / f"{row_id}.wav")
        y = data if data.ndim == 1 else data.mean(axis=1)
        return y.astype(np.float32), sr

    def _pick_speech_end_s(self, row) -> float:
        canonical = float(row["last_active_t"])
        if not self.augment_boundaries:
            return canonical
        candidates = [canonical]
        for col in ("alt_threshold_boundary_s", "silero_boundary_s"):
            v = row.get(col)
            if v is not None and v == v:  # not NaN
                candidates.append(float(v))
        return random.choice(candidates)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y, sr = self._load_wav(row["id"])
        speech_end_s = self._pick_speech_end_s(row)

        ex = build_example(
            y, sr, speech_end_s, self.context_s,
            frame_length_s=FRAME_LENGTH_S, hop_length_s=HOP_LENGTH_S,
            label=bool(row["endpoint_bool"]), row_id=row["id"], source=row["dataset"],
            language=row["language"], synthetic=bool(row["synthetic"]),
            endfiller=row["endfiller_derived"] if pd.notna(row["endfiller_derived"]) else None,
        )

        input_features, valid_frame_mask = extract_whisper_features(
            self.feature_extractor, ex.waveform, ex.valid_sample_mask, sr,
        )

        return {
            "input_features": torch.from_numpy(input_features),
            "valid_frame_mask": torch.from_numpy(valid_frame_mask),
            "label": torch.tensor(float(ex.label), dtype=torch.float32),
            "id": row["id"], "language": row["language"], "dataset": row["dataset"],
            "synthetic": bool(row["synthetic"]), "implicit_incomplete": bool(row["implicit_incomplete"]),
        }


def collate(batch):
    out = {}
    for key in ["input_features", "valid_frame_mask", "label"]:
        out[key] = torch.stack([b[key] for b in batch])
    for key in ["id", "language", "dataset", "synthetic", "implicit_incomplete"]:
        out[key] = [b[key] for b in batch]
    return out
