"""
Step 1 -- PyTorch-facing dataset wrapper around preprocess.build_example + boundary.py, plus
log-mel and (optionally) trajectory-branch feature extraction for B0/B1/B1-f0/A0.

Working dataset: the D2 stratified cache (data_cache/d2_stratified_wavs/, 15,998 clips) split via
tinyturn/splits.py. The official 31,527-row HF test set is intentionally NOT wired in here -- per
the brief's Section 8 discipline it is touched once per finalist, well past Steps 1-5.

Expensive per-clip features (log-mel, trajectory channels, pyin F0) are memoized to disk on first
computation (`data_cache/tinyturn_feature_cache/`) -- purely a performance optimization (values are
byte-identical to computing fresh every time), but it's what makes B1-f0's pyin cost tractable
across multiple epochs without giving it a different epoch budget than B0/B1.
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset

from tinyturn.preprocess import build_example
from tinyturn.boundary import estimate_speech_end
from tinyturn.features import compute_trajectory_channels, compute_f0_channel

CACHE_DIR = Path("data_cache")
WAV_DIR = CACHE_DIR / "d2_stratified_wavs"
FEATURE_CACHE_DIR = CACHE_DIR / "tinyturn_feature_cache"

# Bumped when the input-construction contract changes (Phase-2 8b: speech-aligned windows, no
# baked-in post-roll) -- folded into every cache filename so features computed under the old
# N+200ms-post-roll contract can never be silently reused for the new one (implementation rule:
# "Never silently reuse a cache created under another preprocessing version").
PREPROCESSING_VERSION = "pv2-speechend"

N_MELS = 40
N_FFT = 512
FRAME_LENGTH_S = 0.025
HOP_LENGTH_S = 0.010
TARGET_SR = 16000

TRAJECTORY_NAMES = ["rel_energy", "pause_prob", "spectral_tilt", "spectral_flux", "envelope_activity"]

# Languages with any real (non-synthetic) audio anywhere in the dataset (D3) -- everything else is
# one of the 21 fully-synthetic languages and must be reported as a separate group, never blended.
REAL_LANGUAGES = {"eng", "spa"}


def _atomic_save_npz(path: Path, **arrays):
    """Write-to-temp-then-rename so concurrent DataLoader worker processes racing on the same
    cache key never see a partially-written file (os.replace is atomic on the same volume)."""
    tmp = path.with_suffix(f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


@dataclass
class TinyTurnBatchMeta:
    id: str
    language: str
    dataset: str
    synthetic: bool
    implicit_incomplete: bool


TRAIN_BOUNDARY_AUGMENTATION_PATH = CACHE_DIR / "d2_train_boundary_augmentation.parquet"


class TinyTurnDataset(Dataset):
    def __init__(self, split: str, context_s: float = 1.0,
                 include_trajectory: bool = False, include_f0: bool = False,
                 boundary_source: str = "cached", use_disk_cache: bool = True,
                 splits_path: Path = CACHE_DIR / "tinyturn_splits.parquet",
                 signal_features_path: Path = CACHE_DIR / "d2_stratified_signal_features.parquet",
                 transcripts_path: Path = CACHE_DIR / "d2_stratified_transcripts.parquet",
                 wav_dir: Path = WAV_DIR,
                 augment_boundaries: bool = False,
                 boundary_augmentation_path: Path = TRAIN_BOUNDARY_AUGMENTATION_PATH,
                 teacher_logit_path: Optional[Path] = None):
        """
        augment_boundaries (Step 10 distillation ablation, mirrors WhisperTurnDataset's identically-
        named param from the 8g remediation retrain): at __getitem__ time, randomly pick among the
        precomputed canonical/alt-threshold/Silero boundary estimates for this clip, independent of
        label. Disables the on-disk mel/trajectory feature cache for this instance -- that cache is
        keyed only by (row_id, context_s), not by which boundary produced the window, so serving it
        under randomized boundaries would either always return one stale (whichever-was-cached-first)
        window or silently defeat the whole point of augmenting. Intended for the train split only.

        teacher_logit_path: optional parquet with columns [id, teacher_logit] (Step 10 distillation)
        -- merged in as a `teacher_logit` float column (NaN for any id not covered), exposed on every
        item so a distillation loss can select it; unrelated to `augment_boundaries`'s per-epoch
        random draw (the teacher target is fixed per clip, not re-drawn).
        """
        """
        boundary_source:
          "cached"    -- use the precomputed `last_active_t` from D2 (tinyturn.boundary's formula,
                         run once during the EDA) -- the version-0 canonical boundary, reused for
                         numeric identity with the rest of the EDA and to avoid recomputing it.
          "recompute" -- call tinyturn.boundary.estimate_speech_end on the loaded waveform. Same
                         formula, so should closely match "cached" -- this path exists so the
                         boundary estimator is genuinely swappable (Step 1's "configurable boundary
                         estimator" requirement) and so callers with no precomputed value (e.g. a
                         future official-test-set loader) still work.
        """
        assert boundary_source in ("cached", "recompute")
        self.context_s = context_s
        self.include_trajectory = include_trajectory
        self.include_f0 = include_f0
        self.boundary_source = boundary_source
        self.augment_boundaries = augment_boundaries
        # Randomized boundaries can't safely share the (row_id, context_s)-keyed disk cache -- see
        # the augment_boundaries docstring above.
        self.use_disk_cache = use_disk_cache and not augment_boundaries
        self.wav_dir = wav_dir

        splits = pd.read_parquet(splits_path)
        sf_feat = pd.read_parquet(signal_features_path)[
            ["id", "language", "dataset", "synthetic", "endpoint_bool", "endfiller", "last_active_t"]
        ]
        trans = pd.read_parquet(transcripts_path)[["id", "endfiller_derived"]]

        df = splits[splits["split"] == split][["id"]].merge(sf_feat, on="id", how="left")
        df = df.merge(trans, on="id", how="left")
        df["implicit_incomplete"] = (~df["endpoint_bool"]) & (df["endfiller_derived"] == False)  # noqa: E712
        # `endfiller` output field: prefer the ground-truth scripted label (defined for synthetic
        # clips) and fall back to the ASR-derived label only where ground truth is unknown (real
        # audio, per D2/eda_part3_implementation_brief -- raw `endfiller` is None there by
        # construction, not a missing-data accident).
        df["endfiller_resolved"] = df["endfiller"].where(df["endfiller"].notna(), df["endfiller_derived"])

        if augment_boundaries:
            aug = pd.read_parquet(boundary_augmentation_path)[
                ["id", "alt_threshold_boundary_s", "silero_boundary_s"]
            ]
            df = df.merge(aug, on="id", how="left")

        df["teacher_logit"] = np.nan
        if teacher_logit_path is not None:
            tl = pd.read_parquet(teacher_logit_path)[["id", "teacher_logit"]]
            df = df.drop(columns=["teacher_logit"]).merge(tl, on="id", how="left")

        self.df = df.reset_index(drop=True)

        if self.use_disk_cache:
            FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def _load_wav(self, row_id: str):
        data, sr = sf.read(self.wav_dir / f"{row_id}.wav")
        y = data if data.ndim == 1 else data.mean(axis=1)
        y = y.astype(np.float32)
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        return y, sr

    def _log_mel(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        frame_length = int(round(FRAME_LENGTH_S * sr))
        hop_length = int(round(HOP_LENGTH_S * sr))
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_fft=N_FFT, hop_length=hop_length, win_length=frame_length,
            n_mels=N_MELS, center=False,
        )
        return np.log(mel + 1e-6).T.astype(np.float32)  # (T, n_mels)

    def _cache_key(self, row_id: str, suffix: str) -> Path:
        return FEATURE_CACHE_DIR / f"{row_id}_ctx{self.context_s}_{PREPROCESSING_VERSION}_{suffix}.npz"

    def _get_speech_end_s(self, row, y, sr) -> float:
        if self.boundary_source != "cached":
            return estimate_speech_end(y, sr).speech_end_s
        canonical = float(row["last_active_t"])
        if not self.augment_boundaries:
            return canonical
        candidates = [canonical]
        for col in ("alt_threshold_boundary_s", "silero_boundary_s"):
            v = row.get(col)
            if v is not None and v == v:  # not NaN
                candidates.append(float(v))
        import random
        return random.choice(candidates)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        row_id = row["id"]

        mel_cache_path = self._cache_key(row_id, "mel") if self.use_disk_cache else None
        if mel_cache_path is not None and mel_cache_path.exists():
            with np.load(mel_cache_path) as z:
                log_mel = z["log_mel"]
                valid_frame_mask = z["valid_frame_mask"]
            y = sr = ex = None  # not needed unless trajectory/f0 also need (re)computing below
        else:
            y, sr = self._load_wav(row_id)
            speech_end_s = self._get_speech_end_s(row, y, sr)
            ex = build_example(
                y, sr, speech_end_s, self.context_s,
                frame_length_s=FRAME_LENGTH_S, hop_length_s=HOP_LENGTH_S,
                label=bool(row["endpoint_bool"]), row_id=row_id, source=row["dataset"],
                language=row["language"], synthetic=bool(row["synthetic"]),
                endfiller=row["endfiller_resolved"] if pd.notna(row["endfiller_resolved"]) else None,
            )
            log_mel = self._log_mel(ex.waveform, sr)
            n_frames = log_mel.shape[0]
            valid_frame_mask = ex.valid_frame_mask[:n_frames]
            if len(valid_frame_mask) < n_frames:
                pad = np.zeros(n_frames - len(valid_frame_mask), dtype=bool)
                valid_frame_mask = np.concatenate([valid_frame_mask, pad])
            if mel_cache_path is not None:
                _atomic_save_npz(mel_cache_path, log_mel=log_mel, valid_frame_mask=valid_frame_mask)

        n_frames = log_mel.shape[0]
        item = {
            "log_mel": torch.from_numpy(log_mel),
            "valid_frame_mask": torch.from_numpy(valid_frame_mask),
            "label": torch.tensor(float(row["endpoint_bool"]), dtype=torch.float32),
            "id": row_id,
            "language": row["language"],
            "dataset": row["dataset"],
            "synthetic": bool(row["synthetic"]),
            "implicit_incomplete": bool(row["implicit_incomplete"]),
            "is_pause_event": False,
            "teacher_logit": torch.tensor(float(row["teacher_logit"]), dtype=torch.float32),
        }

        if self.include_trajectory:
            traj_cache_path = self._cache_key(row_id, "traj") if self.use_disk_cache else None
            f0_cache_path = self._cache_key(row_id, "f0") if self.use_disk_cache else None

            traj_base = None
            if traj_cache_path is not None and traj_cache_path.exists():
                with np.load(traj_cache_path) as z:
                    traj_base = z["traj"]

            f0_arr = None
            if self.include_f0 and f0_cache_path is not None and f0_cache_path.exists():
                with np.load(f0_cache_path) as z:
                    f0_arr = z["f0"]

            need_recompute = traj_base is None or (self.include_f0 and f0_arr is None)
            if need_recompute:
                if y is None:
                    y, sr = self._load_wav(row_id)
                if ex is None:
                    speech_end_s = self._get_speech_end_s(row, y, sr)
                    ex = build_example(
                        y, sr, speech_end_s, self.context_s,
                        frame_length_s=FRAME_LENGTH_S, hop_length_s=HOP_LENGTH_S,
                        label=bool(row["endpoint_bool"]), row_id=row_id, source=row["dataset"],
                        language=row["language"], synthetic=bool(row["synthetic"]),
                        endfiller=row["endfiller_resolved"] if pd.notna(row["endfiller_resolved"]) else None,
                    )
                frame_length = int(round(FRAME_LENGTH_S * sr))
                hop_length = int(round(HOP_LENGTH_S * sr))

                if traj_base is None:
                    chans = compute_trajectory_channels(ex.waveform, sr, ex.valid_sample_mask,
                                                         FRAME_LENGTH_S, HOP_LENGTH_S)
                    traj_base = np.stack([chans[n][:n_frames] for n in TRAJECTORY_NAMES], axis=-1).astype(np.float32)
                    if traj_base.shape[0] < n_frames:
                        traj_base = np.pad(traj_base, ((0, n_frames - traj_base.shape[0]), (0, 0)))
                    if traj_cache_path is not None:
                        _atomic_save_npz(traj_cache_path, traj=traj_base)

                if self.include_f0 and f0_arr is None:
                    f0_arr = compute_f0_channel(ex.waveform, sr, n_frames=n_frames,
                                                 hop_length=hop_length, frame_length=frame_length)
                    if f0_cache_path is not None:
                        _atomic_save_npz(f0_cache_path, f0=f0_arr)

            traj = traj_base
            if self.include_f0:
                f0_col = f0_arr[:n_frames]
                if len(f0_col) < n_frames:
                    f0_col = np.pad(f0_col, (0, n_frames - len(f0_col)))
                traj = np.concatenate([traj, f0_col[:, None]], axis=-1)
            item["trajectory"] = torch.from_numpy(traj.astype(np.float32))

        return item


def collate(batch):
    """All examples share a fixed context_s => fixed T, so a plain stack suffices."""
    out = {}
    for key in ["log_mel", "valid_frame_mask", "label", "teacher_logit"]:
        out[key] = torch.stack([b[key] for b in batch])
    if "trajectory" in batch[0]:
        out["trajectory"] = torch.stack([b["trajectory"] for b in batch])
    for key in ["id", "language", "dataset", "synthetic", "implicit_incomplete", "is_pause_event"]:
        out[key] = [b[key] for b in batch]
    return out
