"""
Step 7 -- internal-pause continuation events.

```text
audio ends exactly at the start of an internal pause -> label = incomplete
```

By definition, a clip cut at the moment an internal pause begins is a true "incomplete" example:
the speaker is mid-utterance and about to continue. This reuses the exact same `build_example`
contract as every other experiment (Phase-2 8b: N seconds ending at a boundary, no post-roll) --
the only difference is which moment in the clip counts as the "speech end" boundary: instead of the
canonical final `last_active_t`, a pause event's boundary is the *start* of one of the clip's own
internal pauses. There is no 200ms post-roll to land inside the pause anymore -- the runtime 200ms
wait is VAD policy, evaluated at the pause boundary itself, not baked into what the model sees.

Eligibility: a pause must be at least as long as the runtime trigger duration (`RUNTIME_TRIGGER_S`,
0.2s) -- a shorter pause wouldn't actually have kept the turn-taking system waiting long enough to
act, so it isn't a realistic "the system paused here" event. The event also qualifies only if
speech demonstrably resumes after the pause: `tinyturn.boundary.estimate_speech_end` only ever
records a span in `internal_pause_spans_s` when it's strictly between two other runs in the
clip's run-length encoding (`runs[1:-1]`), and consecutive runs always alternate active/inactive by
construction (`segment_runs`) -- so any recorded internal pause is always followed by another
active (speech) run. This is verified directly in `tests/test_pause_events.py`, not just assumed.
Capped at `MAX_EVENTS_PER_CLIP` per clip (longest pauses first) so a handful of pathological
clips with 20-50 short pauses don't dominate the augmented training set.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from tinyturn.boundary import estimate_speech_end
from tinyturn.preprocess import build_example
from tinyturn.features import compute_trajectory_channels
from tinyturn.dataset import (
    CACHE_DIR, WAV_DIR, FEATURE_CACHE_DIR, N_MELS, N_FFT, FRAME_LENGTH_S, HOP_LENGTH_S,
    TRAJECTORY_NAMES, TARGET_SR, PREPROCESSING_VERSION, _atomic_save_npz,
)

RUNTIME_TRIGGER_S = 0.200  # runtime VAD wait (Phase-2 8b) -- eligibility floor only, not baked
                           # into the model input the way the old post-roll was.
MIN_PAUSE_S = RUNTIME_TRIGGER_S
MAX_EVENTS_PER_CLIP = 2
EVENTS_PATH = CACHE_DIR / "tinyturn_pause_events.parquet"


def extract_pause_events_for_clip(clip_id: str, y: np.ndarray, sr: int,
                                   min_pause_s: float = MIN_PAUSE_S,
                                   max_events: int = MAX_EVENTS_PER_CLIP) -> list[dict]:
    est = estimate_speech_end(y, sr)
    eligible = [(s, e) for s, e in est.internal_pause_spans_s if (e - s) >= min_pause_s]
    eligible.sort(key=lambda span: span[1] - span[0], reverse=True)  # longest pauses first
    eligible = eligible[:max_events]
    return [
        {"clip_id": clip_id, "event_idx": i, "pause_start_s": s, "pause_end_s": e,
         "pause_duration_s": e - s}
        for i, (s, e) in enumerate(eligible)
    ]


class PauseEventDataset(Dataset):
    """Same item schema as TinyTurnDataset (log_mel, valid_frame_mask, label, trajectory, id,
    language, dataset, synthetic, implicit_incomplete) plus `is_pause_event=True`, so the two can
    be concatenated directly (`torch.utils.data.ConcatDataset`) for P1's augmented training set."""

    def __init__(self, split: str, context_s: float = 1.0,
                 include_trajectory: bool = True, events_path: Path = EVENTS_PATH,
                 splits_path: Path = CACHE_DIR / "tinyturn_splits.parquet",
                 wav_dir: Path = WAV_DIR):
        self.context_s = context_s
        self.include_trajectory = include_trajectory
        self.wav_dir = wav_dir

        events = pd.read_parquet(events_path)
        meta = pd.read_parquet(splits_path)[["id", "split", "dataset", "language", "synthetic"]] \
            .rename(columns={"id": "clip_id"})
        df = events.merge(meta, on="clip_id", how="left")
        df = df[df["split"] == split].reset_index(drop=True)
        self.df = df

    def __len__(self):
        return len(self.df)

    def _load_wav(self, clip_id: str):
        data, sr = sf.read(self.wav_dir / f"{clip_id}.wav")
        y = data if data.ndim == 1 else data.mean(axis=1)
        y = y.astype(np.float32)
        if sr != TARGET_SR:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        return y, sr

    def _log_mel(self, waveform, sr):
        import librosa
        frame_length = int(round(FRAME_LENGTH_S * sr))
        hop_length = int(round(HOP_LENGTH_S * sr))
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_fft=N_FFT, hop_length=hop_length, win_length=frame_length,
            n_mels=N_MELS, center=False,
        )
        return np.log(mel + 1e-6).T.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        event_id = f"{row['clip_id']}__pause{row['event_idx']}"
        mel_cache = FEATURE_CACHE_DIR / f"{event_id}_ctx{self.context_s}_{PREPROCESSING_VERSION}_mel.npz"
        traj_cache = FEATURE_CACHE_DIR / f"{event_id}_ctx{self.context_s}_{PREPROCESSING_VERSION}_traj.npz"

        if mel_cache.exists():
            with np.load(mel_cache) as z:
                log_mel, valid_frame_mask = z["log_mel"], z["valid_frame_mask"]
            ex = None
        else:
            y, sr = self._load_wav(row["clip_id"])
            ex = build_example(
                y, sr, float(row["pause_start_s"]), self.context_s,
                frame_length_s=FRAME_LENGTH_S, hop_length_s=HOP_LENGTH_S, label=False,
                row_id=event_id, source=row["dataset"], language=row["language"],
                synthetic=bool(row["synthetic"]),
            )
            log_mel = self._log_mel(ex.waveform, sr)
            n_frames = log_mel.shape[0]
            valid_frame_mask = ex.valid_frame_mask[:n_frames]
            if len(valid_frame_mask) < n_frames:
                valid_frame_mask = np.concatenate(
                    [valid_frame_mask, np.zeros(n_frames - len(valid_frame_mask), dtype=bool)])
            FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _atomic_save_npz(mel_cache, log_mel=log_mel, valid_frame_mask=valid_frame_mask)

        n_frames = log_mel.shape[0]
        item = {
            "log_mel": torch.from_numpy(log_mel),
            "valid_frame_mask": torch.from_numpy(valid_frame_mask),
            "label": torch.tensor(0.0, dtype=torch.float32),
            "id": event_id,
            "language": row["language"],
            "dataset": row["dataset"],
            "synthetic": bool(row["synthetic"]),
            "implicit_incomplete": False,
            "is_pause_event": True,
            # Internal holds never carry a distillation target (Step 10: "hard labels on internal
            # holds") -- NaN, not 0.0, so a bug that accidentally used this value would be loud
            # (NaN propagates through any loss term that touches it) rather than silently training
            # against a fake teacher logit of 0.
            "teacher_logit": torch.tensor(float("nan"), dtype=torch.float32),
        }

        if self.include_trajectory:
            if traj_cache.exists():
                with np.load(traj_cache) as z:
                    traj = z["traj"]
            else:
                if ex is None:
                    y, sr = self._load_wav(row["clip_id"])
                    ex = build_example(
                        y, sr, float(row["pause_start_s"]), self.context_s,
                        frame_length_s=FRAME_LENGTH_S, hop_length_s=HOP_LENGTH_S, label=False,
                        row_id=event_id, source=row["dataset"], language=row["language"],
                        synthetic=bool(row["synthetic"]),
                    )
                chans = compute_trajectory_channels(ex.waveform, sr, ex.valid_sample_mask,
                                                     FRAME_LENGTH_S, HOP_LENGTH_S)
                traj = np.stack([chans[n][:n_frames] for n in TRAJECTORY_NAMES], axis=-1).astype(np.float32)
                if traj.shape[0] < n_frames:
                    traj = np.pad(traj, ((0, n_frames - traj.shape[0]), (0, 0)))
                _atomic_save_npz(traj_cache, traj=traj)
            item["trajectory"] = torch.from_numpy(traj.astype(np.float32))

        return item
