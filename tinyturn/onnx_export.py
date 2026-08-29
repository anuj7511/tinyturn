"""
ONNX export + batch-1 CPU latency harness. Latency is measured **including feature extraction**
(waveform -> log-mel [+ trajectory] -> model forward), not classifier-forward-pass time alone --
the brief explicitly calls out the external reference solution's headline 8.71ms figure for
excluding feature extraction and asks not to repeat that ambiguity.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

# torch's dynamo-based ONNX exporter prints a unicode checkmark; Windows consoles default to
# cp1252, which can't encode it -- reconfigure stdout so `export_onnx` doesn't crash on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from tinyturn.dataset import N_MELS, FRAME_LENGTH_S, HOP_LENGTH_S, N_FFT
from tinyturn.features import compute_trajectory_channels, compute_f0_channel
import librosa


def export_onnx(model, out_path: Path, n_frames: int, n_mels: int = N_MELS,
                 trajectory_dim: int = 0):
    model.eval()
    dummy_mel = torch.randn(1, n_frames, n_mels)
    dummy_mask = torch.ones(1, n_frames, dtype=torch.bool)
    input_names = ["log_mel", "valid_frame_mask"]
    dynamic_axes = {"log_mel": {1: "time"}, "valid_frame_mask": {1: "time"}, "logit": {0: "batch"}}
    args = (dummy_mel, dummy_mask)
    if trajectory_dim > 0:
        dummy_traj = torch.randn(1, n_frames, trajectory_dim)
        args = args + (dummy_traj,)
        input_names.append("trajectory")
        dynamic_axes["trajectory"] = {1: "time"}

    # dynamo=False: pin the legacy TorchScript-based exporter. Newer torch defaults `dynamo=True`,
    # whose dynamic-shape handling doesn't accept `dynamic_axes` cleanly (raises TorchExportError
    # here) -- discovered while adding Section 8c's ONNX/PyTorch-parity regression test.
    torch.onnx.export(
        model, args, str(out_path), input_names=input_names, output_names=["logit"],
        dynamic_axes=dynamic_axes, opset_version=17, dynamo=False,
    )
    return out_path


def _feature_extract(y: np.ndarray, sr: int, include_trajectory: bool, trajectory_names: list,
                      valid_sample_mask: np.ndarray):
    frame_length = int(round(FRAME_LENGTH_S * sr))
    hop_length = int(round(HOP_LENGTH_S * sr))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=hop_length,
                                          win_length=frame_length, n_mels=N_MELS, center=False)
    log_mel = np.log(mel + 1e-6).T.astype(np.float32)
    traj = None
    if include_trajectory:
        chans = compute_trajectory_channels(y, sr, valid_sample_mask, FRAME_LENGTH_S, HOP_LENGTH_S)
        n_frames = log_mel.shape[0]
        if "f0_semitone" in trajectory_names:
            chans["f0_semitone"] = compute_f0_channel(y, sr, n_frames=n_frames, hop_length=hop_length,
                                                        frame_length=frame_length)
        traj = np.stack([chans[n][:n_frames] for n in trajectory_names], axis=-1).astype(np.float32)
    return log_mel, traj


def measure_latency_ms(onnx_path: Path, sample_waveform: np.ndarray, sr: int,
                        valid_sample_mask: np.ndarray, include_trajectory: bool,
                        trajectory_names: list, n_warmup: int = 10, n_runs: int = 100) -> dict:
    """Batch-1 CPU latency, one full pass = feature extraction + ONNXRuntime forward."""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def _one_pass():
        t0 = time.perf_counter()
        log_mel, traj = _feature_extract(sample_waveform, sr, include_trajectory, trajectory_names,
                                          valid_sample_mask)
        feed = {
            "log_mel": log_mel[None, :, :],
            "valid_frame_mask": np.ones((1, log_mel.shape[0]), dtype=np.bool_),
        }
        if include_trajectory:
            feed["trajectory"] = traj[None, :, :]
        sess.run(["logit"], feed)
        return (time.perf_counter() - t0) * 1000.0

    for _ in range(n_warmup):
        _one_pass()
    times = [_one_pass() for _ in range(n_runs)]
    return {
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
        "mean_ms": round(float(np.mean(times)), 3),
        "n_runs": n_runs,
    }


def onnx_file_size_bytes(onnx_path: Path) -> int:
    return Path(onnx_path).stat().st_size
