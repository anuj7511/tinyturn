"""
Step 10 planning, item 8 -- THE official-test-set evaluation. Touched exactly once, per this
project's Section 8 discipline ("official 31,527-row HF test set... touched once per finalist").

Frozen per explicit user instruction:
  checkpoint: experiments/data_scale_64k_holdloss0.5_5050sampling_seed43/checkpoint.pt
    (sha256 ddaf7a8ea95b6675022920b68b95e7a1f8202ab403c3e7e11e08dc5f0892694f -- see
    experiments/data_scale_64k_holdloss0.5_5050sampling_seed43/FROZEN_MANIFEST.json)
  temperature: T=1.5142 (user said 1.514; stored value from temperature_scaling_64k.json used
    verbatim, matches to 3 decimals)
  threshold: 0.612 (on the TEMPERATURE-SCALED probability, selected on calib only -- see Section 6f)
  preprocessing: pv2-speechend, context_s=1.0

Three phases, run separately and in order, so a bug discovered in report aggregation never forces
re-running model inference against the test set, and a bug discovered in feature extraction never
forces re-downloading/re-decoding the ~4.84GB of test audio:

  --phase fetch_features   Downloads all 10 parquet shards of pipecat-ai/smart-turn-data-v3.2-test
                            (confirmed via HfApi.list_repo_files: data/train-{i:05d}-of-00010.parquet,
                            i=0..9), decodes each of the 31,527 clips, computes the canonical v0
                            boundary FRESH (tinyturn.boundary.estimate_speech_end -- the official test
                            set has no precomputed last_active_t the way D2 does; this is exactly the
                            "recompute" path tinyturn/dataset.py's TinyTurnDataset docstring already
                            anticipated for "a future official-test-set loader"), builds the
                            context_s=1.0 example (tinyturn.preprocess.build_example), and computes
                            log-mel + trajectory-channel features identically to TinyTurnDataset's own
                            feature pipeline (same N_MELS/N_FFT/frame/hop constants, same
                            TRAJECTORY_NAMES). No ASR -- endfiller/midfiller ground truth ships
                            natively on synthetic rows in the raw HF metadata (same convention D2
                            already relies on), so nothing plain evaluation needs is lost by skipping
                            it. Saved to data_cache/official_test_features.npz (fixed-shape arrays,
                            since context_s=1.0 always yields exactly 98 frames) +
                            data_cache/official_test_metadata.parquet. Resumable via a checkpoint file.

  --phase infer             Loads the frozen checkpoint ONCE, runs a single forward pass per clip over
                            the cached features from `fetch_features`, and saves raw per-clip results
                            (id, y_true, logit, prob_raw, prob_temp_scaled, decision, language,
                            dataset, synthetic, endfiller) to
                            experiments/official_test_per_clip_results.parquet. This is the one and
                            only inference pass against the official test set for this project.

  --phase report            Computes the standard Section-8 `full_report` (same function used for
                            every other checkpoint in this project) from the saved per-clip results --
                            purely a re-derivable aggregation step, safe to re-run if a bug is found,
                            since it never touches the model or the test audio again. Saved to
                            experiments/official_test_report.json.

KNOWN LIMITATION, disclosed rather than papered over: `implicit_incomplete` ground truth requires
`endfiller` -- 100% populated for synthetic test rows, 100% NULL for real-audio test rows (checked
directly against data_cache/d1_test_hashes_fingerprints.parquet, same as D2's real-audio subset).
D2's real audio got ASR-derived endfiller labels from a prior (pre-Step-10) transcription pass; running
that fresh for 5,863 real test clips was judged out of scope for "run the official test once" and not
requested. The `implicit_incomplete` slice below therefore only reflects SYNTHETIC clips with known
ground truth; real-audio rows are excluded from that specific slice (not silently folded in as
"not implicit"), and this is called out again in the saved report.

Usage:
  python scripts/run_official_test_evaluation.py --phase fetch_features
  python scripts/run_official_test_evaluation.py --phase infer
  python scripts/run_official_test_evaluation.py --phase report
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.preprocess import build_example
from tinyturn.boundary import estimate_speech_end
from tinyturn.features import compute_trajectory_channels
from tinyturn.models import TinyTurnModel
from tinyturn.evaluate import full_report, EvalOutputs

REPO = "pipecat-ai/smart-turn-data-v3.2-test"
N_SHARDS = 10
CACHE_DIR = Path("data_cache")
FEATURES_NPZ = CACHE_DIR / "official_test_features.npz"
METADATA_PARQUET = CACHE_DIR / "official_test_metadata.parquet"
CKPT_META = CACHE_DIR / "_official_test_checkpoint_metadata.parquet"
CKPT_ARR_DIR = CACHE_DIR / "_official_test_checkpoint_arrays"
CHECKPOINT_EVERY = 2000
N_WORKERS = 14

FROZEN_CHECKPOINT_DIR = Path("experiments") / "data_scale_64k_holdloss0.5_5050sampling_seed43"
TEMPERATURE = 1.5142
THRESHOLD_TEMP_SCALED = 0.612
CONTEXT_S = 1.0
N_MELS, N_FFT = 40, 512
FRAME_LENGTH_S, HOP_LENGTH_S = 0.025, 0.010
TARGET_SR = 16000
TRAJECTORY_NAMES = ["rel_energy", "pause_prob", "spectral_tilt", "spectral_flux", "envelope_activity"]


def _log_mel(y, sr):
    frame_length = int(round(FRAME_LENGTH_S * sr))
    hop_length = int(round(HOP_LENGTH_S * sr))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=hop_length,
                                          win_length=frame_length, n_mels=N_MELS, center=False)
    return np.log(mel + 1e-6).T.astype(np.float32)


# N_FRAMES_FIXED derived EMPIRICALLY from the real _log_mel path, not a hand-derived formula --
# librosa's center=False frame count depends on n_fft (512), not win_length (frame_length_s*sr=400),
# a real off-by-one caught by checking against an actual cached training feature
# (data_cache/tinyturn_feature_cache/*_ctx1.0_pv2-speechend_mel.npz has shape (97, 40), not (98, 40)
# as a naive 1 + (16000 - 400) // 160 formula would give).
N_FRAMES_FIXED = _log_mel(np.zeros(int(round(CONTEXT_S * TARGET_SR)), dtype=np.float32), TARGET_SR).shape[0]

PER_CLIP_RESULTS_PATH = Path("experiments") / "official_test_per_clip_results.parquet"
REPORT_PATH = Path("experiments") / "official_test_report.json"


def _pad_to(arr, n_frames):
    arr = arr[:n_frames]
    if len(arr) < n_frames:
        pad_shape = (n_frames - len(arr),) + arr.shape[1:]
        arr = np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)
    return arr


def process_row(row_dict):
    ex_id = row_dict["id"]
    try:
        audio_bytes = row_dict["audio"]["bytes"]
        data, sr = sf.read(io.BytesIO(audio_bytes))
        y = data if data.ndim == 1 else data.mean(axis=1)
        y = y.astype(np.float32)
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        speech_end_s = estimate_speech_end(y, sr).speech_end_s
        endfiller_raw = row_dict.get("endfiller")
        ex = build_example(
            y, sr, speech_end_s, CONTEXT_S, frame_length_s=FRAME_LENGTH_S, hop_length_s=HOP_LENGTH_S,
            label=bool(row_dict["endpoint_bool"]), row_id=ex_id, source=row_dict["dataset"],
            language=row_dict["language"], synthetic=bool(row_dict["synthetic"]),
            endfiller=endfiller_raw if endfiller_raw is not None else None,
        )
        log_mel = _pad_to(_log_mel(ex.waveform, sr), N_FRAMES_FIXED)
        vfm = _pad_to(ex.valid_frame_mask.astype(bool), N_FRAMES_FIXED)
        chans = compute_trajectory_channels(ex.waveform, sr, ex.valid_sample_mask, FRAME_LENGTH_S, HOP_LENGTH_S)
        traj = np.stack([chans[n] for n in TRAJECTORY_NAMES], axis=-1).astype(np.float32)
        traj = _pad_to(traj, N_FRAMES_FIXED)

        meta = {
            "id": ex_id, "language": row_dict["language"], "dataset": row_dict["dataset"],
            "synthetic": bool(row_dict["synthetic"]), "endpoint_bool": bool(row_dict["endpoint_bool"]),
            "endfiller": endfiller_raw, "speech_end_s": float(speech_end_s), "error": None,
        }
        return meta, log_mel, vfm, traj
    except Exception as e:
        meta = {"id": ex_id, "language": row_dict.get("language"), "dataset": row_dict.get("dataset"),
                "synthetic": row_dict.get("synthetic"), "endpoint_bool": row_dict.get("endpoint_bool"),
                "endfiller": None, "speech_end_s": None, "error": str(e)}
        return meta, None, None, None


def phase_fetch_features():
    CKPT_ARR_DIR.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    all_meta = []
    if CKPT_META.exists():
        all_meta = pd.read_parquet(CKPT_META).to_dict(orient="records")
        done_ids = {m["id"] for m in all_meta}
        print(f"resuming: {len(done_ids):,} rows already processed", flush=True)

    t0 = time.time()
    since_checkpoint = 0
    for shard_idx in range(N_SHARDS):
        fname = f"data/train-{shard_idx:05d}-of-{N_SHARDS:05d}.parquet"
        print(f"fetching shard {shard_idx} ({fname})...", flush=True)
        path = hf_hub_download(REPO, fname, repo_type="dataset")
        table = pq.read_table(path)
        df = table.to_pandas()
        df = df[~df["id"].isin(done_ids)].reset_index(drop=True)
        print(f"  shard {shard_idx}: {len(df):,} new rows (of {table.num_rows:,} total)", flush=True)
        if len(df) == 0:
            continue

        n_done = 0
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = [pool.submit(process_row, row.to_dict()) for _, row in df.iterrows()]
            for fut in as_completed(futures):
                meta, log_mel, vfm, traj = fut.result()
                all_meta.append(meta)
                done_ids.add(meta["id"])
                if meta["error"] is None:
                    np.savez(CKPT_ARR_DIR / f"{meta['id']}.npz", log_mel=log_mel, vfm=vfm, traj=traj)
                n_done += 1
                since_checkpoint += 1
                if n_done % 1000 == 0:
                    print(f"  [{shard_idx}] {n_done}/{len(df)} ({time.time()-t0:.0f}s elapsed total)", flush=True)
                if since_checkpoint >= CHECKPOINT_EVERY:
                    pd.DataFrame(all_meta).to_parquet(CKPT_META, index=False)
                    since_checkpoint = 0
                    print(f"  [checkpoint] {len(all_meta):,} rows saved to staging", flush=True)
        print(f"  shard {shard_idx} done at {time.time()-t0:.0f}s total elapsed", flush=True)

    pd.DataFrame(all_meta).to_parquet(CKPT_META, index=False)
    meta_df = pd.DataFrame(all_meta)
    n_errors = meta_df["error"].notna().sum()
    print(f"\ntotal rows: {len(meta_df):,} ({n_errors} decode errors)", flush=True)

    ok = meta_df[meta_df["error"].isna()].reset_index(drop=True)
    print(f"packing {len(ok):,} clips into a single array cache...", flush=True)
    log_mels = np.zeros((len(ok), N_FRAMES_FIXED, N_MELS), dtype=np.float32)
    vfms = np.zeros((len(ok), N_FRAMES_FIXED), dtype=bool)
    trajs = np.zeros((len(ok), N_FRAMES_FIXED, len(TRAJECTORY_NAMES)), dtype=np.float32)
    for i, row in ok.iterrows():
        with np.load(CKPT_ARR_DIR / f"{row['id']}.npz") as z:
            log_mels[i] = z["log_mel"]
            vfms[i] = z["vfm"]
            trajs[i] = z["traj"]
        if (i + 1) % 5000 == 0:
            print(f"  packed {i+1}/{len(ok)}", flush=True)

    np.savez(FEATURES_NPZ, log_mel=log_mels, valid_frame_mask=vfms, trajectory=trajs,
             ids=ok["id"].values)
    meta_df.to_parquet(METADATA_PARQUET, index=False)
    print(f"saved {FEATURES_NPZ} and {METADATA_PARQUET}", flush=True)
    print(f"DONE fetch_features in {time.time()-t0:.0f}s total.")


def _load_frozen_model():
    cfg = json.load(open(FROZEN_CHECKPOINT_DIR / "config.json"))
    model = TinyTurnModel(n_mels=N_MELS, trajectory_dim=len(TRAJECTORY_NAMES),
                           mel_channels=cfg["mel_channels"], traj_channels=cfg["traj_channels"])
    model.load_state_dict(torch.load(FROZEN_CHECKPOINT_DIR / "checkpoint.pt", map_location="cpu"))
    model.eval()
    return model


def phase_infer():
    print("loading cached features...", flush=True)
    with np.load(FEATURES_NPZ, allow_pickle=True) as z:
        log_mel = z["log_mel"]
        vfm = z["valid_frame_mask"]
        traj = z["trajectory"]
        ids = z["ids"]
    meta_df = pd.read_parquet(METADATA_PARQUET).set_index("id")
    print(f"loaded {len(ids):,} clips", flush=True)

    model = _load_frozen_model()
    print("running single inference pass over the official test set (this is THE evaluation)...", flush=True)

    logits = []
    batch_size = 128
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            lm = torch.from_numpy(log_mel[start:end])
            m = torch.from_numpy(vfm[start:end])
            tr = torch.from_numpy(traj[start:end])
            out = model(lm, m, tr)
            logits.extend(out.numpy().tolist())
            if (end // batch_size) % 20 == 0:
                print(f"  {end}/{len(ids)} ({time.time()-t0:.0f}s)", flush=True)
    logits = np.array(logits)

    prob_raw = 1 / (1 + np.exp(-logits))
    prob_temp_scaled = 1 / (1 + np.exp(-logits / TEMPERATURE))
    decision = prob_temp_scaled >= THRESHOLD_TEMP_SCALED

    rows = []
    for i, ex_id in enumerate(ids):
        m = meta_df.loc[ex_id]
        rows.append({
            "id": ex_id, "y_true": bool(m["endpoint_bool"]), "logit": float(logits[i]),
            "prob_raw": float(prob_raw[i]), "prob_temp_scaled": float(prob_temp_scaled[i]),
            "decision": bool(decision[i]), "language": m["language"], "dataset": m["dataset"],
            "synthetic": bool(m["synthetic"]), "endfiller": m["endfiller"],
        })
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(PER_CLIP_RESULTS_PATH, index=False)
    print(f"\nsaved {PER_CLIP_RESULTS_PATH}: {len(out_df):,} rows in {time.time()-t0:.0f}s")
    print("THE official test evaluation is done. Do not re-run this phase.")


def phase_report():
    df = pd.read_parquet(PER_CLIP_RESULTS_PATH)
    print(f"loaded {len(df):,} per-clip results", flush=True)

    implicit_known = df["endfiller"].notna()
    implicit_incomplete = (~df["y_true"]) & (df["endfiller"] == False) & implicit_known  # noqa: E712
    n_real_unknown = int((~df["synthetic"] & ~implicit_known).sum())

    outputs = EvalOutputs(
        ids=df["id"].tolist(), y_true=df["y_true"].values.astype(int),
        y_prob=df["prob_temp_scaled"].values, language=df["language"].tolist(),
        dataset=df["dataset"].tolist(), synthetic=df["synthetic"].tolist(),
        implicit_incomplete=implicit_incomplete.tolist(),
    )
    report = full_report(outputs, THRESHOLD_TEMP_SCALED)
    report["n_total"] = len(df)
    report["n_real_audio_implicit_incomplete_unknown_no_asr"] = n_real_unknown
    report["temperature"] = TEMPERATURE
    report["threshold_temp_scaled"] = THRESHOLD_TEMP_SCALED
    report["checkpoint"] = str(FROZEN_CHECKPOINT_DIR / "checkpoint.pt")
    report["note_implicit_incomplete"] = (
        "implicit_incomplete slice covers SYNTHETIC clips only (ground-truth endfiller known); "
        f"{n_real_unknown} real-audio rows have unknown endfiller (no ASR run against the official "
        "test set) and are excluded from that slice, not folded in as 'not implicit'."
    )

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"saved {REPORT_PATH}")
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_source",)}, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["fetch_features", "infer", "report"])
    args = ap.parse_args()
    if args.phase == "fetch_features":
        phase_fetch_features()
    elif args.phase == "infer":
        if PER_CLIP_RESULTS_PATH.exists():
            print(f"ERROR: {PER_CLIP_RESULTS_PATH} already exists -- the official test evaluation "
                  f"has already been run. Refusing to overwrite it (touched-once discipline). "
                  f"Delete it manually first if this was truly a bug-fix re-run, not a repeat "
                  f"evaluation.")
            sys.exit(1)
        phase_infer()
    elif args.phase == "report":
        phase_report()


if __name__ == "__main__":
    main()
