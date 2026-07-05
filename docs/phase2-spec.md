# Phase 2 Spec — Clean Corpus & Offline Rendering Pipeline

**Deliverable:** a Python package + CLI that (1) builds the clean guitar input corpus, (2) renders it through every validated capture from Phase 1 on GPU, producing the fixed training dataset on disk, and (3) exposes PyTorch `Dataset` classes for the two downstream tasks (one-to-many emulation, contrastive encoder).

This document is standalone, but it consumes Phase 1 outputs:
- `data/manifests/manifest.parquet` — ~400 validated captures, one row per device, with stable `device_id` (0…N-1), `file_path` to the `.nam` file, `architecture`, `make/model/gain_bucket`, license/attribution fields.
- `data/captures/{tone_id}/{model_id}.nam` — the capture files.

## 1. Context (read once)

We recreate the Open-Amp paper (arXiv:2411.14972). Downstream training needs pairs *(clean clip, device_id) → amp-processed clip* plus, for the contrastive encoder, two different clips processed by the same device. The paper rendered online during training; we render **offline, once**, because NAM models are heavy. Consequences the design must respect:

- Rendering must be **exact and artifact-free** (no clip-boundary transients, no receptive-field warmup glitches).
- The dataset is fixed, so the **train/val/test split is decided here** and never leaks.
- Everything **resumable and idempotent**: re-running skips completed devices/files.
- Nothing from this dataset is ever published (capture licenses forbid redistribution; renders count as derived data — keep private).

## 2. Environment

- Same repo/env as Phase 1: **Python 3.11 (pinned)**, Linux, `uv venv --python 3.11`.
- Additional deps: `torch` (CUDA build matching the machine), `torchaudio`, `neural-amp-modeler`, `soundfile`, `numpy`, `pandas`+`pyarrow`, `tqdm`, `typer`, `matplotlib` (QA plots only). Pin versions; avoid PyTorch Lightning 2.6.2/2.6.3 (supply-chain incident).
- GPU required for the render job (any modern NVIDIA card). Renders use float32; no mixed precision (bit-exactness and safety over speed).
- Global constants: `SAMPLE_RATE = 48_000` Hz, mono, `CLIP_SECONDS = 2.0` (96,000 samples).

## 3. Input corpus

### 3.1 Sources
1. **EGDB — direct-input (DI) electric guitar recordings.** Use only the clean DI audio, not the dataset's amp-rendered versions. Obtaining it involves a manual download (links on the EGDB GitHub page point to hosted archives); the operator places the archive at `data/raw/egdb/` and the tool unpacks/verifies it. The CLI must print clear instructions if the folder is missing.
2. **NAM standardized reamp signal `v3_0_0.wav`** (~3 min: chirps, noise bursts, guitar snippets) from the official neural-amp-modeler training resources; operator places it at `data/raw/nam_sweep/v3_0_0.wav`. Purpose: coverage of amplitude/frequency extremes.

### 3.2 Preparation (`corpus prepare`)
1. Scan `data/raw/egdb/` for DI WAVs; reject files < 4 s or with clipping (>1% samples at full scale).
2. Convert all sources to **48 kHz mono float32** (high-quality resampler, e.g. `torchaudio` kaiser-windowed sinc; EGDB is 44.1 kHz).
3. **Level normalization:** per source file, normalize to −18 dBFS RMS (guitar-DI-typical level NAM captures expect), then hard-limit true peaks to −1 dBFS. Record applied gain per file.
4. **Corpus size:** select source files (seeded shuffle) until reaching `minutes_total` (config, default **40 min**) of EGDB audio; the sweep is always appended on top of that budget.
5. **Split by source file** (never by clip): ~80/10/10 train/val/test by duration. Constraints: the **sweep goes to train only**; test must be realistic playing only.
6. Write processed files to `data/clean/{split}/{file_id}.flac` (24-bit) and `data/manifests/corpus.parquet`:
   `file_id, split, source (egdb|sweep), orig_path, duration_s, applied_gain_db, sha256`.
7. **Clip grid:** define non-overlapping 2 s clip boundaries per file (drop the trailing remainder); write `data/manifests/clips.parquet`: `clip_id, file_id, split, start_sample, num_samples`. The same grid is used for clean and rendered audio, so pairs align by construction.

## 4. Rendering (`render run`)

### 4.1 Strategy
Render **whole files, then slice** — never render 2 s clips independently. Per device:

1. Load the `.nam` model via the `neural-amp-modeler` Python package, move to GPU, `eval()`, `torch.inference_mode()`.
2. Determine the model's **receptive field** R from its config (A1/A2 are feedforward convolutional stacks, so R is finite and left-context conditioning is exact).
3. Stream each clean file through the model in chunks of ~10 s with **R samples of left-context** carried from the previous chunk; discard the context region of each output chunk. Result is bit-identical to a single full-file forward pass.
4. If the model's expected sample rate ≠ 48 kHz (rare; from Phase 1 metadata), resample in→model rate→out and flag `resampled=true`.

### 4.2 Output handling
- Compute the device's global output peak across all files. If peak > 0.999, apply one **per-device scalar** `output_scale = 0.999/peak` to all of that device's renders; record it. (ESR/MRSL training can undo it via the index; do not normalize per-file.)
- Write `data/renders/{device_id:04d}/{file_id}.flac` (24-bit, 48 kHz, mono).
- Append per (device, file) rows to `data/manifests/renders.parquet`:
  `device_id, file_id, split, path, output_scale, out_rms_db, out_peak_db, render_sha256, rendered_at, nam_sha256`.

### 4.3 Job mechanics
- Iterate devices in `device_id` order; a device is **done** when all its files exist with matching index rows and hashes — done devices are skipped (idempotent/resumable after crashes).
- Batch across files where GPU memory allows (stack equal-length chunks); otherwise sequential is fine — with A2's efficiency, expect well over 50× realtime, i.e. the full job (~400 devices × ~43 min ≈ 290 h of audio) in roughly a few hours, not days.
- On per-device failure (load error, NaN output): mark the device `render_failed` in a status column, continue, and report at the end. NaN/Inf in any chunk aborts that device.
- `--devices 0-9` / `--dry-run` flags for pilot runs. **Required first step: pilot on 3 devices**, run `render verify` (below), and only then launch the full job.

### 4.4 Storage estimate
~43 min × 400 devices at 24-bit/48 kHz mono ≈ **75–110 GB** as FLAC (plus ~0.4 GB clean corpus). `render run` must check free disk space up front and refuse to start if < 1.3× the estimate.

## 5. Verification (`render verify`)

Automated checks over `renders.parquet` + files:
1. Completeness: every (validated device × corpus file) pair present; hashes match.
2. Signal sanity per device: no NaN/Inf; not silent (RMS > −70 dBFS on EGDB files); not identical to input (ESR(input, output) > 0.001 — catches pass-through/broken models); silence maps to near-silence.
3. Alignment: for 5 random (device, file) pairs, confirm sample counts equal the clean file's; NAM forward is same-length, so any mismatch is a bug.
4. QA artifacts: for 8 random devices, export a 10 s clean/rendered pair as WAV + a spectrogram PNG to `results/phase2_qa/` for human listening/eyeballing.
5. Emit `results/phase2_report.md`: per-device table (RMS, peak, scale, status), split durations, storage totals, failures.

Devices that fail verification are excluded by removing them from the **final device list** `data/manifests/devices_final.parquet` (device_id, capture metadata, render stats). If the final count drops below 400, report it; topping up requires re-running Phase 1's finalize with more candidates, then rendering only the new devices (the pipeline must support this incremental case).

## 6. PyTorch Dataset classes (`src/data/datasets.py`)

Both classes read the manifests and use `soundfile` seek-reads so DataLoader workers stream clips without loading whole files.

1. **`EmulationDataset(split)`** — indexable over (clip × device) pairs.
   - `__getitem__` → `dict(input=FloatTensor[96000], target=FloatTensor[96000], device_id=long, output_scale=float)`.
   - Constructor options: `devices=` subset, `pairs_per_epoch=` (seeded random subsampling — the full product is ~400 × ~1,300 clips ≈ 500K pairs/epoch, usually subsampled).
2. **`ContrastivePairDataset(split)`** — for SimCLR training.
   - `__getitem__` → two **different random 2 s crops** (free augmentation, since full files are stored; crops may ignore the clip grid) from the same device, different positions and preferably different source files → `dict(view1, view2, device_id)`.
   - Optional light on-the-fly augmentation (random gain ±6 dB per view); off by default, flag-controlled.
3. Both must pass a smoke test: build a DataLoader (`num_workers=4`), pull 50 batches, assert shapes/dtypes/no-NaN, and log clips/s throughput on the target machine.

## 7. CLI summary

```
corpus prepare   # raw → 48kHz clean corpus + splits + clip grid
render run       # GPU render all devices (resumable; --devices, --dry-run)
render verify    # completeness + sanity checks + QA exports + report
dataset smoke    # DataLoader smoke test for both Dataset classes
```

Config in `configs/phase2.yaml`: `minutes_total`, split ratios, seed, chunk seconds, output format, paths. All randomness seeded from one config value.

## 8. Acceptance criteria

1. `corpus prepare` yields ≥ 40 min EGDB + sweep, 48 kHz mono, file-level splits, clip grid; deterministic given the seed.
2. Pilot (3 devices) passes `render verify` before the full run; full run completes resumably with ≥ 400 devices in `devices_final.parquet` (or a clear shortfall report).
3. `render verify` report shows zero NaN/silent/pass-through devices among finals; QA WAV/spectrogram exports exist.
4. Both Dataset classes pass `dataset smoke`; unit tests (`pytest`) cover chunked-vs-full-forward equivalence on a tiny model (must be bit-identical), clip alignment, scale bookkeeping, and split integrity (no file in two splits).
5. Re-running any command is a no-op when outputs are current.
6. README documents the manual EGDB/sweep download steps and any deviations discovered.

## 9. Out of scope

Model training (Phases 3–4), enrollment (Phase 5), full-rig captures, augmentation beyond the flags above, publishing any audio.