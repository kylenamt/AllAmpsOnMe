# `oa3k` — Phase 2: clean corpus + offline rendering pipeline

The `data` package (CLI `oa3k`) builds the clean guitar-input corpus, renders it through every
validated capture from Phase 1 on GPU (offline, once), and exposes PyTorch `Dataset` classes for
downstream training. It consumes the Phase 1 output manifest and never republishes any audio.

The full requirement spec lives in [docs/phase2-spec.md](docs/phase2-spec.md). For the wider
project context see the [README](README.md); for Phase 1 acquisition see [T3K.md](T3K.md).

> **Package name.** Per the spec (§6) the datasets live at `src/data/datasets.py`, so the Phase 2
> package is `data` (importable as `import data`). It is distinct from the runtime output tree
> `./data/` (git-ignored) — the former is code under `src/`, the latter is where outputs land.

---

## What it does

```
oa3k corpus prepare   # raw EGDB + NAM sweep -> 48 kHz clean corpus + splits + clip grid
oa3k render run       # GPU render every device (resumable; --devices, --dry-run)
oa3k render verify    # completeness + sanity checks + QA exports + report -> devices_final
oa3k dataset smoke    # DataLoader smoke test for both Dataset classes
```

Every stage is **idempotent/resumable**, **deterministic given the seed**, and writes manifests
atomically — the same contract as Phase 1.

---

## Setup

### 1. Install the render stack

Phase 2 needs a heavy GPU stack (kept out of the fast Phase 1 install):

```bash
pip install -e ".[phase2]"        # torch(+CUDA), torchaudio, neural-amp-modeler, soundfile, typer, matplotlib
# or: pip install -r requirements-phase2.txt
```

Use a **CUDA-enabled** `torch` build matching your machine; `oa3k render run` requires a GPU.
As in Phase 1, avoid `pytorch-lightning` 2.6.2/2.6.3 (supply-chain incident) — the pins cap below.

### 2. Provide the raw inputs (manual, one-time)

Two sources are downloaded by hand and placed under the data root (`T3K_DATA_DIR`, default `./data`):

| Source | Place at | Notes |
|---|---|---|
| **EGDB** direct-input (DI) WAVs | `data/raw/egdb/` | Download from the EGDB GitHub release; use **only the clean DI audio**, not the amp-rendered versions. Files < 4 s or with >1% clipping are auto-rejected. |
| **NAM sweep** `v3_0_0.wav` | `data/raw/nam_sweep/v3_0_0.wav` | The standardized reamp signal from the neural-amp-modeler training resources. Always appended on top of the minutes budget and sent to the **train** split only. |

If either is missing, `oa3k corpus prepare` prints exactly where to put it and stops.

### 3. Phase 1 must be finalized

`render run` reads `data/manifests/manifest.parquet` (Phase 1's final manifest: one row per
`device_id` with `file_path`, `architecture`, `sample_rate`) and the `.nam` capture files under
`data/captures/`. Run Phase 1 through `t3k finalize` first (see [T3K.md](T3K.md)).

---

## Usage

### Config

All tunables live in [configs/phase2.yaml](configs/phase2.yaml): `minutes_total` (corpus size),
`split_ratios`, `seed`, `chunk_seconds`, `sample_rate`, `clip_seconds`, `output_format`. All
randomness derives from the single `seed`. Pass a different file with `--config path.yaml`.

### 1. Corpus

```bash
oa3k corpus prepare
```

Converts every source to 48 kHz mono float32, normalizes to −18 dBFS RMS then limits true peaks
to −1 dBFS, selects EGDB files (seeded) up to `minutes_total`, splits **by file** ~80/10/10 (sweep
→ train; test → EGDB only), and writes:

- `data/clean/{split}/{file_id}.flac` (24-bit)
- `data/manifests/corpus.parquet` — `file_id, split, source, orig_path, duration_s, applied_gain_db, sha256`
- `data/manifests/clips.parquet` — the fixed 2 s clip grid (`clip_id, file_id, split, start_sample, num_samples`)

### 2. Render (GPU)

**Pilot first** (spec §4.3): render a few devices, verify, then launch the full job.

```bash
oa3k render run --devices 0-2      # pilot
oa3k render verify                 # inspect results/phase2_qa + report
oa3k render run                    # full job (resumable; re-run to continue after a crash)
```

Rendering is **whole-file, then sliced** — each file is streamed through the model in
`chunk_seconds` windows carrying the model's receptive field R as left-context and discarding the
warmup region, so the result is **bit-identical to a single full-file forward pass** (no
clip-boundary transients). Per device, a single output scalar pins the global peak at 0.999.
Outputs: `data/renders/{device_id:04d}/{file_id}.flac` + `data/manifests/renders.parquet`. A device
whose files all exist with matching rows is skipped. The job refuses to start if free disk is below
1.3× the estimate (~75–110 GB for ~400 devices).

### 3. Verify

```bash
oa3k render verify
```

Checks completeness (+ hashes), signal sanity (no NaN/Inf, not silent, not a pass-through via
`ESR(input, output) > 0.001`), and alignment; exports clean/render WAV + spectrogram PNG pairs to
`results/phase2_qa/`, writes `results/phase2_report.md`, and emits
`data/manifests/devices_final.parquet` (verified devices only — failures are excluded). If the
final count drops below 400 it is reported; top up by re-running Phase 1 finalize with more
candidates, then render only the new device_ids.

### 4. Datasets

```python
from data.config import load_config
from data.datasets import EmulationDataset, ContrastivePairDataset

cfg = load_config()
emu = EmulationDataset("train", cfg, pairs_per_epoch=200_000)   # (clip x device) -> input/target/device_id/output_scale
con = ContrastivePairDataset("train", cfg)                       # two crops of the same device -> view1/view2/device_id
```

Both stream 2 s clips via `soundfile` seek-reads. Smoke-test the DataLoader path:

```bash
oa3k dataset smoke --split train --workers 4
```

---

## Outputs & manifests

| Manifest | Written by | Contents |
|---|---|---|
| `corpus.parquet` | `corpus prepare` | one row per clean source file + applied gain + hash |
| `clips.parquet` | `corpus prepare` | the fixed 2 s clip grid (shared by clean + rendered) |
| `renders.parquet` | `render run` | one row per (device, file): path, scale, RMS/peak, hashes, status |
| `devices_final.parquet` | `render verify` | verified devices + render stats (the Phase 3 contract) |

Renders and the clean corpus are **derived from license-restricted captures** and live under the
git-ignored `data/` tree; QA audio exports under `results/phase2_qa/` are git-ignored too. Only the
metadata-only `results/phase2_report.md` is tracked.

---

## Deviations & implementation notes (spec §8.6, §9)

- **CLI framework:** the spec suggests `typer`; the CLI is a thin typer app (`data/cli.py`). All
  logic is in the library modules, so the framework is immaterial to behavior.
- **Heavy-dep isolation:** `torch`/`neural-amp-modeler` are used only in `data/nam_render.py` and
  `soundfile`/`torchaudio` only in `data/audio_io.py`, always via local imports. The rest of the
  package imports on a bare numpy/pandas stack, so the core logic is unit-tested without a GPU
  (mirrors Phase 1's `t3k.nam_backend` isolation). Model construction reuses Phase 1's
  version-tolerant loader (`t3k.nam_backend._import_nam_model`); DSP metrics reuse `t3k.probe`.
- **Receptive field:** read from the NAM net/config when available, else a safe large fallback —
  over-estimating R only costs a little context compute, never correctness (`data/nam_render.py`).

---

## Testing

The signal-processing and bookkeeping core is covered by `pytest` **without** the GPU stack:

```bash
pip install -e ".[dev]"
pytest tests/test_render.py tests/test_corpus.py tests/test_verify.py \
       tests/test_datasets.py tests/test_phase2_config.py -q
```

Key proof: `test_render.py` asserts the chunked render is **bit-identical** to a full forward on a
numpy FIR stand-in across receptive fields and chunk sizes. Tests that need torch/soundfile
(`dataset smoke`, tensor `__getitem__`) are `importorskip`-guarded and run only where those deps
are installed.
