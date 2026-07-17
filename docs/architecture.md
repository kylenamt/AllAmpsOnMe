# Open-Amp3000 — architecture

This document describes the `openamp` package after the purpose-based
reorganization: what the pipeline does end to end, and the role of every file.

It follows the [Open-Amp paper](https://arxiv.org/abs/2411.14972): acquire a
diverse set of neural amp captures from [TONE3000](https://www.tone3000.com),
render a clean guitar corpus through each captured amp, and train a one-to-many
FiLM-conditioned TCN that emulates every captured amp from a shared model plus a
per-device embedding.

The whole thing runs as one process behind a single CLI (`openamp <verb>`). Each
stage reads and writes files (parquet manifests + audio), so stages are
resumable and can be run independently in order.

---

## Package layout

```
src/openamp/
  cli.py              # the one CLI: a thin verb per pipeline stage
  __main__.py         # `python -m openamp` entry point

  core/               # shared foundation (pure python; no heavy deps)
    config.py         #   the one Config: env + yaml -> paths + knobs
    constants.py      #   fixed audio facts + vocabulary (splits, gain buckets, filenames)
    manifest.py       #   every parquet schema + atomic read/write
    util.py           #   sha256_file, utc_now
    selection.py      #   diversity-selection engine (caps + gain floor + round-robin)

  dsp/                # audio + model backend (heavy: soundfile / torch / NAM)
    audio.py          #   audio I/O, level/ESR metrics, the deterministic validation probe
    nam.py            #   neural-amp-modeler adapter (load .nam, run inference)

  acquire/            # get amp captures from the TONE3000 API (the only API-facing group)
    auth.py           #   OAuth 2.0 + PKCE standard flow, token storage
    client.py         #   rate-limited, retrying HTTP client + typed endpoints
    catalog.py        #   static heuristics: search terms, gain keywords, make aliases
    normalize.py      #   defensive normalization of live API objects -> candidate rows
    discover.py       #   stage: build the candidate model pool -> candidates.parquet
    select.py         #   stage: diversity selection -> manifest.parquet (selected)
    download.py       #   stage: download each capture's .nam (resumable, checksummed)
    validate.py       #   stage: load + render-probe each capture (finite/audible/non-exploding)
    dedup.py          #   stage: drop exact-hash + ESR near-duplicate captures
    finalize.py       #   stage: assign stable device_ids -> final manifest + rejected
    migrate.py        #   stage: re-point finalized devices at their A2 captures (device_ids kept)

  corpus/             # build the clean corpus and render it through every device
    build.py          #   stage: EGDB + sweep -> 48 kHz clean FLAC + corpus/clips manifests
    subset.py         #   stage: pick a diverse, gain-balanced device subset to render
    render.py         #   stage: GPU-render the corpus through each device (resumable)
    verify.py         #   stage: completeness + signal sanity + QA exports -> devices_final

  emulate/            # one-to-many amp foundation models (FiLM-conditioned)
    tcn.py            #   fully-parametric FiLM-TCN + per-device embedding table
    wavenet.py        #   the NAM A2 capture topology, FiLM-conditioned (same contract)
    models.py         #   arch selection: `emulate.arch` -> build_model()
    dataset.py        #   clean-in / render-out training pairs (real left-context warmup)
    train.py          #   the one training script (pre-emph ESR + MRSTFT) + sanity ladder
    evaluate.py       #   size-comparison harness (-> comparison.csv) + demo export
```

`core` and (mostly) `acquire` load on a bare numpy/pandas stack; `dsp` and
`emulate` pull torch / soundfile / NAM (`emulate` also uses auraloss). `cli.py`
imports each stage module lazily inside its command body, so `openamp --help`
and metadata-only verbs stay fast without loading torch.

---

## The pipeline, stage by stage

Run order (each is one `openamp` verb). "Reads" / "Writes" are files under the
`data_dir` (default `./data`) and `results_dir` (default `./results`); paths are
all derived from `openamp/core/config.py`.

### A. Acquisition — `openamp.acquire`

| Verb | Module | Reads | Writes |
|---|---|---|---|
| `auth` | `auth.py` | — | `./.openamp_tokens.json` (OAuth tokens) |
| `discover` | `discover.py` | TONE3000 API | `manifests/candidates.parquet` |
| `select` | `select.py` | `candidates.parquet` | `manifests/manifest.parquet` (status=`selected`) |
| `download` | `download.py` | `manifest.parquet` | `captures/{tone_id}/{model_id}.nam`, manifest updated |
| `validate` | `validate.py` | downloaded `.nam` | `{model_id}.probe.npy`, manifest status=`validated`/`rejected` |
| `dedup` | `dedup.py` | validated set + probes | manifest status=`duplicate` |
| `finalize` | `finalize.py` | `manifest.parquet` | final `manifest.parquet` (with `device_id`) + `manifests/rejected.parquet` |
| `migrate-a2` | `migrate.py` | final `manifest.parquet` | A2 `.nam` per device, manifest rewritten in place (`device_id` preserved) + `manifests/manifest.a1.parquet` backup |

The acquisition manifest is a single DataFrame carried from discovery through
finalize; each stage mutates `status` (+ its own columns) and writes it back, so
re-running a stage is a no-op for rows already past it. `finalize` assigns the
**stable `device_id`** (rank by make, model, model_id) — the embedding-table
index that must never change once training has seen it.

`discover`/`download`/`validate`/`dedup`/`finalize` are heuristics over live API
data; `catalog.py` + `normalize.py` hold all the fuzzy classification (gain
bucket, make, amp-vs-pedal) so it is auditable in one place. This group is
effectively **frozen** — the durable asset is `manifest.parquet` + `captures/`,
not the acquisition code.

### B. Corpus + rendering — `openamp.corpus`

| Verb | Module | Reads | Writes |
|---|---|---|---|
| `corpus` | `build.py` | `raw/egdb/*.wav`, `raw/nam_sweep/v3_0_0.wav` | `clean/{split}/*.flac`, `manifests/corpus.parquet`, `manifests/clips.parquet` |
| `subset` | `subset.py` | final `manifest.parquet` | `manifests/render_subset.txt` (device id list) |
| `render` | `render.py` | `clean/`, `captures/`, `render_subset.txt` | `renders/{device:04d}/*.flac`, `manifests/renders.parquet` |
| `verify` | `verify.py` | `renders.parquet`, `corpus.parquet` | `manifests/devices_final.parquet`, `results/qa/*`, `results/render_report.md` |

- **build**: scans EGDB direct-input WAVs (+ the NAM standardized sweep),
  resamples to 48 kHz mono, level-normalizes to −18 dBFS RMS with −1 dBFS peak
  headroom, selects to a minutes budget, and splits **by file** (never by clip)
  so no clean source leaks across train/val/test. Emits the clean FLACs plus the
  2 s clip grid.
- **subset**: the acquisition manifest can hold more devices than we want to
  render (it keeps every validated capture). This picks a diverse, gain-balanced
  subset via the shared `core/selection.py` engine (per-tone/creator/make-model
  caps + a per-gain-bucket floor). The manifest is never modified.
- **render**: loads each device's `.nam` on the GPU and streams every clean file
  through it **whole-file, chunked with left-context** (bit-identical to a single
  forward pass, no clip-boundary transients), then scales/encodes/hashes the
  output. Resumable and idempotent per device.
- **verify**: completeness + per-signal sanity (non-finite / silent /
  pass-through / misaligned), writes the final verified device list, and exports
  a few clean/render WAV + spectrogram QA pairs.

### C. Emulation foundation model — `openamp.emulate`

| Verb | Module | Reads | Writes |
|---|---|---|---|
| `emulate` | `train.py` | `renders/`, `renders.parquet`, `corpus.parquet` | `results/emulate/<name>/{checkpoint,last}.pt`, `config.yaml`, `metrics.json`, `train_log.csv`, `embedding.pt` |
| `emulate-compare` | `evaluate.py` | run dirs + test renders | `results/emulate/comparison.csv` |
| `emulate-demo` | `evaluate.py` | a run + test renders | `results/emulate/demos/*.wav` |

One FiLM-conditioned model emulates **all**
devices, steered by a learnable per-device embedding (FiLM scale+shift at every
layer). Two architectures share the same contract and are selected by
`emulate.arch` (`models.build_model`): the fully-parametric FiLM-TCN (`tcn.py` —
`blocks`, `layers_per_block`, `channels`, `kernel_size`, `dilation_growth`,
`embedding_dim` are all plain config knobs) and the FiLM-WaveNet (`wavenet.py` —
the exact NAM A2 topology every corpus capture uses, 23 dilated layers /
receptive field 6347, with the device FiLM at the schema's pre-activation hook),
so a sweep is copy-a-config-and-change-numbers with no code change. `dataset.py`
serves clean-in / render-out pairs with the receptive field of **real left-context**
prefixed for warmup (loss only on the warmed region). `train.py` optimizes
pre-emphasized ESR + multi-resolution STFT (auraloss) with Adam + reduce-on-plateau,
runs the sanity ladder (`--overfit`, `--limit-devices`), and saves the embedding
table + its manifest hash separately (Phase 5 extends it). `evaluate.py` appends
one test-split row per run to `comparison.csv` — the architecture-exploration
deliverable — and exports clean/target/prediction listening demos.

---

## Shared foundation — `openamp.core` and `openamp.dsp`

These carry no pipeline logic; every stage imports them.

- **core/config.py** — one `Config` dataclass for the whole run. API access
  (key, base URL, token path, rate limit) comes from the environment / `.env`;
  pipeline knobs (`corpus`/`render`/`emulate`) come from `configs/openamp.yaml`
  (per-run emulator files under `configs/emulate/` override just `emulate:`);
  every path derives from one `data_dir` + one `results_dir`, and all randomness
  derives from one `seed`. Defaults live once, on the dataclass fields —
  `load_config` only overrides what a source provides.
- **core/constants.py** — fixed audio facts (sample rate, clip length,
  normalization/verification thresholds) and shared vocabulary: split labels,
  **gain-bucket names** (`clean`/`crunch`/`high_gain`), render statuses, and
  manifest filenames. Kept in one place so the clip grid, render slicing, and
  datasets agree by construction.
- **core/manifest.py** — the schema and atomic parquet I/O for every manifest
  (candidates, acquisition, corpus, clips, renders, devices_final).
- **core/util.py** — `sha256_file`, `utc_now`.
- **core/selection.py** — the schema-agnostic diversity-selection engine
  (`two_phase_select`: gain-floor Phase A + brand round-robin Phase B, under
  per-tone/creator/make-model caps). Shared by `acquire/select.py` (over
  candidate models) and `corpus/subset.py` (over the finalized manifest); each
  passes in its own id column and tie-break key.
- **dsp/audio.py** — the one place that depends on `soundfile`/`torchaudio`:
  read/write/seek-read, resampling, level (RMS/peak dBFS) and error (ESR)
  metrics, and the deterministic 5 s validation probe (sweep + silence + DI)
  used by acquisition `validate`/`dedup`.
- **dsp/nam.py** — the one place that depends on the `neural-amp-modeler` package
  and its version-varying internal API: parse a `.nam`, reconstruct the net, and
  run inference (CPU probe for `validate`, GPU forward + receptive field for
  `render`). Isolated so NAM version drift touches only this file. Also owns the
  two capture architectures: an **A2** `SlimmableContainer` resolves to its
  full-width variant, and a legacy **A1** export is upgraded to the current schema
  on load. Requires NAM >= 0.13.

---

## Data & result artifacts

Under `data_dir` (default `./data`):

```
raw/egdb/*.wav                 clean guitar DI source (EGDB)
raw/nam_sweep/v3_0_0.wav       NAM standardized reamp sweep
captures/{tone_id}/{model_id}.nam   downloaded amp captures (+ .probe.npy)
clean/{train,val,test}/*.flac  normalized 48 kHz clean corpus
renders/{device:04d}/*.flac    each device's rendered corpus
manifests/
  candidates.parquet           discovery pool
  manifest.parquet             acquisition manifest (final = accepted devices)
  rejected.parquet             excluded captures + reasons
  corpus.parquet, clips.parquet   clean corpus + 2 s clip grid
  renders.parquet              one row per (device, file) render
  devices_final.parquet        verified devices + render stats
  render_subset.txt            device ids chosen for rendering
```

Under `results_dir` (default `./results`):

```
qa/                            clean/render WAV + spectrogram QA pairs
render_report.md               verify-stage summary
emulate/<name>/                one emulation run (checkpoint, config, metrics, curves, embedding)
emulate/comparison.csv         one test-split row per emulation run (size-sweep deliverable)
emulate/demos/*.wav            clean/target/prediction listening demos
```

## Conventions

- **Determinism**: every stage is seeded from `config.seed`; re-running a stage
  with the same inputs reproduces its output. Selection and splits use a total
  order where possible so results don't depend on RNG.
- **Resumability**: stages check for existing outputs and skip completed work
  (per-row for manifests, per-device for renders, per-file for the corpus).
- **Atomic writes**: manifests, audio, and checkpoints are written to a temp
  sibling then renamed, so a crash never leaves a half-written file that resume
  logic would mistake for complete.
