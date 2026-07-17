# Pipeline operator guide

The whole pipeline is one package (`openamp`) and one CLI (`openamp`). Stages run in order; every
stage reads/writes the parquet manifests under `data/manifests/`, is **idempotent/resumable**, and is
**deterministic given the `seed`**. This guide covers prerequisites, what each stage does, resume
semantics, and troubleshooting. For the design rationale behind each stage see
[`history/`](history/); for the quick tour see the [README](../README.md).

```
acquire :  auth · discover · select · download · validate · dedup · finalize · status
corpus  :  corpus · subset · render · verify
emulate :  emulate · emulate-compare · emulate-demo
```

## Install & configure

```bash
conda activate open-amp3000        # Python 3.10 env
pip install -e ".[dev]"            # all deps + pytest
cp .env.example .env               # add OPENAMP_API_KEY
```

`torch` must be a **CUDA-enabled** build for `render` (GPU) and `emulate`; eval runs on CPU or GPU.

> **`neural-amp-modeler` must be >= 0.13** — it is the first release whose WaveNet parses the A2
> layer schema, so older versions cannot load an A2 capture at all. (0.13 also dropped the
> `pkg_resources` import, so the old `setuptools<81` cap is gone.) `pytorch-lightning` is pinned
> `==2.5.2`; avoid 2.6.2/2.6.3 (supply-chain incident).

### Configuration

All pipeline tunables live in [`configs/openamp.yaml`](../configs/openamp.yaml) — top-level `seed` /
`sample_rate`, then `corpus:` / `render:` sections; emulator knobs live in per-run files under
[`configs/emulate/`](../configs/emulate/). Pass a different file to the corpus/render/emulate
commands with `--config path.yaml`. API access and the data root come from the environment / `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAMP_API_KEY` | — | **Required for acquisition.** TONE3000 publishable key (`t3k_pub_…`). |
| `OPENAMP_TOKEN_PATH` | `./.openamp_tokens.json` | OAuth token cache (chmod 600). |
| `OPENAMP_BASE_URL` | `https://www.tone3000.com/api/v1` | API base. |
| `OPENAMP_REDIRECT_URI` | `http://localhost:3001` | OAuth redirect (localhost allowed in dev). |
| `OPENAMP_DATA_DIR` | `./data` | Output root (git-ignored). |
| `OPENAMP_RATE_LIMIT_RPM` | `80` | Client-side cap (server allows 100). |
| `OPENAMP_SEED` | `1234` | Overrides the yaml `seed`. |

## 1 — Acquire (TONE3000)

```bash
openamp auth        # OAuth standard flow (PKCE); verifies GET /user, prints username
openamp discover    # 2k–4k candidate models          -> data/manifests/candidates.parquet
openamp select      # deterministic diversity pick     -> data/manifests/manifest.parquet
openamp download    # fetch .nam (Bearer, resumable)   -> data/captures/{tone_id}/{model_id}.nam
openamp validate    # NAM load + 5 s render probe
openamp dedup       # drop exact-hash + ESR near-duplicates
openamp finalize    # top up to ≥400, assign device_ids -> manifest.parquet + rejected.parquet
openamp status      # status counts anytime
```

`auth` is interactive: it opens a browser once, then caches tokens (auto-refreshed on 401/expiry).
On a headless machine use `openamp auth --headless` (prints the URL, you paste the redirect back).

- **discover** — several `/tones/search` passes (trending/downloads/newest sorts + a brand keyword
  sweep), fetches each tone's `/models`, keeps **amp-only NAM** captures of the configured
  architecture (`acquire.architecture`, default **A2**), excludes cab/IR/full-rig/pedal/bass, dedupes
  by model id.

  > The architecture filter must be passed to **both** `/tones/search` **and** `/models`. It is not
  > inherited: `/models` defaults to A1, so passing it only to the search silently harvests A1
  > captures out of tones the search selected for having A2. This is exactly how the original 844-device
  > corpus ended up entirely A1.
- **select** — hard filters (DI amp, usable license, has `model_url`) → caps (≤2/tone, ≤8/creator,
  ≤6 per make-model) → gain minimums (≥20% each of clean/crunch/high-gain) → greedy brand round-robin,
  tie-broken by downloads. Fully deterministic given candidates + seed.
- **download** — Bearer-authenticated, streamed, checksummed (`sha256`), atomic, resumable. A
  permanently failing A2 model falls back to the tone's A1 sibling (`architecture_fallback`).
- **validate** — parse JSON, load with `neural-amp-modeler`, push a fixed 5 s probe (2 s log sweep +
  1 s silence + 2 s DI) through the model and assert output is finite / audible / non-exploding /
  quiet-in-silence. Probe output saved as `{model_id}.probe.npy` for dedup. A2 load failures retry
  via the A1 sibling if a client is available.

### Capture architectures: A1 vs A2

**A2 is canonical.** An A2 export is not a bare net: it is a `SlimmableContainer` holding several
*width variants* of the same capture ([Slimmable NAM](https://arxiv.org/abs/2511.07470)) — a runtime
CPU/quality dial, where `max_value` is the variant's width fraction. Every variant models the same
device, the narrow ones less accurately, so offline rendering always resolves the container to its
**full-width (`max_value: 1`) variant**. A1 is a plain WaveNet in a pre-0.7 schema that current NAM
no longer parses; `dsp/nam.py` upgrades it on load (verified bit-identical to the output of the NAM
version the A1 corpus was rendered with), so both architectures stay loadable.

- **migrate-a2** — one-off: re-points the **finalized** devices at their A2 captures, rewriting
  `manifest.parquet` in place and backing the old one up to `manifest.a1.parquet`. It runs on the
  final manifest rather than through select/finalize so `device_id` is never reassigned — the
  rendered audio is addressed by `device_id`, and re-deriving those ids would orphan 106 GB of it.
  Pairing is by **exact model name**: TONE3000 did not retrain every model on multi-model tones, and
  the models under one tone are the same amp at different gain settings, so an unmatched device keeps
  its A1 capture (flagged `architecture_fallback`) rather than being paired by similarity.
  Existing renders are *not* invalidated — they came from A1 captures and `renders.parquet` records
  the `nam_sha256` each one used, so the provenance stays truthful; only future renders pick up A2.
- **dedup** — exact `sha256` duplicates dropped globally; within each (make, model) group, probe
  outputs with ESR < 0.01 collapse to the higher-download capture.
- **finalize** — if fewer than 400 remain, loops back through select/download/validate/dedup for
  replacements (needs an API key; `--no-top-up` disables), then assigns the stable, zero-padded
  `device_id` (the embedding-table index — **never changes once assigned**) and writes
  `manifest.parquet` (accepted) + `rejected.parquet` (with reasons).

## 2 — Corpus + render

### Raw inputs (manual, one-time)

Place two sources under the data root (`OPENAMP_DATA_DIR`, default `./data`):

| Source | Place at | Notes |
|---|---|---|
| **EGDB** direct-input WAVs | `data/raw/egdb/` | Clean DI audio only (not the amp-rendered versions). Files < 4 s or > 1% clipping are auto-rejected. |
| **NAM sweep** `v3_0_0.wav` | `data/raw/nam_sweep/v3_0_0.wav` | Standardized reamp signal; appended on top of the minutes budget, sent to the **train** split only. |

If either is missing, `openamp corpus` prints exactly where to put it and stops. Acquisition
(`openamp finalize`) must have produced `data/manifests/manifest.parquet` and the `.nam` files first.

```bash
openamp corpus                       # raw sources -> 48 kHz clean corpus + splits + clip grid
openamp subset --size 450            # (optional) diverse, gain-balanced device subset
openamp render --devices 0-2         # pilot a few devices
openamp verify                       # inspect results/qa + render_report.md
openamp render --devices @data/manifests/render_subset.txt   # full job (resumable)
```

- **corpus** — converts every source to 48 kHz mono float32, normalizes to −18 dBFS RMS then limits
  true peaks to −1 dBFS, selects EGDB files up to `corpus.minutes_total`, splits **by file** ~80/10/10
  (sweep → train; test → EGDB only), and writes `data/clean/{split}/{file_id}.flac`,
  `corpus.parquet`, and the fixed 2 s `clips.parquet` grid.
- **subset** — the acquisition manifest often exceeds the ~400 target; this trims it to a diverse
  subset (per-tone/creator/make-model caps + a per-gain-bucket floor) and writes the device-id list to
  `render_subset.txt`. It does not touch the manifest; feed the list to `render --devices @<path>`.
- **render** — whole-file, then sliced: each file is streamed through the model in `render.chunk_seconds`
  windows carrying the receptive field R as left-context and discarding the warmup, so the result is
  **bit-identical to a single full-file forward** (no clip-boundary transients). One output scalar per
  device pins the global peak at 0.999. Writes `data/renders/{device_id:04d}/{file_id}.flac` +
  `renders.parquet`. A device whose files all exist with matching rows is skipped. Refuses to start if
  free disk is below 1.3× the estimate (~75–110 GB for ~400 devices). `--io-workers` overlaps the
  FLAC-encode/hash pass with the GPU forward (a large NFS win).
- **verify** — completeness (+ hashes), signal sanity (no NaN/Inf, not silent, not a pass-through via
  `ESR > 0.001`), and alignment; exports clean/render WAV + spectrogram pairs to `results/qa/`, writes
  `results/render_report.md`, and emits `devices_final.parquet` (verified devices only). If the count
  drops below 400 it is reported; grow the pool by raising `subset --size` (or the caps) and rendering
  the added device ids.

## 3 — Emulate (amp foundation model)

One FiLM-conditioned model emulates **all** rendered devices at once,
steered by a learnable per-device embedding. The architecture (`emulate.arch`: FiLM-TCN or the A2
FiLM-WaveNet) and every size are config knobs, so exploring variants is
copy-a-config-and-change-numbers. Runs consume the renders (`renders.parquet` +
`corpus.parquet`).

```bash
# Sanity ladder first (cheap): overfit one batch, then a ~1 h mini-run.
openamp emulate --config configs/emulate/paper.yaml --overfit
openamp emulate --config configs/emulate/paper.yaml --limit-devices 10

openamp emulate --config configs/emulate/paper.yaml       # full run -> results/emulate/paper/
openamp emulate --config configs/emulate/embed16.yaml     # a size-sweep variant
openamp emulate-compare results/emulate/*                 # test-split rows -> comparison.csv
openamp emulate-demo results/emulate/paper                # clean/target/pred WAVs (listening check)
```

- **Config** — all knobs live in the `emulate:` section; a per-run file under `configs/emulate/<name>.yaml`
  overrides just that section and the **run name is the file stem**. Shipped configs: `paper` (default),
  `embed16`/`embed256`, `channels32`, `deep3x8`, `wavenet_a2` (the capture-native architecture), and
  `baseline_{clean,crunch,high_gain}` (one-to-one `single_device` references for the one-to-many gap).
  Copy one, change numbers, rerun — no code changes.
- **Model** — both archs are causal with FiLM (per-channel scale+shift from the device embedding) at
  every layer. `film_tcn`: `blocks × layers_per_block` dilated convs (1021 samples / 21 ms receptive
  field for the paper default). `film_wavenet`: the exact NAM A2 WaveNet topology of the corpus's own
  captures (23 dilated layers, channels 8, 6347 samples / 132 ms), FiLM at the A2 schema's
  pre-activation hook. Training clips are prefixed with the receptive field of **real left-context** from the
  source file, so warmup uses actual audio and the loss covers only conditioned samples.
- **Device holdout** — `holdout_frac` (default 0.1) excludes a seeded ~10% of render-ok devices from
  training entirely, so Phase 5 can enroll them as truly unseen devices (the reference code's 90/10
  device split). Drawn once and persisted to `data/manifests/emulate_holdout.txt` — the file is the
  source of truth, so the holdout stays fixed even if more devices are rendered later. The held-out ids
  travel in every `checkpoint.pt` and `embedding.pt`; `single_device` runs ignore the holdout (that is
  how Phase 5 trains one-to-one references on held-out devices).
- **Training** — pre-emphasized ESR + multi-resolution STFT (auraloss), 1:1; Adam lr 5e-4, reduce-on-plateau,
  early-stopped at the val-ESR plateau; single GPU, AMP. Each run writes `results/emulate/<name>/`:
  `checkpoint.pt` (best) + `last.pt`, `config.yaml`, `metrics.json`, `train_log.csv`, and `embedding.pt`
  (the table + its manifest hash, saved separately for Phase 5). Every run reports `val_esr_shuffled`
  (val ESR with the embeddings permuted across devices) — it must be **worse** than `val_esr`, which
  proves the conditioning is doing real work.
- **Compare** — `emulate-compare` evaluates each run on the held-out **test** split and appends one row per
  run to `results/emulate/comparison.csv` (`run_name, params, receptive_field_ms, embedding_dim, channels,
  blocks_x_layers, test_ESR_mean, test_ESR_median, test_MRSL_mean, train_hours`). That CSV is the
  architecture-exploration deliverable; re-evaluating a run replaces its row.

> `auraloss` (multi-resolution STFT loss) is a dependency; `pip install -e .` pulls it in. GPU strongly
> recommended — the paper-default run is ~187 K steps; use `--limit-devices` / `--epochs` for quick passes.

## Testing

```bash
python -m pytest
```

Acquisition tests mock the API and inject the NAM renderer; corpus/render/dataset tests use tiny real
audio in `tmp_path`. A few `test_validate.py` / `test_finalize.py` cases are skipped unless small
`.nam` fixtures exist — see [`../tests/fixtures/README.md`](../tests/fixtures/README.md). Key proofs:
the chunked render is bit-identical to a full forward; the FiLM-TCN's receptive field matches its
formula and its output is causal; FiLM conditioning actually changes the output.

## TONE3000 API notes

Parsing is defensive — the live schema ([`github.com/tone-3000/api`](https://github.com/tone-3000/api))
differs from early field names, so missing/renamed/wrapped fields are tolerated and malformed records
logged and skipped (see `src/openamp/acquire/normalize.py`). Notably: `architecture_version` →
`A2`/`A1`; `downloads_count`/`favorites_count` fall back to short names; `make` derived from tone-level
`makes[]` or inferred from the title; `license` coerced to text (missing → `unknown`); `sample_rate`
defaults to 48 kHz. The one place tied to the exact `neural-amp-modeler` Python API is isolated in
`src/openamp/dsp/nam.py`; if your NAM version exposes a different entry point, adapt only that file.

**Etiquette:** official API only (no scraping, no auth bypass); the rate limiter stays on (~80 req/min)
even though the server allows 100; license terms are recorded per capture and capture files are never
redistributed.
