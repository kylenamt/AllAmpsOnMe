# Open-Amp3000

A general-purpose guitar amp model trained on data captured from [TONE3000](https://www.tone3000.com).

This project follows the [Open-Amp paper](https://arxiv.org/abs/2411.14972) (arXiv:2411.14972):
a single FiLM-conditioned TCN that emulates hundreds of amps at once, trained on audio rendered
through neural amp captures. Instead of the paper's original capture set, we source diverse
**amp-only (DI)** [neural amp modeler (NAM)](https://www.neuralampmodeler.com/) captures from the
TONE3000 community, each becoming one "device" the model learns to emulate.

It is **one Python package (`openamp`) and one CLI (`openamp`)** that runs the whole pipeline as a
single sequence of verbs:

```
acquire :  auth · discover · select · download · validate · dedup · finalize · status
corpus  :  corpus · subset · render · verify
emulate :  emulate · emulate-compare · emulate-demo
```

The package is grouped by purpose into subpackages (`core`, `dsp`, `acquire`, `corpus`,
`emulate`); see [docs/architecture.md](docs/architecture.md) for the role of every file.

Each capture is assigned a stable `device_id` at `finalize` time that becomes the embedding-table
index used everywhere downstream — so the acquisition manifest is the contract the rest of the
pipeline builds on.

## Setup

The project uses **conda** (Python 3.10). Create the environment once and install everything:

```bash
conda create -n open-amp3000 python=3.10
conda activate open-amp3000
pip install -e ".[dev]"          # all deps + pytest
cp .env.example .env             # add your TONE3000 publishable key (OPENAMP_API_KEY)
```

> **NAM version:** `neural-amp-modeler>=0.13` is a hard requirement — it is the first release that
> can parse an **A2** capture (the canonical architecture; see
> [docs/pipeline.md](docs/pipeline.md)). On older versions every A2 capture fails validation as
> `load_or_render_failed`.

Pipeline tunables live in [`configs/openamp.yaml`](configs/openamp.yaml) and per-run emulator
configs under [`configs/emulate/`](configs/emulate/); API access and the data root come from the
environment / `.env` (`OPENAMP_*`). See [`docs/pipeline.md`](docs/pipeline.md) for the full
operator guide (prerequisites, resume semantics, NFS notes).

## The pipeline

Run `openamp --help` for the full command list, or `openamp <command> --help` for any one.

**1 — Acquire** diverse DI amp captures from TONE3000 into `data/`:

```bash
openamp auth                     # one-time browser OAuth authorization
openamp discover                 # build the candidate model pool -> candidates.parquet
openamp select                   # diversity selection -> manifest.parquet
openamp download                 # download the selected .nam files
openamp validate                 # load + render-probe each capture
openamp dedup                    # drop exact-hash + near-duplicate (ESR) captures
openamp finalize                 # top up to quota, assign device_ids -> final manifest
openamp status                   # status counts at any time
```

**2 — Corpus + render** the clean input through each device (offline, once, on GPU):

```bash
openamp corpus                   # raw sources -> 48 kHz clean corpus + splits + clip grid
openamp subset --size 450        # (optional) trim to a diverse, gain-balanced device subset
openamp render --devices 0-2     # pilot a few devices first...
openamp render --devices @data/manifests/render_subset.txt   # ...then the full job (resumable)
openamp verify                   # completeness + sanity + QA + report -> devices_final.parquet
```

**3 — Emulate** all devices with one FiLM-conditioned TCN (every size is a config knob):

```bash
openamp emulate --config configs/emulate/paper.yaml --overfit   # sanity: overfit one batch
openamp emulate --config configs/emulate/paper.yaml             # full run -> results/emulate/paper/
openamp emulate-compare results/emulate/*                       # test-split rows -> comparison.csv
openamp emulate-demo results/emulate/paper                      # clean/target/pred listening WAVs
```

Every stage reads/writes the parquet manifests under `data/manifests/` and is independently
re-runnable (idempotent), so you can stop and resume at any point.

## Repository layout

```
src/openamp/            the one pipeline package
  cli.py                the one CLI (all commands, in run order)
  core/                 config, constants, manifest I/O, utils, diversity selection
  dsp/                  audio I/O + metrics + probe (audio.py); NAM adapter (nam.py)
  acquire/              TONE3000 acquisition (auth, client, discover, select, ...)
  corpus/               clean corpus build + subset + GPU rendering + verify
  emulate/              FiLM-TCN emulator: tcn.py, dataset.py, train.py, evaluate.py
configs/openamp.yaml    pipeline tunables (seed, acquisition, corpus, render)
configs/emulate/        per-run emulator configs (architecture + training knobs)
docs/                   architecture.md, pipeline.md (operator guide), history/ (design specs)
notebooks/              emulator listening demo
tests/                  pytest suite
data/                   captures + clean corpus + renders + manifests (git-ignored)
results/                reports + emulator runs (git-ignored except reports)
```

## Data, licensing & etiquette

Downloaded `.nam` capture files, OAuth tokens, and API keys are **never committed** (`data/` and
secrets are git-ignored) and capture files are **never redistributed** — only per-capture metadata
and license terms are recorded in the manifest. Acquisition uses the official TONE3000 API only
(no scraping), with a client-side rate limiter kept on at all times.

The clean corpus and rendered audio are **derived data** from the same license-restricted captures —
they stay under the git-ignored `data/` tree and are **never published**. Only metadata manifests
and the metadata-only report (`results/render_report.md`) are tracked.

## Development

```bash
python -m pytest              # run the test suite
python -m openamp --help      # the CLI, also available as the `openamp` console script
```

A few tests in `test_validate.py` / `test_finalize.py` are skipped unless small `.nam` fixtures
exist under `tests/fixtures/` — see [`tests/fixtures/README.md`](tests/fixtures/README.md).

## License

See [LICENSE](LICENSE).
