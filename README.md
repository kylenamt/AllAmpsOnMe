# AllAmpsOnMe

A general-purpose guitar amp model trained on data captured from [TONE3000](https://www.tone3000.com).

Following the [Open-Amp paper](https://arxiv.org/abs/2411.14972), a single FiLM-conditioned TCN
emulates hundreds of amps at once. Each amp is a diverse **amp-only (DI)**
[NAM](https://www.neuralampmodeler.com/) capture from the TONE3000 community — one "device" the
model learns to emulate.

Everything lives in one Python package and CLI (`openamp`) that runs the whole pipeline as a
sequence of verbs:

```
acquire :  auth · discover · select · download · validate · dedup · finalize · status
corpus  :  corpus · subset · render · verify
emulate :  emulate · emulate-compare · emulate-demo
```

See [docs/architecture.md](docs/architecture.md) for the code layout and
[docs/pipeline.md](docs/pipeline.md) for the full operator guide.

## Setup

The project uses **conda** (Python 3.10):

```bash
conda create -n open-amp3000 python=3.10
conda activate open-amp3000
pip install -e ".[dev]"      # deps + pytest
cp .env.example .env         # add your TONE3000 key (OPENAMP_API_KEY)
```

> **NAM version:** `neural-amp-modeler>=0.13` is required — it is the first release that can parse
> an **A2** capture. On older versions every A2 capture fails validation.

Tunables live in [`configs/openamp.yaml`](configs/openamp.yaml) (pipeline) and
[`configs/emulate/`](configs/emulate/) (per-run emulator); API access and the data root come from
`.env` (`OPENAMP_*`).

## Pipeline

Run `openamp --help` for the full command list, or `openamp <command> --help` for any one. Every
stage reads and writes the parquet manifests under `data/manifests/` and is idempotent, so you can
stop and resume at any point.

```bash
# 1 — Acquire diverse DI amp captures from TONE3000 into data/
openamp auth        # one-time browser OAuth
openamp discover    # build the candidate pool
openamp select      # diversity selection -> manifest
openamp download    # fetch the selected .nam files
openamp validate    # load + render-probe each capture
openamp dedup       # drop duplicate captures
openamp finalize    # assign device_ids -> final manifest

# 2 — Build the clean corpus and render it through each device (GPU)
openamp corpus                          # 48 kHz clean corpus + splits
openamp subset --size 450               # optional: gain-balanced device subset
openamp render --devices @data/manifests/render_subset.txt
openamp verify                          # completeness + QA -> devices_final.parquet

# 3 — Train one FiLM-TCN over all devices
openamp emulate --config configs/emulate/paper.yaml
openamp emulate-compare results/emulate/*     # test-split comparison
openamp emulate-demo results/emulate/paper    # listening WAVs
```

## Data & licensing

Downloaded `.nam` captures, OAuth tokens, and API keys are **never committed**, and captures are
**never redistributed** — only per-capture metadata and license terms are recorded in the manifest.
The clean corpus and rendered audio are derived data that stay under the git-ignored `data/` tree
and are never published. Acquisition uses the official TONE3000 API only (no scraping), rate-limited
at all times.

## Development

```bash
python -m pytest              # run the test suite
python -m openamp --help      # the CLI (also the `openamp` console script)
```

Some `test_validate.py` / `test_finalize.py` tests are skipped unless small `.nam` fixtures exist
under `tests/fixtures/` — see [`tests/fixtures/README.md`](tests/fixtures/README.md).

## License

See [LICENSE](LICENSE).
