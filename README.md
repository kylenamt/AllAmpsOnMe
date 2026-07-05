# Open-Amp3000

A general-purpose guitar amp model trained on data captured from [TONE3000](https://www.tone3000.com).

This project recreates the [Open-Amp paper](https://arxiv.org/abs/2411.14972) (arXiv:2411.14972):
training a contrastive guitar-effects encoder and a one-to-many amp-emulation model on audio rendered
through hundreds of neural amp captures. Instead of the paper's original capture set, we source diverse
**amp-only (DI)** [neural amp modeler (NAM)](https://www.neuralampmodeler.com/) captures from the
TONE3000 community, each becoming one "device" the model learns to emulate.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| **1. Acquisition** | Authenticate, discover, select ~400 diverse DI amp captures, download, validate, dedup, and produce the device manifest. | ✅ Implemented — see **[T3K.md](T3K.md)** |
| **2. Rendering** | Build the clean input corpus, render it through each captured device on GPU (offline, once), and expose PyTorch `Dataset` classes. | ✅ Implemented — see **[OA3K.md](OA3K.md)** |
| **3. Training** | Train the contrastive effects encoder + one-to-many amp-emulation model. | Planned |
| **4. Evaluation** | Objective/subjective evaluation against the paper's baselines. | Planned |

Each capture is assigned a stable `device_id` in Phase 1 that becomes the embedding-table index used by
all later phases — so the Phase 1 manifest is the contract the rest of the project builds on.

## Getting started

This project uses **conda** for environment management. Create and activate the shared environment
(Python 3.10) once, then install the per-phase extras into it:

```bash
conda create -n open-amp3000 python=3.10   # create the shared env (matches requires-python >=3.10)
conda activate open-amp3000
```

Phase 1 is a Python package + CLI (`t3k`). Full setup, commands, and API notes are in **[T3K.md](T3K.md)**:

```bash
pip install -e ".[dev]"      # install (see T3K.md for the validation extras)
cp .env.example .env         # add your TONE3000 publishable key
t3k auth                     # one-time browser authorization
t3k discover && t3k select && t3k download && t3k validate && t3k dedup && t3k finalize
```

Phase 2 is a Python package + CLI (`oa3k`) that consumes the Phase 1 manifest. Full setup (GPU
stack, the manual EGDB/NAM-sweep downloads) and commands are in **[OA3K.md](OA3K.md)**:

```bash
pip install -e ".[phase2]"                       # torch(+CUDA), NAM, soundfile, typer, ...
oa3k corpus prepare                              # raw sources -> clean corpus + splits + clip grid
oa3k render run --devices 0-2 && oa3k render verify   # pilot, then `oa3k render run` for the full job
```

## Repository layout

```
src/t3k/            Phase 1 acquisition package (see T3K.md)
src/data/           Phase 2 corpus + rendering package + PyTorch datasets (see OA3K.md)
scripts/t3k.py      Phase 1 CLI entry point
scripts/oa3k.py     Phase 2 CLI entry point
configs/phase2.yaml Phase 2 tunables (seed, budget, splits, chunk size)
tests/              unit tests (pytest) + tiny .nam fixtures
data/               outputs: captures + clean corpus + renders + manifests (git-ignored)
results/            Phase 2 QA exports + report (audio git-ignored)
docs/phase1-spec.md the Phase 1 requirement spec
docs/phase2-spec.md the Phase 2 requirement spec
T3K.md / OA3K.md    Phase 1 / Phase 2 tooling guides
```

## Data, licensing & etiquette

Downloaded `.nam` capture files, OAuth tokens, and API keys are **never committed** (`data/` and secrets
are git-ignored) and capture files are **never redistributed** — only per-capture metadata and license
terms are recorded in the manifest. Acquisition uses the official TONE3000 API only (no scraping), with a
client-side rate limiter kept on at all times. See [T3K.md](T3K.md) for details.

Phase 2's clean corpus and rendered audio are **derived data** from the same license-restricted
captures — they stay under the git-ignored `data/` tree and are **never published**. Only
metadata manifests and the metadata-only `results/phase2_report.md` are tracked.

## License

See [LICENSE](LICENSE).
