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
| **2. Rendering** | Render a training audio dataset through each captured device. | Planned |
| **3. Training** | Train the contrastive effects encoder + one-to-many amp-emulation model. | Planned |
| **4. Evaluation** | Objective/subjective evaluation against the paper's baselines. | Planned |

Each capture is assigned a stable `device_id` in Phase 1 that becomes the embedding-table index used by
all later phases — so the Phase 1 manifest is the contract the rest of the project builds on.

## Getting started

Phase 1 is a Python package + CLI (`t3k`). Full setup, commands, and API notes are in **[T3K.md](T3K.md)**:

```bash
pip install -e ".[dev]"      # install (see T3K.md for the validation extras)
cp .env.example .env         # add your TONE3000 publishable key
t3k auth                     # one-time browser authorization
t3k discover && t3k select && t3k download && t3k validate && t3k dedup && t3k finalize
```

## Repository layout

```
src/t3k/            Phase 1 acquisition package (see T3K.md)
scripts/t3k.py      CLI entry point
tests/              unit tests (pytest) + tiny .nam fixtures
data/               outputs: captures + manifests (git-ignored)
docs/phase1-spec.md the Phase 1 requirement spec
T3K.md              Phase 1 tooling guide
```

## Data, licensing & etiquette

Downloaded `.nam` capture files, OAuth tokens, and API keys are **never committed** (`data/` and secrets
are git-ignored) and capture files are **never redistributed** — only per-capture metadata and license
terms are recorded in the manifest. Acquisition uses the official TONE3000 API only (no scraping), with a
client-side rate limiter kept on at all times. See [T3K.md](T3K.md) for details.

## License

See [LICENSE](LICENSE).
