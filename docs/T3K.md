# `t3k` — Phase 1: TONE3000 amp-capture acquisition

The `t3k` package authenticates against the TONE3000 API, discovers diverse **amp-only (DI) NAM**
captures, selects ~430, downloads them, validates every file by loading it and running a render probe,
deduplicates, and writes a final manifest for the Phase 2 rendering pipeline. Capture files are **never
redistributed** — only metadata + license info is recorded.

The full requirement spec lives in [docs/phase1-spec.md](docs/phase1-spec.md). For the wider project
context see the [README](README.md).

---

## Setup

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .              # core (auth/discover/select/download)
pip install -e ".[validate]"  # adds the NAM/torch stack for `t3k validate`
pip install -e ".[dev]"       # adds pytest
```

(Or `pip install -r requirements.txt` / `requirements-dev.txt`.)

> **Supply-chain note (spec §2):** `pytorch-lightning` 2.6.2/2.6.3 were subject to a known incident.
> The pins cap below that (`==2.5.2`). Do not bump into that range.

### 2. Create a publishable API key

On tone3000.com go to **Settings → API Keys** and create a **publishable** key (`t3k_pub_…`).
Copy `.env.example` to `.env` (git-ignored) and paste it:

```bash
cp .env.example .env
# edit .env -> T3K_PUBLISHABLE_KEY=t3k_pub_xxxxx
```

| Variable | Default | Purpose |
|---|---|---|
| `T3K_PUBLISHABLE_KEY` | — | **Required.** Your publishable key. |
| `T3K_TOKEN_PATH` | `./.t3k_tokens.json` | OAuth token cache (chmod 600). |
| `T3K_BASE_URL` | `https://www.tone3000.com/api/v1` | API base. |
| `T3K_REDIRECT_URI` | `http://localhost:3001` | OAuth redirect (localhost allowed in dev). |
| `T3K_DATA_DIR` | `./data` | Output root (git-ignored). |
| `T3K_RATE_LIMIT_RPM` | `80` | Client-side cap (server allows 100). |
| `T3K_SEED` | `1234` | Deterministic selection seed. |

---

## Usage

Run the stages in order. Every command is **idempotent** — re-running skips completed work and never
re-issues satisfied API calls or downloads. Invoke either via the installed `t3k` command or
`python scripts/t3k.py`.

```bash
t3k auth        # OAuth standard flow (PKCE); verifies GET /user, prints username
t3k discover    # build 2k–4k candidate models         -> data/manifests/candidates.parquet
t3k select      # deterministic diversity pick of ~430 -> data/manifests/manifest.parquet
t3k download    # fetch .nam files (Bearer, resumable) -> data/captures/{tone_id}/{model_id}.nam
t3k validate    # NAM load + 5 s render probe on CPU
t3k dedup       # drop exact-hash + ESR near-duplicates
t3k finalize    # top up to ≥400, assign device_ids    -> manifest.parquet + rejected.parquet
t3k status      # show manifest status counts anytime
```

`t3k auth` is interactive: it opens a browser to authorize once, then caches tokens. On a machine
without a browser use `t3k auth --headless` (it prints the authorize URL and asks you to paste the
redirect URL back). Other useful options: `t3k discover --max-candidates N --no-resume`,
`t3k select --target N --seed N`, `t3k finalize --no-top-up`, and `-v/--verbose` on any command.

### What each stage does

- **auth** — OAuth 2.0 + PKCE "standard flow". Generates a verifier/challenge, opens the browser to
  `/oauth/authorize`, runs a one-shot listener on the redirect port, exchanges the code at
  `/oauth/token`, and stores access/refresh tokens (auto-refreshed on 401/expiry).
- **discover** — several `/tones/search` passes (sorts *trending/downloads/newest* + a ~40-term brand
  keyword sweep), fetches each tone's `/models`, normalizes, keeps **amp-only NAM A1/A2** captures,
  excludes cab/IR/full-rig/pedal/bass, dedupes by model id.
- **select** — hard filters (DI amp, usable license, has `model_url`) → caps (≤2/tone, ≤8/creator,
  ≤6 per make-model) → gain-style minimums (≥20% each of clean/crunch/high-gain) → greedy brand
  round-robin, tie-broken by download count. Fully deterministic given candidates + seed.
- **download** — Bearer-authenticated, streamed, checksummed (`sha256`), atomic (`.part` → rename),
  resumable. A permanently failing A2 model falls back to the tone's A1 sibling (`architecture_fallback`).
- **validate** — parse JSON, load with `neural-amp-modeler`, push a fixed 5 s probe (2 s log sweep +
  1 s silence + 2 s DI) through the model on CPU, and assert output is finite / audible / non-exploding /
  quiet-in-silence. Probe output saved as `{model_id}.probe.npy` for dedup. A2 load failures retry via
  the A1 sibling.
- **dedup** — exact `sha256` duplicates dropped globally; within each normalized (make, model) group,
  probe outputs with ESR < 0.01 collapse to the higher-download capture.
- **finalize** — if fewer than 400 remain, loops back to select/download/validate/dedup for
  replacements; assigns the stable, zero-padded `device_id` (the embedding-table index — **never**
  changes once assigned); writes `manifest.parquet` (accepted) + `rejected.parquet` (with reasons).

---

## Outputs

- `data/captures/{tone_id}/{model_id}.nam` — downloaded captures (git-ignored, never redistributed).
- `data/captures/{tone_id}/{model_id}.probe.npy` — per-capture probe render (float32).
- `data/manifests/candidates.parquet` — discovery pool.
- `data/manifests/manifest.parquet` — working manifest, then the **final** accepted set.
- `data/manifests/rejected.parquet` — everything not accepted, with `reject_reason`.

Final manifest columns are documented in [docs/phase1-spec.md](docs/phase1-spec.md) §6 (one row per
accepted capture, keyed by `device_id`).

---

## Development

```bash
pytest            # 51 unit tests: selection, manifest, validation, dedup, rate-limit, auth, client
```

Tests mock the API and use tiny bundled `.nam` fixtures (`tests/fixtures/`); they do **not** require
network access, a torch install, or credentials. The NAM inference call is dependency-injected, so the
pipeline and its tests never import torch.

Layout:

```
src/t3k/
├── config.py      constants.py   ratelimit.py    # settings, tables, token bucket + backoff
├── auth.py        client.py                       # PKCE flow / token store; rate-limited HTTP + endpoints
├── normalize.py   manifest.py                     # API-object -> row mapping; parquet schema/IO
├── discover.py    select.py      download.py       # pipeline stages ...
├── validate.py    dedup.py       finalize.py
├── probe.py       nam_backend.py                  # probe signal + metrics; NAM load/render adapter
└── cli.py                                         # `t3k` command group
scripts/t3k.py                                     # entry point (no-install shim)
```

---

## Deviations from the spec (live API schema)

The spec's field names (§3/§5) predate a check against the live schema
([`github.com/tone-3000/api` `types.ts`](https://github.com/tone-3000/api)). All parsing is defensive
(missing/renamed/wrapped fields tolerated; malformed records logged and skipped), and the following
mappings are applied in [normalize.py](src/t3k/normalize.py):

| Spec says | Live schema | Handling |
|---|---|---|
| `gears` (per tone) | `gear` (single, may be `{slug}`) | `gear_type`; `gears=amp` still sent as a search filter. |
| `architecture` (per model) | `architecture_version` | mapped to `A2`/`A1` (leading `2`/`1`). |
| `downloads`, `favorites` | `downloads_count`, `favorites_count` | fall back to both names. |
| `make` (per model) | tone-level `makes[]` | derived from `makes[]`, else inferred from title tokens. |
| `model` | not a distinct field | normalized from model `name`/tone `title`. |
| `platform=nam` | tone `format` | filtered via `format`; NAM-only enforced on the record. |
| `license` string | `license` (may be an object) | coerced to text; missing → `unknown` (excluded from the final 400 unless needed, then flagged). |
| `sample_rate` | usually absent | defaults to 48 kHz (NAM standard). |

The one place that depends on the exact `neural-amp-modeler` Python API (loading an exported `.nam`
and running inference) is isolated in [nam_backend.py](src/t3k/nam_backend.py); if your installed NAM
version exposes a different entry point, adapt only that file.

**Platform etiquette:** official API only (no scraping, no auth bypass), the rate limiter stays on
(~80 req/min) even though the server allows 100, and license terms are recorded per capture.
