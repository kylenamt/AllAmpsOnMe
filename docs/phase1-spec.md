# Phase 1 Spec — TONE3000 Amp Capture Acquisition

**Deliverable:** a Python package + CLI that authenticates against the TONE3000 API, selects ~400 diverse amp-only (DI) NAM captures, downloads them with metadata and license info, validates every file, deduplicates, and produces a final manifest for the rendering pipeline (Phase 2).

This document is standalone — everything the implementing agent needs is here.

---

## 1. Context (read once)

We are recreating the Open-Amp paper (arXiv:2411.14972): training a contrastive guitar-effects encoder and a one-to-many amp emulation model on audio rendered through hundreds of neural amp captures. This phase only acquires the captures. Requirements downstream:

- Captures must be **amp-only (DI)** — no cabinet/full-rig captures.
- Captures are **NAM files** (`.nam`, JSON): prefer architecture **A2**, fall back to **A1** per capture if the A2 file fails validation.
- Each capture becomes one "device" with one embedding row, so we need **diversity** (many makes/models/gain styles), not many variants of the same amp.
- License terms must be recorded per capture; files are never redistributed.
- Everything must be **resumable and idempotent** — re-running any command skips completed work.

## 2. Environment & dependencies

- Python ≥ 3.10, Linux. No GPU needed for this phase.
- Dependencies: `requests` (or `httpx`), `pandas` + `pyarrow`, `numpy`, `soundfile`, `neural-amp-modeler` (PyTorch-based; used only in the validation step), `tqdm`, `click` or `typer` for the CLI.
- Pin all versions in `requirements.txt`. Note: NAM's training stack depends on PyTorch Lightning; avoid Lightning 2.6.2/2.6.3 (known supply-chain incident) — pin a safe version.
- Secrets via environment variables / `.env` (git-ignored): `T3K_PUBLISHABLE_KEY`, optional `T3K_TOKEN_PATH` (default `./.t3k_tokens.json`).

## 3. TONE3000 API facts

- Docs: `https://www.tone3000.com/api` · reference client: `https://github.com/tone-3000/api` (`src/tone3000-client.ts` — port its logic to Python).
- Base URL: `https://www.tone3000.com/api/v1`
- **API key:** user creates one at tone3000.com → Settings → API Keys → a publishable key `t3k_pub_…`. The human operator does this manually; the tool reads it from env.
- **Auth: OAuth 2.0 + PKCE, "Standard Flow"** (app fetches tones programmatically after the user connects their account once):
  1. Generate PKCE verifier/challenge; open `GET /api/v1/oauth/authorize` with the publishable key, `redirect_uri=http://localhost:3001` (localhost origins are allowed in development without registration), and standard-flow params.
  2. Run a tiny local HTTP listener on the redirect port; the user completes login in a browser; the listener receives the authorization code.
  3. `POST /api/v1/oauth/token` exchanges the code for access + refresh tokens. Same endpoint refreshes expired access tokens.
  4. Persist tokens to `T3K_TOKEN_PATH` (chmod 600). Auto-refresh on 401; if refresh fails, prompt re-auth.
  - Headless fallback: print the authorize URL so the operator can open it on another machine, and accept a pasted redirect URL.
- **Endpoints used:**
  - `GET /tones/search` — filters include text query, `gears` (use `amp`), `platform` (`nam`), `architecture` (`2`, fall back `1`), sort (e.g. trending/downloads), pagination (`page`, `page_size`).
  - `GET /tones/{id}` — tone detail/metadata.
  - `GET /models?tone_id={id}` — models within a tone pack; each model has `model_url`, name, architecture info.
  - `GET /models/{id}` — single model detail.
  - Model file download: GET the `model_url` **with the Bearer access token** (plain unauthenticated fetch fails).
- **Rate limit: 100 requests/minute.** Implement a client-side token bucket at ~80 req/min, exponential backoff with jitter on 429/5xx, and honor `Retry-After` if present.
- API is v1/beta: wrap all response parsing defensively; log and skip malformed records rather than crashing. If fields named here differ in practice, adapt to the live schema and document the difference in the README.

## 4. Repo layout for this phase

```
src/t3k/
├── auth.py          # PKCE flow, token store, refresh
├── client.py        # rate-limited authenticated HTTP wrapper + typed endpoint methods
├── discover.py      # search & candidate collection
├── select.py        # diversity-based selection of ~400
├── download.py      # model file downloads (resumable, checksummed)
├── validate.py      # NAM load + render checks
├── dedup.py         # near-duplicate removal
└── manifest.py      # manifest schema, read/write, status transitions
scripts/
└── t3k.py           # CLI entry point (subcommands below)
data/captures/       # (gitignored) downloaded .nam files: {tone_id}/{model_id}.nam
data/manifests/      # candidates.parquet, manifest.parquet, rejected.parquet
tests/               # unit tests with recorded/mocked API responses
```

## 5. Pipeline stages (CLI subcommands)

Run order: `auth → discover → select → download → validate → dedup → finalize`. Each stage reads/writes manifests and is independently re-runnable.

### 5.1 `t3k auth`
Perform the OAuth standard flow described above; verify with `GET /user`; print the connected username.

### 5.2 `t3k discover` — build the candidate pool
- Goal: **2,000–4,000 candidate models** (metadata only, no files yet).
- Query strategy (dedupe by model id across all passes):
  - Paginate `GET /tones/search` with `gears=amp`, `platform=nam`, `architecture=2`, across several sorts (trending, most downloaded, newest) to escape ranking bias.
  - Keyword passes to force brand diversity: iterate a built-in list of ~40 amp search terms (fender, marshall, vox, mesa, orange, peavey, soldano, friedman, bogner, hiwatt, engl, evh, diezel, jcm800, plexi, twin reverb, ac30, rectifier, 5150, jazz chorus, tweed, deluxe, princeton, bassman, jtm45, dsl, sl0, mark v, etc.).
- For each tone hit, fetch its models (`GET /models?tone_id=`) and record one row per model.
- **Exclusions at this stage:** anything whose tone/gear metadata indicates cab, IR, full rig, pedal, bass amp (unless tagged usable for guitar), or "experimental capture"; models whose architecture is neither A2 nor A1.
- Record per row (plus a `raw_json` column preserving the full API objects):
  `tone_id, model_id, tone_title, model_name, make, model, tags[], gear_type, capture_type, architecture, sample_rate (if given), license, creator, creator_id, downloads, favorites, created_at, model_url, tone_url`
- Output: `data/manifests/candidates.parquet`.

### 5.3 `t3k select` — pick ~430 for download
Deterministic (seeded) selection targeting **430** (headroom over 400 for validation/dedup losses):

1. **Hard filters:** DI amp captures only; license permits download/use (record license string; exclude captures with explicit no-use terms); exclude candidates with missing `model_url`.
2. **Caps:** max 2 models per tone pack; max 8 models per creator; max 6 per normalized (make, model) pair.
3. **Gain-style balance:** classify each candidate as clean / crunch–mid / high-gain via tag+title keywords (clean, edge, crunch, lead, high gain, metal, djent, gain 8…, etc.; unknown = its own bucket). Target ≥20% in each of the three known buckets.
4. **Brand spread:** greedy round-robin across normalized `make`; break ties by download count (popularity as a quality proxy).
5. Output: `manifest.parquet` with `status=selected`. Print a summary table (per-make, per-bucket counts).

### 5.4 `t3k download`
- Download each selected model's `.nam` to `data/captures/{tone_id}/{model_id}.nam` with Bearer auth.
- Resumable: skip files that exist with matching size/hash; retry with backoff (max 5); record `sha256`, `file_bytes`, `downloaded_at`; set `status=downloaded` or `status=download_failed` (with error).
- If a selected A2 model fails permanently, attempt the same tone's A1 model as substitute (`architecture=1` model list), flagging `architecture_fallback=true`.

### 5.5 `t3k validate`
For every downloaded file:
1. **Parse:** valid JSON; record declared architecture, config summary, expected sample rate (assume 48 kHz if unspecified — NAM standard).
2. **Load:** instantiate with the `neural-amp-modeler` Python package. If an A2 file fails to load and an A1 sibling exists, swap and re-run (this operationalizes the Phase 0 smoke-test decision).
3. **Render probe:** pass a fixed 5 s probe signal (bundled WAV: 2 s logarithmic sine sweep 20 Hz–20 kHz at −12 dBFS + 1 s silence + 2 s guitar DI snippet) through the model on CPU.
4. **Checks:** output finite (no NaN/Inf); not silent (RMS > −80 dBFS on non-silent segments); not exploding (peak < +6 dBFS after applying any calibration/level metadata in the file); silence in → near-silence out (rules out broken DC/noise models).
5. Store the probe output as `data/captures/{tone_id}/{model_id}.probe.npy` (float32) — reused by dedup. Set `status=validated` or `status=rejected` + `reject_reason`.

### 5.6 `t3k dedup`
- Pairwise-compare probe outputs *within* the same normalized (make, model) group and same tone pack (full N² not needed): compute ESR between probe outputs; if ESR < 0.01, mark the lower-download-count one `status=duplicate`.
- Also drop exact file-hash duplicates globally.

### 5.7 `t3k finalize`
- If validated non-duplicate count < 400, loop back: auto-select replacement candidates (same selection rules) and run download/validate/dedup on them until ≥400 or candidates exhausted.
- Assign stable `device_id` = zero-padded index ordered by (make, model, model_id) — **this is the embedding-table index used by all later phases; it must never change once assigned.**
- Write final `manifest.parquet` (only `status=validated`, with device_id) and `rejected.parquet` (everything else, with reasons). Print final summary + license breakdown.

## 6. Manifest schema (final)

One row per accepted capture:

| column | type | notes |
|---|---|---|
| device_id | int | stable embedding index, 0…N-1 |
| tone_id, model_id | int/str | TONE3000 identifiers |
| tone_title, model_name | str | |
| make, model | str | normalized (lowercase, trimmed) |
| tags | list[str] | |
| gain_bucket | str | clean / crunch / high_gain / unknown |
| architecture | str | "A2" or "A1" |
| architecture_fallback | bool | |
| sample_rate | int | |
| file_path, sha256, file_bytes | str/str/int | |
| license | str | verbatim license text/identifier |
| creator, creator_id, tone_url | str | attribution |
| downloads, favorites, created_at | int/int/ts | |
| probe_path | str | |
| downloaded_at, validated_at | ts | |
| raw_json | str | full API record, for future-proofing |

## 7. Acceptance criteria

1. `t3k auth` completes and `GET /user` returns the operator's account.
2. Final manifest contains **≥400 validated, deduplicated, DI amp-only captures**, each loadable in Python and passing the render probe.
3. ≥25 distinct makes; no (make, model) pair exceeds 6 captures; each of clean/crunch/high-gain ≥ 15% of the set.
4. Every row has non-empty `license`, `creator`, and `tone_url`.
5. Re-running any command performs no redundant downloads/API calls (idempotence), and the client never exceeds ~80 requests/min.
6. `pytest` passes: unit tests for selection logic, manifest transitions, and validation checks (mock API; 2–3 tiny bundled .nam fixtures for load/render tests).
7. README section documents setup (API key creation, auth), each command, and any deviations found between this spec and the live API.

## 8. Out of scope for Phase 1

- No audio dataset rendering (Phase 2), no model training, no full-rig or IR captures, no A2→A1 conversion, no publishing or re-uploading of capture files anywhere.

## 9. Notes & gotchas

- **Never commit** `.nam` files, tokens, or the publishable key. `data/` is gitignored.
- Respect the platform: official API only, no HTML scraping, no auth bypass, keep the rate limiter on even if the API tolerates more.
- Licenses: the common "T3K" license permits download/use and royalty-free outputs but forbids redistribution of the files — recording it per capture is mandatory; when license text is missing from the API response, fetch the tone page metadata via the API (not scraping) or mark `license=unknown` and exclude from the final 400 unless needed to reach quota (flag such rows).
- The API may evolve (v1/beta). Field names in §3/§5 come from official docs and the reference client as of July 2026 — verify against live responses first, adapt, and document.
