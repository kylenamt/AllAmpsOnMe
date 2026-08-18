# Pipeline operator guide

- One package (`openamp`), one CLI (`openamp`). Stages run in order; every stage reads/writes the parquet manifests under `data/manifests/`, is **idempotent/resumable**, and is **deterministic given the `seed`**.
- This guide covers prerequisites, what each stage does, resume semantics, and troubleshooting.
- Design rationale per stage: [`history/`](history/). Quick tour: the [README](../README.md).

```
acquire :  auth · discover · select · download · validate · dedup · finalize · status
corpus  :  corpus · subset · render · verify
emulate :  emulate · emulate-compare · emulate-validate · emulate-demo
```

## Install & configure

```bash
conda activate open-amp3000        # Python 3.10 env
pip install -e ".[dev]"            # all deps + pytest
cp .env.example .env               # add OPENAMP_API_KEY
```

- `torch` must be a **CUDA-enabled** build for `render` (GPU) and `emulate`; eval runs on CPU or GPU.
- `neural-amp-modeler` must be **>= 0.13** — the first release whose WaveNet parses the A2 layer schema; older versions can't load an A2 capture at all. (0.13 also dropped the `pkg_resources` import, so the old `setuptools<81` cap is gone.)
- `pytorch-lightning` pinned `==2.5.2`; avoid 2.6.2/2.6.3 (supply-chain incident).

### Configuration

- All pipeline tunables: [`configs/openamp.yaml`](../configs/openamp.yaml) — top-level `seed` / `sample_rate`, then `corpus:` / `render:` sections.
- Emulator knobs: per-run files under [`configs/emulate/`](../configs/emulate/).
- Pass a different file with `--config path.yaml` to the corpus/render/emulate commands.
- API access and data root come from the environment / `.env`:

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

- `auth` is interactive: opens a browser once, caches tokens (auto-refreshed on 401/expiry). Headless: `openamp auth --headless` (prints the URL, you paste the redirect back).
- **discover** — several `/tones/search` passes (trending/downloads/newest sorts + a brand keyword sweep), fetches each tone's `/models`, keeps **amp-only NAM** captures of the configured architecture (`acquire.architecture`, default **A2**), excludes cab/IR/full-rig/pedal/bass, dedupes by model id.
  - The architecture filter must be passed to **both** `/tones/search` **and** `/models` — it is not inherited. `/models` defaults to A1, so passing it only to the search silently harvests A1 captures out of tones the search selected for having A2. This is exactly how the original 844-device corpus ended up entirely A1.
- **select** — hard filters (DI amp, usable license, has `model_url`) → caps (≤2/tone, ≤8/creator, ≤6 per make-model) → gain minimums (≥20% each of clean/crunch/high-gain) → greedy brand round-robin, tie-broken by downloads. Fully deterministic given candidates + seed.
- **download** — Bearer-authenticated, streamed, checksummed (`sha256`), atomic, resumable. A permanently failing A2 model falls back to the tone's A1 sibling (`architecture_fallback`).
- **validate** — parse JSON, load with `neural-amp-modeler`, push a fixed 5 s probe (2 s log sweep + 1 s silence + 2 s DI) through the model, assert output is finite / audible / non-exploding / quiet-in-silence. Probe output saved as `{model_id}.probe.npy` for dedup. A2 load failures retry via the A1 sibling if a client is available.

### Capture architectures: A1 vs A2

- **A2 is canonical.** An A2 export is not a bare net: it's a `SlimmableContainer` holding several *width variants* of the same capture ([Slimmable NAM](https://arxiv.org/abs/2511.07470)) — a runtime CPU/quality dial, where `max_value` is the variant's width fraction.
- Every variant models the same device, the narrow ones less accurately, so offline rendering always resolves the container to its **full-width (`max_value: 1`) variant**.
- A1 is a plain WaveNet in a pre-0.7 schema that current NAM no longer parses; `dsp/nam.py` upgrades it on load (verified bit-identical to the output of the NAM version the A1 corpus was rendered with), so both architectures stay loadable.
- **migrate-a2** — one-off: re-points the **finalized** devices at their A2 captures, rewriting `manifest.parquet` in place and backing the old one up to `manifest.a1.parquet`.
  - Runs on the final manifest rather than through select/finalize so `device_id` is never reassigned — the rendered audio is addressed by `device_id`, and re-deriving those ids would orphan 106 GB of it.
  - Pairing is by **exact model name**: TONE3000 did not retrain every model on multi-model tones, and the models under one tone are the same amp at different gain settings, so an unmatched device keeps its A1 capture (flagged `architecture_fallback`) rather than being paired by similarity.
  - Existing renders are *not* invalidated — they came from A1 captures and `renders.parquet` records the `nam_sha256` each one used, so provenance stays truthful; only future renders pick up A2.
- **dedup** — exact `sha256` duplicates dropped globally; within each (make, model) group, probe outputs with ESR < 0.01 collapse to the higher-download capture.
- **finalize** — if fewer than 400 remain, loops back through select/download/validate/dedup for replacements (needs an API key; `--no-top-up` disables), then assigns the stable, zero-padded `device_id` (the embedding-table index — **never changes once assigned**) and writes `manifest.parquet` (accepted) + `rejected.parquet` (with reasons).

## 2 — Corpus + render

### Raw inputs (manual, one-time)

Place two sources under the data root (`OPENAMP_DATA_DIR`, default `./data`):

| Source | Place at | Notes |
|---|---|---|
| **EGDB** direct-input WAVs | `data/raw/egdb/` | Clean DI audio only (not the amp-rendered versions). Files < 4 s or > 1% clipping are auto-rejected. |
| **NAM sweep** `v3_0_0.wav` | `data/raw/nam_sweep/v3_0_0.wav` | Standardized reamp signal; appended on top of the minutes budget, sent to the **train** split only. |

If either is missing, `openamp corpus` prints exactly where to put it and stops. Acquisition (`openamp finalize`) must have produced `data/manifests/manifest.parquet` and the `.nam` files first.

```bash
openamp corpus                       # raw sources -> 48 kHz clean corpus + splits + clip grid
openamp subset --size 450            # (optional) diverse, gain-balanced device subset
openamp render --devices 0-2         # pilot a few devices
openamp verify                       # inspect results/qa + render_report.md
openamp render --devices @data/manifests/render_subset.txt   # full job (resumable)
```

- **corpus** — converts every source to 48 kHz mono float32, normalizes to −18 dBFS RMS then limits true peaks to −1 dBFS, selects EGDB files up to `corpus.minutes_total`, splits **by file** ~80/10/10 (sweep → train; test → EGDB only). Writes `data/clean/{split}/{file_id}.flac`, `corpus.parquet`, the fixed 2 s `clips.parquet` grid.
- **subset** — the acquisition manifest often exceeds the ~400 target; trims to a diverse subset (per-tone/creator/make-model caps + a per-gain-bucket floor), writes the device-id list to `render_subset.txt`. Doesn't touch the manifest; feed the list to `render --devices @<path>`.
- **render** — whole-file, then sliced: each file is streamed through the model in `render.chunk_seconds` windows carrying the receptive field R as left-context and discarding the warmup, so the result is **bit-identical to a single full-file forward** (no clip-boundary transients).
  - One output scalar per device pins the global peak at 0.999.
  - Writes `data/renders/{device_id:04d}/{file_id}.flac` + `renders.parquet`. A device whose files all exist with matching rows is skipped.
  - Refuses to start if free disk is below 1.3× the estimate (~75–110 GB for ~400 devices).
  - `--io-workers` overlaps the FLAC-encode/hash pass with the GPU forward (a large NFS win).
- **verify** — completeness (+ hashes), signal sanity (no NaN/Inf, not silent, not a pass-through via `ESR > 0.001`), alignment. Exports clean/render WAV + spectrogram pairs to `results/qa/`, writes `results/render_report.md`, emits `devices_final.parquet` (verified devices only). If the count drops below 400, grow the pool by raising `subset --size` (or the caps) and rendering the added device ids.

## 3 — Emulate (amp foundation model)

- One FiLM-conditioned model emulates **all** rendered devices at once, steered by a learnable per-device embedding.
- Architecture (`emulate.arch`: FiLM-TCN or the A2 FiLM-WaveNet) and every size are config knobs — exploring variants is copy-a-config-and-change-numbers.
- Runs consume the renders (`renders.parquet` + `corpus.parquet`).

```bash
# Sanity ladder first (cheap): overfit one batch, then a ~1 h mini-run.
openamp emulate --config configs/emulate/paper.yaml --overfit
openamp emulate --config configs/emulate/paper.yaml --limit-devices 10

openamp emulate --config configs/emulate/paper.yaml       # full run -> results/emulate/paper/
openamp emulate --config configs/emulate/embed16.yaml     # a size-sweep variant
openamp emulate-compare results/emulate/*                 # test-split rows -> comparison.csv
openamp emulate-validate paper                            # per-amp test ESR -> <run>/per_device_esr.csv
openamp emulate-demo results/emulate/paper                # clean/target/pred WAVs (listening check)
```

- **Config** — all knobs live in the `emulate:` section; a per-run file under `configs/emulate/<name>.yaml` overrides just that section, and the **run name is the file stem**.
  - Shipped configs: `paper` (default), `embed16`/`embed256`, `channels32`, `deep3x8`, `wavenet_a2` (capture-native architecture), `wavenet_a2_tanh` (same run, `wn_activation: tanh`), `mlpfilm_a2_256` / `mlpfilm_a2_h16_256` (MLP FiLM generator — `h16` is parameter-matched to `nam_a2_256`, so its delta isolates nonlinearity from capacity), `delta_a2_256` / `delta_a2_r6_256` / `delta_a2_r16_256` (conditioning moved into the conv weights — `r6` is the parameter-matched half of that pair, `r16` the capacity end), `table_a2_256` (the full-rank control: a free per-device weight table, no low-rank basis), `baseline_{clean,crunch,high_gain}` (one-to-one `single_device` references for the one-to-many gap).
  - Two bases to copy from: `paper` for FiLM-TCN runs, `nam_a2` for FiLM-WaveNet ones (same A2 topology as `wavenet_a2`, but with NAM's own training numbers — lr 4e-3, weight decay 3.17e-7, 100 epochs, MRSTFT as a light regularizer rather than a 1:1 term). Copy one, change numbers, rerun — no code changes.
- **Model** — all five archs are causal and conditioned on the device embedding at every layer (FiLM's per-channel scale+shift in three of them, a conv-weight residual in the other two).
  - `film_tcn`: `blocks × layers_per_block` dilated convs (1021 samples / 21 ms receptive field for the paper default).
  - `film_wavenet`: the exact NAM A2 WaveNet topology of the corpus's own captures (23 dilated layers, channels 8, 6347 samples / 132 ms), FiLM at the A2 schema's pre-activation hook, per-layer nonlinearity chosen by `wn_activation` — `leakyrelu` (slope 0.01, what the captures use) or `tanh`, both A2-schema activations so either stays plugin-playable after export.
  - `mlpfilm_wavenet`: that same A2 network, but each layer's FiLM generator is an independent `Linear(E, cond_hidden) → cond_activation → Linear(cond_hidden, 2C)` rather than a single `Linear(E, 2C)`. Still per-channel affine, so it folds away to the identical capture at export — `cond_hidden` costs training parameters and plugin bundle bytes, never real-time CPU. At `embedding_dim: 256` / `wn_channels: 8`, `cond_hidden: 16` is parameter-matched to `film_wavenet` (4,384 vs 4,112 per layer), `32` is roughly double.
  - `delta_wavenet`: that same A2 network with the conditioning one step upstream — no FiLM; the embedding generates a rank-`delta_rank` residual on each layer's conv kernel, bias and mixin gain (`delta = scale · coeff(e) @ normalize(basis)`, one shared basis of unit-norm kernel-shaped directions per layer plus one magnitude scalar — the split matters: without it the delta grows for free and stops being a residual).
    - FiLM is inside that family (`dW = (γ − 1)·W`, `db = (γ − 1)·b + β`, `dm = (γ − 1)·mixin_w`), but a residual can also rotate a kernel, so a device retimes a filter instead of only re-levelling it — a strict superset of the FiLM hook, so a null result is about the hook, not capacity.
    - Folds to the identical stock capture — the fold *is* the addition. At `embedding_dim: 256` / `wn_channels: 8` the generators cost 16,263 × `delta_rank` params (+23 scales), so rank 6 is parameter-matched to `film_wavenet` (97,601 vs 94,576).
  - `tabledelta_wavenet`: the same residual, free and per-device instead of low-rank — each device owns its whole `dW`/`db`/`dm` outright, so nothing is shared but the base kernels. The full-rank limit of `delta_wavenet`, and the control for whether that constraint helps.
    - `embedding_dim` is **derived** from the schedule (10,352 at C=8) and `ecfg.embedding_dim` is ignored; there is no capacity knob; parameters scale with the device count (4.2M at 405 devices, so `--limit-devices` runs are not comparable with full ones).
    - **Cannot be enrolled.** Held-out devices have no row and there is no shared structure to place an unseen amp in, so `emulate-enroll` refuses this arch rather than fitting a table the network never reads. Its val ESR is a capacity ceiling over *trained* devices, not a generalization result.
    - `model.set_delta_parts(kernel=…, bias=…, mixin=…)` masks the residual at inference (one checkpoint, heard every way: is the amp in the filter shape or just the levels?). Not part of the `state_dict`, and the export fold honours it, so a masked capture plays what you heard.
  - Training clips are prefixed with the receptive field of **real left-context** from the source file, so warmup uses actual audio and the loss covers only conditioned samples.
- **Device holdout** — `holdout_frac` (default 0.1) excludes a seeded ~10% of render-ok devices from training entirely, so Phase 5 can enroll them as truly unseen devices (the reference code's 90/10 device split).
  - Drawn once and persisted to `data/manifests/emulate_holdout.txt` — the file is the source of truth, so the holdout stays fixed even if more devices are rendered later.
  - The held-out ids travel in every `checkpoint.pt` and `embedding.pt`; `single_device` runs ignore the holdout (that's how Phase 5 trains one-to-one references on held-out devices).
- **Training** — pre-emphasized ESR + multi-resolution STFT (auraloss), 1:1; Adam lr 5e-4, reduce-on-plateau, early-stopped at the val-ESR plateau; single GPU, AMP.
  - Each run writes `results/emulate/<name>/`: `checkpoint.pt` (best) + `last.pt`, `config.yaml`, `metrics.json`, `train_log.csv`, `embedding.pt` (the table + its manifest hash, saved separately for Phase 5).
  - Every run reports `val_esr_shuffled` (val ESR with the embeddings permuted across devices) — it must be **worse** than `val_esr`, proving the conditioning is doing real work.
  - Console output belongs in the run dir too — launch a long run as `nohup openamp emulate --config configs/emulate/<name>.yaml > results/emulate/<name>/train_stdout.log 2>&1 &` so the log lands beside the checkpoints instead of as a loose `nohup.out` (a resumed leg goes to `train_stdout_resume.log`).
- **Compare** — `emulate-compare` evaluates each run on the held-out **test** split and appends one row per run to `results/emulate/comparison.csv` (`run_name, params, receptive_field_ms, embedding_dim, channels, blocks_x_layers, test_ESR_mean, test_ESR_median, test_MRSL_mean, train_hours`). That CSV is the architecture-exploration deliverable; re-evaluating a run replaces its row.
- **Validate per amp** — `emulate-validate <run-name-or-dir>` takes *one* run, writes `results/emulate/<name>/per_device_esr.csv`: one row per trained device (`device_id, name, make, model, gain_bucket, architecture, n_windows, test_ESR, test_MRSTFT`), sorted best-ESR first, plus a printed summary of the spread, the best/worst amps, medians per gain bucket.
  - Every device is scored on the **same** grid of `--windows` test windows (drawn once from the seed, silence-screened on the clean side), so differences between rows are the model, not window luck.
  - `test_ESR` is pooled per device (one ratio of summed energies, as in training), so one quiet window can't dominate an amp's score.
  - Use it to see which amp types the shared model learned well and whether the run's headline ESR is skewed by a few outlier devices.

> `auraloss` (multi-resolution STFT loss) is a dependency; `pip install -e .` pulls it in. GPU strongly recommended — the paper-default run is ~187 K steps; use `--limit-devices` / `--epochs` for quick passes.

## 4 — Encode (tone encoder)

A standalone experiment, not a step the emulators depend on: it asks whether a device's conditioning vector can be *read off* the capture audio instead of fitted by gradient descent the way `emulate-enroll` does. Nothing consumes its output yet.

```bash
# THE ENV VAR IS PART OF THE RECIPE — see the config header for why.
export OPENAMP_DATA_DIR=data_mixed

# Sanity ladder first (both cheap).
openamp encode --config configs/encoder/smoke.yaml --overfit
openamp encode --config configs/encoder/smoke.yaml --limit-devices 12

# The real run, log beside the checkpoints. `>>` not `>` on a resume.
nohup openamp encode --config configs/encoder/sweep_base.yaml \
      >> results/encoder/sweep_base/train_stdout.log 2>&1 &

# The gate. This, not the training loss, is what decides whether to continue.
openamp encode-eval sweep_base
```

- **Corpus** — `data_mixed` with `sources: [sweep]`: the 171 s NAM capture signal per device, which is the content you actually hold at inference time. `data_mixed` is A1-consistent; do **not** point this at `data_sweep/`, whose renders are A2 for 408/450 devices (§8.5 of `decisions.md`).
- **Two orthogonal splits.** Devices: the same `emulate_holdout.txt` 46 are held back, and they are the *retrieval* test set (unseen amps). Time: a fraction of `sweep_train` is the val region for the same devices (unseen audio). Neither substitutes for the other, and `sweep_val` (9 s) is too short to give ≥4 segments per capture, which is why the time split is taken inside `sweep_train`.
- **Where the time split falls is not cosmetic.** `sweep_train`'s level drops across its length, so holding out the *tail* (`time_split_mode: tail`) validates 14.3 dB below the training level — a quieter operating point of the amp, not held-out audio, and one where every amp is closer to linear and closer to every other amp. The default `strided` mode interleaves instead: `time_split_blocks` blocks, evenly-spaced ones to val, 0.98 dB gap at the default 10. **Read the two `[encode] time split=` / `[encode] split level:` lines in the run header** — the gap is measured per corpus and warns past 3 dB. The 2026-08-11 `sweep_base` run predates this and used the tail split.
- **Reading the numbers.** SupCon bottoms out at `log(K-1)` — 1.0986 at `segments_per_device: 4` — **not** at 0; judge `--overfit` against that. The other end of the scale is `log(P*K-1)` (3.4340 at 8×4), which is what a *collapsed* embedding scores — all points on top of each other. `val_supcon_shuffled` (labels permuted within the batch) is the conditioning control, the analogue of the emulators' `val_esr_shuffled`.
- **If `--overfit` sticks at exactly `log(P*K-1)`, re-run it on another seed before believing it.** One fixed batch at full LR is a harsher setting than training — collapse is an absorbing state there and fresh batches escape it. Measured: a smoke batch collapsed on 4 of 5 seeds while the real 404-device run over the same regions descended 2.89 → 2.64 → 2.44 over three epochs.
- **A branch collapsing in epoch 0 of a real run is normal here — wait for epoch 2.** The dynamic branch pins `scd` at 3.4340 for a few hundred steps partway through epoch 0 and then recovers (3.41 → 2.76 → 2.27). The `[epoch]` line prints only `supcon_s`, so watch the `scd` field on the step lines or `train_supcon_d` in `encoder_log.csv` if you want to see it.
- **The gate** writes `results/encoder/<name>/diagnostics.json` plus `capture_embeddings.npz` and `tsne.{csv,png}`. The two checks that decide the direction: retrieval top-1 on the unseen captures against the `1/46 = 2.2 %` chance line, and the **dry-channel ablation** — in sweep-only mode every device shares one dry file, so the encoder could ignore the dry channel entirely and still ace retrieval while having learned "what this recording sounds like" instead of "what this amp does". If branch redundancy comes back high, the two-branch split is decorative and the honest response is to collapse to one branch, not to retune the weights.
- Encoder runs live under `results/encoder/`, deliberately not `results/emulate/`: they log a contrastive loss, not ESR, and `decisions.md` §6 exists because metrics that share a name got cross-compared once. `scripts/publish_run_dashboard.py` only globs `results/emulate/*/`, so encoder runs do not appear on the dashboard.

## Testing

```bash
python -m pytest
```

- Acquisition tests mock the API and inject the NAM renderer; corpus/render/dataset tests use tiny real audio in `tmp_path`.
- A few `test_validate.py` / `test_finalize.py` cases are skipped unless small `.nam` fixtures exist — see [`../tests/fixtures/README.md`](../tests/fixtures/README.md).
- Key proofs: the chunked render is bit-identical to a full forward; the FiLM-TCN's receptive field matches its formula and its output is causal; FiLM conditioning actually changes the output.

## TONE3000 API notes

- Parsing is defensive — the live schema ([`github.com/tone-3000/api`](https://github.com/tone-3000/api)) differs from early field names, so missing/renamed/wrapped fields are tolerated and malformed records logged and skipped (see `src/openamp/acquire/normalize.py`).
- Notably: `architecture_version` → `A2`/`A1`; `downloads_count`/`favorites_count` fall back to short names; `make` derived from tone-level `makes[]` or inferred from the title; `license` coerced to text (missing → `unknown`); `sample_rate` defaults to 48 kHz.
- The one place tied to the exact `neural-amp-modeler` Python API is isolated in `src/openamp/dsp/nam.py`; if your NAM version exposes a different entry point, adapt only that file.

**Etiquette:** official API only (no scraping, no auth bypass); the rate limiter stays on (~80 req/min) even though the server allows 100; license terms are recorded per capture and capture files are never redistributed.
