# AllAmpsOnMe — architecture

- Follows the [Open-Amp paper](https://arxiv.org/abs/2411.14972): acquire a diverse set of neural amp captures from [TONE3000](https://www.tone3000.com), render a clean guitar corpus through each captured amp, and train a one-to-many FiLM-conditioned TCN that emulates every captured amp from a shared model plus a per-device embedding.
- One process behind a single CLI (`openamp <verb>`). Each stage reads/writes files (parquet manifests + audio), so stages are resumable and can run independently in order.

---

## Package layout

```
src/openamp/
  cli.py                  # the one CLI: a thin verb per pipeline stage
  __main__.py             # `python -m openamp` entry point

  core/                   # shared foundation (pure python; no heavy deps)
    config.py             #   the one Config: env + yaml -> paths + knobs
    constants.py          #   fixed audio facts + vocabulary (splits, gain buckets, filenames)
    manifest.py           #   every parquet schema + atomic read/write
    util.py               #   sha256_file, utc_now
    selection.py          #   diversity-selection engine (caps + gain floor + round-robin)

  dsp/                    # audio + model backend (heavy: soundfile / torch / NAM)
    audio.py              #   audio I/O, level/ESR metrics, the deterministic validation probe
    nam.py                #   neural-amp-modeler adapter (load .nam, run inference)

  acquire/                # get amp captures from the TONE3000 API (the only API-facing group)
    auth.py               #   OAuth 2.0 + PKCE standard flow, token storage
    client.py             #   rate-limited, retrying HTTP client + typed endpoints
    catalog.py            #   static heuristics: search terms, gain keywords, make aliases
    normalize.py          #   defensive normalization of live API objects -> candidate rows
    discover.py           #   stage: build the candidate model pool -> candidates.parquet
    select.py             #   stage: diversity selection -> manifest.parquet (selected)
    download.py           #   stage: download each capture's .nam (resumable, checksummed)
    validate.py           #   stage: load + render-probe each capture (finite/audible/non-exploding)
    dedup.py              #   stage: drop exact-hash + ESR near-duplicate captures
    finalize.py           #   stage: assign stable device_ids -> final manifest + rejected
    migrate.py            #   stage: re-point finalized devices at their A2 captures (device_ids kept)

  corpus/                 # build the clean corpus and render it through every device
    build.py              #   stage: EGDB + sweep -> 48 kHz clean FLAC + corpus/clips manifests
    subset.py             #   stage: pick a diverse, gain-balanced device subset to render
    render.py             #   stage: GPU-render the corpus through each device (resumable)
    verify.py             #   stage: completeness + signal sanity + QA exports -> devices_final

  emulate/                # one-to-many amp foundation models (FiLM-conditioned)
    tcn.py                #   fully-parametric FiLM-TCN + per-device embedding table
    wavenet.py            #   the NAM A2 capture topology, FiLM-conditioned: linear or MLP generator
    models.py             #   arch selection: `emulate.arch` -> build_model()
    dataset.py            #   clean-in / render-out training pairs (real left-context warmup)
    train.py              #   the one training script (pre-emph ESR + MRSTFT) + sanity ladder
    evaluate.py           #   size-comparison harness (-> comparison.csv) + per-amp ESR + demo export
    enroll.py             #   Phase 5: fit an embedding for an unseen device, frozen net
    export.py             #   plugin bundle + pasteable embedding profiles (weight folding)

  encoder/                # tone encoder: (dry, wet) capture audio -> 512-d tone vector
    model.py              #   the two-branch (static / dynamic) waveform encoder
    dataset.py            #   P x K capture episodes + the train/val time-split policy
    train.py              #   contrastive trainer (SupCon + cross-covariance + probes)
    evaluate.py           #   the diagnostics gate (retrieval, dry ablation, redundancy, probes)
```

- `core` and (mostly) `acquire` load on a bare numpy/pandas stack; `dsp`, `emulate` and `encoder` pull torch / soundfile / NAM (`emulate` also uses auraloss).
- **`encoder` is standalone**: no `emulate` module imports from it, and nothing consumes the tone vector yet. The one dependency runs the other way — `encoder/dataset.py` reuses `emulate/dataset.py`'s corpus plumbing (`EmulationDataset` for manifest resolution, source filtering and device-row mapping; `read_with_retry` for the NFS retry policy).
- `cli.py` imports each stage module lazily inside its command body, so `openamp --help` and metadata-only verbs stay fast without loading torch.

---

## The pipeline, stage by stage

Run order (each is one `openamp` verb). "Reads" / "Writes" are files under `data_dir` (default `./data`) and `results_dir` (default `./results`); paths all derive from `openamp/core/config.py`.

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

- The acquisition manifest is one DataFrame carried from discovery through finalize; each stage mutates `status` (+ its own columns) and writes it back, so re-running a stage is a no-op for rows already past it.
- `finalize` assigns the **stable `device_id`** (rank by make, model, model_id) — the embedding-table index that must never change once training has seen it.
- `discover`/`download`/`validate`/`dedup`/`finalize` are heuristics over live API data; `catalog.py` + `normalize.py` hold all the fuzzy classification (gain bucket, make, amp-vs-pedal) so it is auditable in one place.
- This group is effectively **frozen** — the durable asset is `manifest.parquet` + `captures/`, not the acquisition code.

### B. Corpus + rendering — `openamp.corpus`

| Verb | Module | Reads | Writes |
|---|---|---|---|
| `corpus` | `build.py` | `raw/egdb/*.wav`, `raw/nam_sweep/v3_0_0.wav` | `clean/{split}/*.flac`, `manifests/corpus.parquet`, `manifests/clips.parquet` |
| `subset` | `subset.py` | final `manifest.parquet` | `manifests/render_subset.txt` (device id list) |
| `render` | `render.py` | `clean/`, `captures/`, `render_subset.txt` | `renders/{device:04d}/*.flac`, `manifests/renders.parquet` |
| `verify` | `verify.py` | `renders.parquet`, `corpus.parquet` | `manifests/devices_final.parquet`, `results/qa/*`, `results/render_report.md` |

- **build**: scans EGDB direct-input WAVs (+ the NAM standardized sweep), resamples to 48 kHz mono, level-normalizes to −18 dBFS RMS with −1 dBFS peak headroom, selects to a minutes budget, splits **by file** (never by clip) so no clean source leaks across train/val/test. Emits the clean FLACs plus the 2 s clip grid.
- **subset**: acquisition can hold more devices than we want to render (keeps every validated capture). Picks a diverse, gain-balanced subset via the shared `core/selection.py` engine (per-tone/creator/make-model caps + a per-gain-bucket floor). Manifest is never modified.
- **render**: loads each device's `.nam` on the GPU, streams every clean file through it **whole-file, chunked with left-context** (bit-identical to a single forward pass, no clip-boundary transients), then scales/encodes/hashes the output. Resumable and idempotent per device.
- **verify**: completeness + per-signal sanity (non-finite / silent / pass-through / misaligned), writes the final verified device list, exports a few clean/render WAV + spectrogram QA pairs.

### C. Emulation foundation model — `openamp.emulate`

| Verb | Module | Reads | Writes |
|---|---|---|---|
| `emulate` | `train.py` | `renders/`, `renders.parquet`, `corpus.parquet` | `results/emulate/<name>/{checkpoint,last}.pt`, `config.yaml`, `metrics.json`, `train_log.csv`, `embedding.pt` |
| `emulate-compare` | `evaluate.py` | run dirs + test renders | `results/emulate/comparison.csv` |
| `emulate-validate` | `evaluate.py` | one run + test renders | `results/emulate/<name>/per_device_esr.csv` |
| `emulate-demo` | `evaluate.py` | a run + test renders | `results/emulate/demos/*.wav` |
| `encode` | `encoder/train.py` | `renders/` + `corpus.parquet` (sweep source) | `results/encoder/<name>/{checkpoint,last}.pt`, `config.yaml`, `metrics.json`, `encoder_log.csv` |
| `encode-eval` | `encoder/evaluate.py` | one encoder run + renders | `results/encoder/<name>/{diagnostics.json,capture_embeddings.npz,tsne.csv,tsne.png}` |

- One conditioned model emulates **all** devices, steered by a learnable per-device embedding (conditioning at every layer). Five architectures share the same contract, selected by `emulate.arch` (`models.build_model`):
  - **FiLM-TCN** (`tcn.py`) — fully parametric: `blocks`, `layers_per_block`, `channels`, `kernel_size`, `dilation_growth`, `embedding_dim` are all plain config knobs.
  - **FiLM-WaveNet** (`wavenet.py`) — the exact NAM A2 topology every corpus capture uses (23 dilated layers, receptive field 6347), device FiLM at the schema's pre-activation hook. Knobs: `wn_channels`, `wn_activation` (`leakyrelu` — the captures' own — or `tanh`).
  - **MLP-FiLM WaveNet** (`mlpfilm_wavenet`, same file/topology) — each layer's embedding → (γ, β) generator is an independent `Linear(E, cond_hidden) → cond_activation → Linear(cond_hidden, 2C)` instead of one `Linear(E, 2C)`, so a device's position in embedding space steers the network nonlinearly.
  - **Delta WaveNet** (`delta_wavenet`, same file/topology) — no FiLM: the embedding generates a rank-`delta_rank` residual on each layer's **own weights**, `z = conv_{W + dW(e), b + db(e)}(x) + (mixin_w + dm(e))·clean`. FiLM is the special case `dW = (γ−1)·W`, `db = (γ−1)·b + β`, `dm = (γ−1)·mixin_w` — a strict superset of the FiLM hook, so a device can retime a filter rather than only re-gain it.
    - The residual is `scale · coeff(e) @ normalize(basis)`: unit-norm directions with magnitude held in one scalar per layer. Without the split, magnitude spreads through `basis` for free and the optimizer rebuilds each device's kernel from its own residual — measured at 5.1× the base weight before the split was introduced.
  - **Table-Delta WaveNet** (`tabledelta_wavenet`, same file/topology) — the full-rank control for the above: the same `dW`/`db`/`dm` residual, but looked up whole from a `[N_devices, 10352]` table rather than generated, so nothing is shared between devices except the base kernels. Its `embedding_dim` is **derived** from the schedule (`ecfg.embedding_dim` is ignored), it has no capacity knob, and parameters scale with the device count (4.2M at 405). `set_delta_parts(kernel=…, bias=…, mixin=…)` masks the residual at inference — an analysis knob for which slice carries the amp, honoured by the fold so a masked capture matches what you heard.
  - All five fold into a plugin-playable A2 capture — the conditioning generator never runs in the real-time DSP loop (for the two delta archs, the fold *is* the addition the layer would have done). A sweep is copy-a-config-and-change-numbers, no code change.
  - The first four share one `[N_devices × embedding_dim]` table that is a *map* into network behaviour, so enrollment, morphing and profile export are architecture-independent. `tabledelta_wavenet` is the exception by construction: its rows are per-device weight blobs with no shared structure, so an unseen device has nothing to be positioned within and `emulate-enroll` refuses it outright (`enroll.require_enrollable`) rather than fitting a table the network never reads.
- `dataset.py` serves clean-in / render-out pairs, with the receptive field of **real left-context** prefixed for warmup (loss only on the warmed region).
- `train.py` optimizes pre-emphasized ESR + multi-resolution STFT (auraloss) with Adam + reduce-on-plateau, runs the sanity ladder (`--overfit`, `--limit-devices`), and saves the embedding table + its manifest hash separately (Phase 5 extends it).
- `evaluate.py` appends one test-split row per run to `comparison.csv` (the architecture-exploration deliverable), breaks a single run back out per amp into `per_device_esr.csv` (every device scored on the *same* window grid, so rows rank amps against each other), and exports clean/target/prediction listening demos.

### Tone encoder (`encoder*.py`) — the amortized counterpart to enrollment

Enrollment obtains a device's conditioning vector by gradient descent against a frozen network (up to 30 epochs per amp). The tone encoder is the feed-forward version of that question: a map from the capture audio itself to a tone vector.

- **Input is a 2-channel `(dry, wet)` pair**, not just the wet. An amp is a transfer *relation*, so the encoder is shown "this input produced that output". Corpus renders are sample-aligned with their clean source by construction, so no alignment step is needed (`enroll.blip_lag` is only for real hardware captures).
- **Two branches over one strided stem** (256× downsample → 187.5 Hz frames): a **static** branch (3 dilated blocks, RF 29 frames = 155 ms, mean-pooled) for spectral character, and a **dynamic** branch (8 blocks, RF 1021 frames = 5.45 s, pooled as `mean, std, |mean(Δ)|, std(Δ)`) for attack/sustain/compression. Each is L2-normalized **separately** before the concat, so neither dominates the 512-d vector by norm.
- **Trained with SupCon on `device_id`** (= one capture = one amp at one setting), plus a Barlow-Twins style cross-covariance term that stops the dynamic branch re-deriving static character, plus two low-weight probes that give each branch a meaning: `tone_id` (the *amp*, 204 groups over 450 captures) on the static side and a measured crest-factor delta on the dynamic side.
- **Batches are P captures × K segments**, built into the dataset index rather than a Sampler, so the SupCon positives exist by construction and items stay a pure function of `(seed, i)`.
- **The train/val time split is interleaved, not a tail** (`time_split_regions`). The capture signal's level falls across its length, so holding out the end validates the amp at a 14.3 dB quieter operating point rather than on held-out audio. `encode` measures the realized train/val level gap at startup and warns past 3 dB.
- **`encode-eval` is the gate**, and it is where the design is falsified rather than tuned: retrieval on unseen captures, a dry-channel ablation, branch redundancy, and the two probes. Nothing consumes the 512-d vector yet — wiring it into a generator is a separate decision that this gate exists to inform.

---

## Shared foundation — `openamp.core` and `openamp.dsp`

No pipeline logic here; every stage imports these.

- **core/config.py** — one `Config` dataclass for the whole run.
  - API access (key, base URL, token path, rate limit) from environment / `.env`.
  - Pipeline knobs (`corpus`/`render`/`emulate`/`encoder`) from `configs/openamp.yaml` (per-run files under `configs/emulate/` and `configs/encoder/` replace just their own section).
  - Every path derives from one `data_dir` + one `results_dir`; all randomness derives from one `seed`.
  - Defaults live once, on the dataclass fields — `load_config` only overrides what a source provides.
- **core/constants.py** — fixed audio facts (sample rate, clip length, normalization/verification thresholds) and shared vocabulary: split labels, **gain-bucket names** (`clean`/`crunch`/`high_gain`), render statuses, manifest filenames. Kept in one place so the clip grid, render slicing, and datasets agree by construction.
- **core/manifest.py** — schema and atomic parquet I/O for every manifest (candidates, acquisition, corpus, clips, renders, devices_final).
- **core/util.py** — `sha256_file`, `utc_now`.
- **core/selection.py** — schema-agnostic diversity-selection engine (`two_phase_select`: gain-floor Phase A + brand round-robin Phase B, under per-tone/creator/make-model caps). Shared by `acquire/select.py` (over candidate models) and `corpus/subset.py` (over the finalized manifest); each passes in its own id column and tie-break key.
- **dsp/audio.py** — the one place that depends on `soundfile`/`torchaudio`: read/write/seek-read, resampling, level (RMS/peak dBFS) and error (ESR) metrics, the deterministic 5 s validation probe (sweep + silence + DI) used by acquisition `validate`/`dedup`.
- **dsp/nam.py** — the one place that depends on the `neural-amp-modeler` package and its version-varying internal API: parse a `.nam`, reconstruct the net, run inference (CPU probe for `validate`, GPU forward + receptive field for `render`). Isolated so NAM version drift touches only this file.
  - Owns both capture architectures: an **A2** `SlimmableContainer` resolves to its full-width variant; a legacy **A1** export is upgraded to the current schema on load.
  - Requires NAM >= 0.13.

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
emulate/<name>/per_device_esr.csv  one test-split row per device for that run (per-amp breakdown)
emulate/demos/*.wav            clean/target/prediction listening demos
encoder/<name>/                one tone-encoder run (checkpoint, config, metrics, encoder_log.csv)
encoder/<name>/diagnostics.json    the six-check gate; capture_embeddings.npz + tsne.* beside it
```

## Conventions

- **Determinism**: every stage is seeded from `config.seed`; re-running a stage with the same inputs reproduces its output. Selection and splits use a total order where possible so results don't depend on RNG.
- **Resumability**: stages check for existing outputs and skip completed work (per-row for manifests, per-device for renders, per-file for the corpus).
- **Atomic writes**: manifests, audio, and checkpoints are written to a temp sibling then renamed, so a crash never leaves a half-written file that resume logic would mistake for complete.
