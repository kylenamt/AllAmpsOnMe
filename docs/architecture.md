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

  spec.py                 # the resolved experiment: RunSpec + its five sections
  data/                   # datasets and dataloaders (registry: `data.dataset`)
    corpus_index.py       #   CorpusPaths, the device index, the holdout file
    emulation.py          #   clean-in / render-out pairs, real left-context warmup
    grid.py               #   the same fixed window grid for every device
    joint.py              #   a window plus a disjoint reference segment
    pair.py               #   one (dry, wet) recording, for pair enrollment
    amps.py               #   one amp per item: fit clips + score clips, for the unfolder
    loaders.py            #   make_loader: seeded, worker-safe DataLoader

  models/                 # conditioned architectures (registry: `model.arch`)
    blocks.py             #   activations, per-sample conv, normalized embedding,
                          #   the FiLM and weight-delta generators
    a2_wavenet.py         #   the NAM A2 topology in its three conditioned variants
    joint.py              #   JointModel: encoder -> embedding -> generator

  encoders/               # conditioning encoders (registry: `joint.encoder.kind`)
    pooling.py            #   attentive stats pooling, cross-attention, the frame store
    wavenet.py            #   encode a (dry, wet) window with a non-causal A2 backbone
    fingerprint.py        #   look up a frozen codec's pooled vector by device id
    frames.py             #   pool a frozen codec's raw frames with a trained head

  losses.py               # the objectives (registry: `loss.name`) + device weighting

  training/               # the two loops and the bookkeeping they share
    artifacts.py          #   RunDir: everything a run writes; the resume check
    optim.py              #   plateau scheduler, AMP dtype/scaler, grad clip
    emulate.py            #   the table-conditioned loop + sanity ladder
    joint.py              #   the encoder+generator loop + collapse tripwire
    unfold.py             #   the learned-enrollment-steps loop; both networks frozen

  evaluation/             # scoring a finished run
    metrics.py            #   the two ESR definitions every caller shares
    emulate.py            #   load_model, comparison.csv, per-amp ESR, demo WAVs
    joint.py              #   collapse and leakage diagnostics
    geometry.py           #   k-NN retrieval + t-SNE writer

  enroll/                 # fit an embedding for an amp the network never saw
    fit.py                #   the fitting loop; corpus holdout and (dry, wet) pair
    alignment.py          #   reamp latency, capture regions, chunked rendering
    unfold.py             #   deep unfolding: encoder guess + K learned gradient steps

  export.py               # plugin weight bundle + pasteable profiles (weight folding)

  probes/                 # standalone feasibility probes, wired into nothing
    extract.py            #   frozen DAC over each render, frames kept
    probe.py              #   cross-attention pooler regressed onto a trained table
    probe_attn.py         #   same regression, the shipped AttentionEncoder instead
    probe_esr.py          #   a trained probe_attn encoder -> frozen decoder, seen vs unseen ESR
```

- `core` and (mostly) `acquire` load on a bare numpy/pandas stack; everything from `spec.py` down pulls torch / soundfile / auraloss.
- **The training half is five seams**, each a registry plus a contract plus a config section that names the implementation: `data`, `models`, `encoders`, `losses`, and the objective's own `training` knobs. Adding one is a new file and a config line; the loops name no implementation. See the worked example in [README.md](../README.md#adding-an-architecture).
- **`probes/` is standalone**: nothing imports from it, and the dependency runs one way — it reuses `data`, `evaluation`, `encoders.pooling` and (`probe_attn.py`) `encoders.refine`.
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

### C. Training — `openamp.training`

| Verb | Module | Reads | Writes |
|---|---|---|---|
| `emulate` | `training/emulate.py` | `renders/`, `renders.parquet`, `corpus.parquet` | `results/emulate/<name>/{checkpoint,last}.pt`, `resolved_config.yaml`, `metrics.json`, `train_log.csv`, `embedding.pt` |
| `joint` | `training/joint.py` | the same, plus the encoder's fingerprints/frames | `results/joint/<name>/…` (same set, no `embedding.pt`) |
| `emulate-compare` | `evaluation/emulate.py` | run dirs + test renders | `results/emulate/comparison.csv` |
| `emulate-validate` | `evaluation/emulate.py` | one run + test renders | `results/emulate/<name>/per_device_esr.csv` |
| `emulate-demo` | `evaluation/emulate.py` | a run + test renders | `results/emulate/demos/*.wav` |
| `joint-eval` | `evaluation/joint.py` | one joint run + renders | `results/joint/<name>/{diagnostics.json,tsne.csv}` |
| `emulate-enroll` | `enroll/fit.py` | a finished run + a held-out device's renders | `<run>/enroll/{enrolled_embeddings.pt,enrollment.csv,metrics.json}` |
| `unfold` | `training/unfold.py` | a finished emulate run (generator) + a finished joint run (encoder) + renders | `results/unfold/<name>/{checkpoint,last}.pt`, `train_log.csv`, `metrics.json`, `report_per_device.csv`, `encoder_vectors.pt` |

**One experiment is one `RunSpec`.** `openamp emulate --config configs/emulate/<name>.yaml`
reads that file once, in `spec.load_run_spec`, and everything below takes the spec.
Its sections each name an implementation in a registry:

| Section | Registry | Contract lives in |
|---|---|---|
| `model:` | `models.MODELS` | `models/__init__.py` |
| `data:` | `data.DATASETS` | `data/__init__.py` |
| `loss:` | `losses.LOSSES` | `losses.py` |
| `joint.encoder:` | `encoders.ENCODERS` | `encoders/__init__.py` |
| `training:` | — (the loop's own knobs) | `spec.TrainingSpec` |
| `unfold:` | — (two finished run dirs + the unfolder's knobs) | `spec.UnfoldSpec` |

`model.kwargs` and `loss.kwargs` are opaque and reach the registered class
untouched, so a config uses constructor parameter names and a new implementation
needs no schema change. See [README.md](../README.md#adding-an-architecture).

#### The architectures

One conditioned model emulates **all** devices, steered by a learnable per-device
embedding fed to every layer. Three share the NAM A2 topology every corpus capture
uses (23 dilated layers, receptive field 6347) and differ only in how the embedding
reaches a layer:

- **`film_wavenet`** — device FiLM at the schema's pre-activation hook, generated by
  one `Linear(E, 2C)` per layer.
- **`mlpfilm_wavenet`** — the same hook, but each layer's generator is an independent
  `Linear(E, H) → act → Linear(H, 2C)`, so a device's position in embedding space
  steers the network nonlinearly. A separate arch, not a knob: the two have
  incompatible `state_dict`s.
- **`delta_wavenet`** — no FiLM. The embedding generates a rank-`delta_rank` residual
  on each layer's **own weights**, `z = conv_{W + dW(e), b + db(e)}(x) + (mixin_w + dm(e))·clean`.
  FiLM is the special case `dW = (γ−1)·W`, so this is a strict superset of the hook:
  a device can retime a filter rather than only re-gain it.
  - The residual is `scale · coeff(e) @ normalize(basis)` — unit-norm directions with
    magnitude in one scalar per layer. Without that split, magnitude spreads through
    `basis` for free and the optimizer rebuilds each device's kernel from its own
    residual, measured at 5.1× the base weight.

All three fold into a plugin-playable A2 capture: the conditioning generator never
runs in the real-time DSP loop, and for `delta_wavenet` the fold *is* the addition the
layer would have done. All three share one `[N_devices × embedding_dim]` table that is
a *map* into network behaviour, so enrollment, morphing and profile export are
architecture-independent.

#### The two loops

- **`training/emulate.py`** conditions on a table row. Optimizes pre-emphasized ESR +
  multi-resolution STFT with Adam + reduce-on-plateau, runs the sanity ladder
  (`--overfit`, `--limit-devices`), saves the embedding table with its manifest hash.
- **`training/joint.py`** conditions on an encoder's output instead, with the table
  frozen and unused. Two parameter groups — the encoder needs its own learning rate
  or it collapses to silence — and a per-epoch collapse tripwire
  (`val_esr_shuffled`, `emb_spread`), because an encoder emitting a near-constant
  vector still posts a respectable ESR. AdamW here, not emulate's Adam, because
  `enc_weight_decay` is a real value under test.

- **`training/unfold.py`** trains neither network. Both are finished runs, frozen —
  the generator an emulate run, the first guess a joint run's encoder — and what
  trains is the *enrollment steps between them* (see below). Its `model`/`data`/`loss`
  come from the generator run, not the config.

The first two stay two loops on purpose: they disagree about what validation *means*, and one
loop with flags would hide that rather than remove it. What they genuinely shared —
run-directory I/O, the resume check, the schedule/AMP setup — is `training/artifacts.py`
and `training/optim.py`. `training.scheduler` picks `plateau` (the default, on val ESR) or
`cosine` (on epoch count, each group annealing to `cosine_min_lr_frac` of its own LR).

#### Conditioning encoders (`openamp.encoders`)

Enrollment obtains a device's vector by gradient descent against a frozen network. An
encoder is the feed-forward version of the same question, trained jointly with the
generator. Three, differing in what evidence they read:

- **`wavenet`** — encodes a short `(dry, wet)` reference window with a non-causal A2
  backbone. Two input channels because an amp is a transfer *relation*.
- **`fingerprint`** — looks up a frozen codec's pooled latent by device id. A codec
  fingerprint needs long audio to be a reliable statistic (DAC own-nearest accuracy
  falls from 99.8% at 85 s to 17.8% at 12 s), well above a reference window.
- **`dac_frames`** — looks up that codec's *raw frames* and pools them with a trained
  cross-attention head, which is the one thing it changes relative to `fingerprint`.

`ref_mode` on each class says what the dataset must yield — a real window, or a device
id to look up — so the training and evaluation loops agree without either naming a kind.

What makes an embedding carry *tone* rather than *audio*: the reference is a different
segment of the recording, in practice a different corpus file, than the one the
generator must reconstruct. Only what survives that gap is the amp.

#### Unfolded enrollment (`openamp.enroll.unfold`)

Between the two ways of getting an unseen amp's vector — enrollment (thousands of
gradient steps, close) and an encoder (one forward pass, far) — the unfolder starts
from the encoder's guess and takes K gradient steps whose shape is *learned*.

- A `delta_wavenet` reads `e` only through one linear map per layer
  (`c_l = W_l e + b_l`), so `dL/de = Σ_l W_lᵀ g_l` exactly. Each unfolded step
  replaces `W_lᵀ` with its own trainable copy `V_{k,l}` and adds a per-dimension
  step size: `e ← P(e − s_k ⊙ Σ_l V_{k,l}ᵀ ĝ_l)`, with `ĝ` the coeff gradient
  normalized per amp and `P` the generator's own projection back onto its sphere.
- `V` starts at `W` and the step sizes are calibrated to one length, so an untrained
  unfolder *is* normalized gradient descent.
- **First-order**: each `g` is a constant to the outer loss, so memory is one
  generator pass at any K.
- One step costs one generator forward + backward on the amp's fit clips; K steps
  cost K of them. Only `delta_wavenet` is supported — other archs have no single
  linear map to learn around.

---

## Shared foundation — `openamp.core` and `openamp.dsp`

No pipeline logic here; every stage imports these.

- **core/config.py** — the `Config` the *pipeline* stages need. Training is not here.
  - API access (key, base URL, token path, rate limit) from environment / `.env`.
  - Acquisition/corpus/render knobs from `configs/openamp.yaml`.
  - Every path derives from one `data_dir` + one `results_dir`; all randomness derives from one `seed`.
  - Defaults live once, on the dataclass fields — `load_config` only overrides what a source provides.
- **spec.py** — the `RunSpec` a *training* run is. Read from its own
  `configs/{emulate,joint}/<name>.yaml` by `load_run_spec`, and written back to
  `<run>/resolved_config.yaml` so a finished run records the corpus, seed and name
  it actually used rather than leaving them to a shell prefix.
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
emulate/<name>/                one emulation run: {checkpoint,last}.pt, resolved_config.yaml,
                               config_history.jsonl, metrics.json, train_log.csv, embedding.pt
emulate/comparison.csv         one test-split row per emulation run (size-sweep deliverable)
emulate/<name>/per_device_esr.csv  one test-split row per device for that run (per-amp breakdown)
emulate/<name>/enroll/         fitted embeddings for held-out devices + enrollment.csv
emulate/demos/*.wav            clean/target/prediction listening demos
joint/<name>/                  one joint run, same layout minus embedding.pt
joint/<name>/diagnostics.json  the collapse + leakage gate; tsne.csv beside it
```

## Conventions

- **Determinism**: every stage is seeded from `config.seed`; re-running a stage with the same inputs reproduces its output. Selection and splits use a total order where possible so results don't depend on RNG.
- **Resumability**: stages check for existing outputs and skip completed work (per-row for manifests, per-device for renders, per-file for the corpus).
- **Atomic writes**: manifests, audio, and checkpoints are written to a temp sibling then renamed, so a crash never leaves a half-written file that resume logic would mistake for complete.
