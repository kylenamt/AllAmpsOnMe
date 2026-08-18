# Decisions — why the repo is the way it is

A repo-wide audit of every design choice, trap, and result, written so nobody (human or
assistant) has to re-derive them by backtracking through code and logs again. Casual on
purpose; the file-level docstrings stay the formal reference. Current as of 2026-08-11.

The project recreates the [Open-Amp paper](https://arxiv.org/abs/2411.14972) on
[TONE3000](https://www.tone3000.com) community captures: acquire ~450 amp captures, render a
clean guitar corpus through each one, then train **one** conditioned network that emulates all
of them, steered by a per-device embedding. A new amp is added later by fitting *only* an
embedding vector against the frozen network ("enrollment"), and the whole thing folds into a
stock NAM A2 capture that a real-time plugin can play.

---

## 1. The plumbing everything sits on

- **Everything is files.** Each pipeline stage reads and writes parquet manifests + audio
  under one data root, so every stage is resumable, idempotent, and inspectable with pandas.
  There is no database and no hidden state.
- **One `Config`, one seed.** `core/config.py` merges `.env` + `configs/openamp.yaml` into a
  single dataclass. Every path derives from `data_dir` + `results_dir`; every RNG derives
  from `seed: 1234`. Defaults live once, on the dataclass fields — yaml/env only override
  what they actually provide.
- **`OPENAMP_DATA_DIR` picks the data root, but results always land in `cwd/results`.**
  That asymmetry is load-bearing: corpus experiments live in separate data dirs
  (`data_sweep/`, `data_mixed/`, …) while all runs share one results tree and one
  `comparison.csv`. It also means **the env var is part of a run's recipe** — launch a
  `*_sweep` or `ch16_512` config without it and you silently train a different experiment
  under the same name. Every affected config screams this in its header.
- **Separate data dirs instead of mutating `data/`.** `data/` (the EGDB corpus and its
  ~106 GB of renders) is treated as immutable once runs have trained on it, so the eight
  historical runs stay reproducible and `--resume`-able. New corpus ideas get their own dir
  built by a script (`scripts/build_sweep_corpus.py`, `scripts/build_mixed_corpus.py`),
  usually via symlinks back into `data/` so only genuinely new bytes cost disk.
- **Atomic writes everywhere.** Manifests, FLACs, and checkpoints all write to a temp
  sibling then `rename()`, so a crash can never leave a half-written file that resume logic
  mistakes for complete.
- **Heavy imports are local.** torch / soundfile / NAM import inside function bodies, so
  `openamp --help` is instant and the unit tests run on a bare numpy/pandas stack.
  `dsp/nam.py` is the *single* file allowed to touch NAM's version-unstable internals.
- **Per-run emulate configs replace, not merge.** `configs/emulate/<name>.yaml` is read
  *instead of* the base `emulate:` section — any key you omit falls back to the dataclass
  default (paper-TCN numbers: `stft_weight 1.0`, `lr 5e-4`, `weight_decay 0`), silently
  undoing the NAM recipe. So every config is a **full copy** with numbers changed, never a
  delta. The run name is the config file stem.
- **PyYAML trap, forever enshrined:** bare `4e-3` parses as a *string*. Every config writes
  `lr: 4.0e-3` with the decimal point and a comment saying why.

## 2. Acquisition (frozen — the asset is the manifest, not the code)

- **Official API only**, OAuth+PKCE, client-side rate limit at 80 req/min under the server's
  100, licenses recorded verbatim, capture files never redistributed.
- **Diversity is engineered, not hoped for.** One shared selection engine
  (`core/selection.py`): per-tone / per-creator / per-(make,model) caps, a minimum share per
  gain bucket (clean/crunch/high-gain, assigned by a keyword heuristic in
  `acquire/catalog.py`), then brand round-robin fill. Deterministic — rows are ordered by
  downloads with a total tie-break, so no RNG is needed. Acquisition uses caps 2/8/6 with a
  20% gain floor; the render subset reuses the same engine with looser caps 4/12/6 and a 15%
  floor, because drawing ~450 from an already-curated 844 exhausts the clean-amp pool under
  the strict caps.
- **The architecture filter must go to BOTH endpoints.** `/models` defaults to A1 and does
  not inherit the search's filter — which is exactly how the original 844-device corpus
  ended up entirely A1. `acquire.architecture: 2` is now passed to both.
- **`device_id` is sacred.** `finalize` assigns it once (rank by make, model, model_id) and
  it never changes: it is the embedding-table index and the address of ~106 GB of rendered
  audio. This is also why `migrate-a2` rewrites the final manifest **in place** rather than
  re-running select/finalize — re-deriving ids would orphan the renders.
- **`migrate-a2` pairs by *exact* model name**, never similarity (the models under one tone
  are the same amp at different gain settings, so a near-miss would silently swap an amp's
  identity). No A2 twin → keep A1, flag `architecture_fallback` (42 devices). Old renders are
  *not* invalidated; `renders.parquet` records the `nam_sha256` each was made from. The
  pre-migration manifest is backed up to `manifest.a1.parquet`.
- **Acceptance requires attribution, not a license string** — TONE3000's API returns an
  empty license for everything, so requiring one would reject the whole corpus.
- **Validation/dedup** run a fixed deterministic 5 s probe (log sweep + silence + synthetic
  DI) through each capture: output must be finite, audible, non-exploding, quiet-in-silence.
  Dedup drops exact-hash duplicates globally, then probe-ESR < 0.01 within a (make, model)
  group, keeping the more-downloaded capture.

## 3. Clean corpus and rendering

- **Corpus recipe:** 48 kHz mono float32; normalize each EGDB file to −18 dBFS RMS backed
  off to a −1 dBFS true-peak cap; select files to a 40-minute budget; split **by file**
  (never by clip) so no source audio leaks across train/val/test; 2 s clip grid, trailing
  remainder dropped. Sweep → train only; test is EGDB (realistic playing) only.
- **Renders are whole-file, then sliced.** `chunked_forward` streams a file through the
  model in 10 s chunks carrying the receptive field of left-context and discarding the
  warmup — **bit-identical to a single full-file forward** (unit-tested against an FIR
  stand-in), so there are no clip-boundary transients anywhere in the training data.
- **`output_scale`: one scalar per device, peak pinned at 0.999.** A high-gain capture's raw
  output can peak well above 1.0, which 24-bit FLAC can't store; the render job makes two
  passes (measure global peak, then scale everything by one number). Consequences we now
  understand well:
  - the (clean → render) pair stays a valid function — the scalar is just "the amp's master
    volume turned down once";
  - relative loudness *between* devices is compressed (each device is scaled by its own
    number), and
  - the scalar lives in `renders.parquet` to 8 dp, which is what later made corpus
    re-assembly possible **without re-rendering** (see §9).
- **`verify` is the gate to `devices_final.parquet`**: completeness, alignment
  (NAM forwards are same-length; any mismatch is a bug), and per-signal sanity — non-finite,
  silent (< −70 dBFS), or pass-through (ESR(out, in) ≤ 0.001) all fail the device.

## 4. The model family

Five architectures, one contract (`forward(x, device_idx) → [B,1,T]` causal, a
`receptive_field` property, an `embedding` table), one dispatch point in
`emulate/models.py`. Adding an arch = one class + one branch.

- **`film_tcn`** — the paper's fully-parametric FiLM-TCN. Kept as the reference; it lost to
  the WaveNet line early (test ESR means 0.15–0.7 vs 0.075) and hasn't been revisited.
- **`film_wavenet`** — the corpus's own capture topology: the exact A2 schedule (23 dilated
  layers, kernels 6/15, receptive field 6347 ≈ 132 ms), verified identical across all 774 A2
  exports, with FiLM activated at the pre-activation hook NAM's schema already defines.
  Two deliberate departures from a capture: FiLM conditioning (starts near-identity, so an
  untrained net *is* a plain A2 WaveNet), and causal same-length padding instead of NAM's
  valid-conv-and-trim (sample-identical once warmed).
- **`mlpfilm_wavenet`** — same network, but each layer's embedding → (γ, β) map is a small
  per-layer MLP instead of one Linear. Asked whether a *nonlinear* read of embedding space
  helps. Answer: no (see the ledger, §11).
- **`delta_wavenet`** — conditioning moved one step upstream: no FiLM; the embedding
  generates a low-rank **residual on each layer's own conv weights**
  (`delta = scale · coeff(e) @ normalize(basis)`). FiLM is provably a special case of this
  family, so a win here is about the hook, not capacity. The direction/magnitude split is a
  hard-won constant: the first run had no `scale`, magnitude leaked diffusely through
  `basis`, and by epoch 12 the "residual" was 5.1× the base weights — each device had
  quietly rebuilt its own private kernel, exactly what the shared space exists to avoid.
  Unit-norm basis + one scalar per layer makes delta growth cost one visible, decayable
  parameter. Batched training uses a grouped-conv trick (`per_sample_conv1d`) so every batch
  item convolves with its own weights in one kernel launch.
- **`tabledelta_wavenet`** — the full-rank control: the same residual, looked up whole from
  a `[N_devices × 10,352]` zero-init table, nothing shared but the base kernels.
  `embedding_dim` is derived, not configured; it **cannot be enrolled** (no shared structure
  to place a new amp in), and `enroll.py` refuses it *explicitly* — because it would not
  fail naturally: the fit would run and report a plausible ESR for a table the network never
  reads. A `set_delta_parts(kernel/bias/mixin)` inference mask lets one checkpoint answer
  "is the amp in the filter shape or just the levels?", and the export fold honours it.
- **`head_scale` is a fixed buffer, not a parameter.** Made learnable, Adam moves this one
  global-gain scalar ~lr per step and drives the model into the silence solution (output 0,
  ESR 1.0) before anything else can fit. Observed, not theorized.
- **Everything folds into a stock A2 capture** (`emulate/export.py`). For a fixed embedding,
  FiLM is absorbed into conv/mixin weights; for the delta archs the fold *is* the addition.
  The conditioning generator never runs in the plugin's DSP loop — it re-folds only when the
  morph point moves. Format decisions with teeth: non-FiLM bundles deliberately **omit**
  `film_w`/`film_b` so an old reader dies on a missing key instead of folding garbage and
  playing the wrong amp at full volume; profiles carry the checkpoint's sha8 so the plugin
  can reject embeddings enrolled against a different network.

## 5. Training

- **The loss** is pre-emphasized ESR (coeff 0.85) + auraloss MultiResolutionSTFT.
  ESR is **pooled** — one ratio of sums over the batch, not a mean of per-window ratios —
  because window energy spans ~8 orders of magnitude in clean guitar and per-window ratios
  let near-silent windows dominate. Known cost of that pooling: effective weight per device
  scales with its render energy, a measured **4,335× spread** across devices, and quiet
  devices are measurably learned worse (ESR vs loudness spearman −0.23…−0.43 across five
  finished runs). A per-device reference-energy loss has been proposed but not run — it
  changes the objective, so it's a new experiment, not a patch.
- **`stft_weight: 0.05` is a derived number, not a vibe.** NAM regularizes with MRSTFT at
  5e-4 *against MSE*; ESR = MSE / mean(t²), and the EGDB renders sit at median
  mean(t²) ≈ 0.0096, so 5e-4 / 0.0096 ≈ 0.05 — twenty times below the paper default of 1.0.
  The IQR of that estimate is 0.025–0.13: treat it as an order of magnitude. It is
  deliberately held at 0.05 even on corpora where the median energy differs, so corpus
  comparisons don't entangle a loss-weight change.
- **NAM's recipe, verbatim where a knob exists** (`nam_a2.yaml` documents every mapping):
  lr 4e-3, weight_decay 3.17e-7, batch 16 (→ 32 in later runs: the model is *launch-bound*,
  not FLOP-bound, so bigger batches buy wall-clock the GPU was idling through; lr stays at
  4e-3 because 1e-2 diverged), 100 epochs. Three parts of NAM's recipe are code, not config,
  and differ knowingly: ReduceLROnPlateau instead of ExponentialLR-0.994 (watch that lr
  doesn't collapse early), ESR instead of MSE (absorbed by `stft_weight`), and 2 s clips
  instead of 0.17 s segments.
- **bf16, not fp16.** fp16 forward overflow NaN-killed two runs; bf16 has fp32's exponent
  range so it needs no loss scaling. A NaN guard recovers instead of dying: non-finite val →
  reload best + drop to fp32 (or halve lr); non-finite train with finite val → the scaler
  absorbed it, just drop to fp32. Grad-clip 20 is a spike safety net (healthy norms sit at
  p50≈4), not a tuning knob.
- **Resume semantics:** the config is authoritative — lr/weight_decay from the file
  overwrite the checkpoint's optimizer values (that's how a run is hand-annealed), while
  Adam's moment buffers are preserved (that's the point of resuming). Structural knobs are
  frozen per run and checked before the state_dict load; `wn_activation` is structural
  *despite being parameter-free*, because swapping it loads cleanly and silently plays a
  different network. `config_history.jsonl` appends one line per start/resume as a
  never-overwritten audit trail.
- **The dataset** draws random (device, file, window) triples, prefixes each clip with the
  receptive field of **real left-context**, and computes loss only on warmed samples.
  Draws are uniform over *files*, not duration-weighted — which is why the 171 s sweep file
  is only 1.4% of training windows in `data_mixed` (a deliberate choice: keep the sampler
  out of the variable set). NFS read flakes retry the *same* item twice before redrawing
  (the val set must not silently swap items), and four consecutive failures raise — that's
  an outage, not a flake.
- **Device holdout:** 10% of render-ok devices (45 ids) held out of training entirely for
  enrollment, drawn once and persisted — **the file is the source of truth**, never the RNG.
  Holdout ids travel in every checkpoint. Flip side: they're the default `emulate-enroll`
  target list, so a device parked in the holdout file for *exclusion* reasons (486) will be
  enrolled as if it were a clean test subject unless `--devices` is passed explicitly.
- **Sanity ladder before any long run:** `--overfit` one batch to ~0 ESR, a
  `--limit-devices` mini-run, and every full run ends by reporting `val_esr_shuffled` —
  val ESR with embeddings permuted across devices. Shuffled sits at ~1.85–1.92 vs real
  0.03–0.12 on every run, which is the standing proof that conditioning does the work.

## 6. Three different ESRs — don't cross-compare them

All called "ESR", all different numbers. This has burned us before:

| Where | Definition | Quirks |
|---|---|---|
| training / `val_esr` | **pooled** ratio-of-sums, **pre-emphasized** (0.85) | what early stopping and `best_val_esr` use |
| `comparison.csv` test | **mean of per-window raw ratios**, silence-gated at −50 dBFS | per-window means are outlier-sensitive (see `embed256`'s mean 0.71 vs median 0.045) |
| `per_device_esr.csv` | pooled ratio-of-sums **per device**, raw, on a **shared window grid** | rows are comparable *between* devices precisely because the grid is fixed |

The shared-grid decision matters: with independent random windows, a device can look good or
bad on window luck alone. The grid is drawn once from the seed, silence-screened on the
*clean* side (so it stays identical across devices), and replayed for every device.

**⚠ `emulate-validate` / `emulate-demo` silently use whatever `OPENAMP_DATA_DIR` says at call
time — and it bit us.** `results/emulate/nam_a2_256_sweep/per_device_esr.csv` (2026-08-10) was
generated **without** `OPENAMP_DATA_DIR=data_sweep`, so it scored the sweep-trained model
against EGDB audio and A1-capture targets. Proof: its per-device values correlate 0.998
(log-space) with a fresh `data/` evaluation and only 0.693 with a `data_sweep` one. The damage
is not subtle — devices the CSV calls catastrophic (439, 470–473, 649–651, 353–355 at ESR
1.4–2.1) actually measure **ESR 0.012–0.044 on their own corpus**, up to 175× better. Treat
that CSV as a domain-gap measurement, not as "how well the sweep run learned its amps", and
re-run every sweep-run evaluation with the env var. The checkpoint already carries
`manifest_sha256`; `enroll.py` warns when it disagrees with the live manifest but
`evaluate.py` does not — that guard is the obvious fix and is not yet written.

**Why a badly-fit device also goes quiet.** Least squares makes attenuation the optimal hedge
when the model can't predict the waveform, so level and fit are locked together by
`rms_ratio ≈ sqrt(1 − ESR)` (measured correlation 0.73 in-domain). ESR 0.05 → −0.2 dB
(inaudible); 0.5 → −3 dB; 0.8 → −7 dB. So a quiet demo is a *symptom* of underfitting, not a
cause of the ESR: on the correct corpus the level error accounts for only 0.8–2% of median
ESR, and rescaling the output would not meaningfully improve it. In-domain the sweep model is
only −0.27 dB on median, with ~0.6 dB of dynamic compression (quiet frames shrink more than
loud ones — what energy-pooled ESR rewards) and a noise floor it cannot get below (frames
under −40 dB come out +3.6 dB hot). Cross-domain the failure mode changes character: the bad
devices emit roughly the *right* energy but essentially **uncorrelated** waveform (ρ≈0,
optimal rescale ≈ 0), with individual devices landing anywhere from −5 dB (Matchless 649) to
+7 dB (Hartke 433, Lel 3000 486). Not causes, checked and excluded: `output_scale` (1.0 for
all these devices in both corpora) and A1-vs-A2 capture level (median −0.01 dB, only 4 of 408
devices past 3 dB).

## 7. Enrollment (Phase 5)

- Freeze every network weight, swap in a fresh embedding table, fit **only** that against
  the new device's audio. Two front doors, one fitting loop: `emulate-enroll` (holdout
  devices, from their on-disk renders) and `enroll_pair` (one wet/dry recording — the
  TONE3000 capture workflow — driven from a notebook).
- **Init default changed 2026-08-09 to `uniform`** at the trainer's own embedding-init
  variance. The old `table_mean` start looks better (lower baseline ESR) but is a peculiar
  point: in 256-D the mean of 405 rows has ~1/5 the norm of any actual row — a region no
  trained device occupies. A better baseline does not imply a better optimum. Consequence
  for analysis: old enroll artifacts started elsewhere, so compare best val ESR, never the
  epoch −1 baseline.
- **Pair enrollment needs `stft_weight: 0`.** On a single-pair fit the STFT magnitude term
  can dominate and drive the embedding to a spectrally-plausible but waveform-uncorrelated
  optimum (val ESR rises while train loss falls — observed on wavenet_a2).
- **Alignment: first-arrival beats cross-correlation.** The interface round-trip is a pure
  delay; the amp's impulse response starts immediately but has its energy centroid a few
  samples in. Whole-signal xcorr recovers delay *plus* group delay and can't separate them
  (tens of samples of bias — a 37-sample error ~tripled val ESR). `blip_lag` reads the NAM
  calibration blip's leading edge instead (+2 samples on a known-512 test), with whitened
  xcorr as the blip-free fallback.
- `enroll.py` also owns `nam_signal_regions` — NAM's own train/val slicing of the v3
  capture signal (lead-in dropped, blips kept in train, val = the last 9 s), identified by
  sample length.

## 8. The sweep saga (how three extra data dirs happened)

The single most consequential bug in the project, and the chain of decisions it forced:

1. **`NAM_SWEEP_FILENAME` never matched the file on disk** (`v3_0_0.wav` in constants vs
   `T3K-sweep-v3.wav` in `data/raw/nam_sweep/`), so `corpus/build.py` silently set
   `have_sweep = False` and **every emulate run before 2026-08-10 trained sweep-free** —
   while the docs and configs said the corpus included the sweep.
2. **Deliberately not fixed at the root.** Fixing the resolver would make a future
   `openamp corpus --force` silently ingest the whole raw 190 s file — lead-in, calibration
   blips, val block — into the train split with no region slicing. That needs its own
   change; until then the bug is documented here, and `tests/test_config.py` pins the
   current filename so the fix has to be conscious.
3. **The measured damage** (`notebooks/sweep_domain_gap.ipynb`): EGDB-trained models score
   ~3.4× worse on sweep-style guitar DI and **16–70× worse on the chirp/noise content** —
   and it's content, not level. Meanwhile pair-enrollment against the sweep already sits
   inside the network's own per-device floor distribution, so the gap is a corpus property,
   not an enrollment failure.
4. **`data_sweep/`** (built by `scripts/build_sweep_corpus.py`): the capture signal *as its
   own corpus*, sliced NAM's way, at native level (no −18 dBFS normalization — matching
   TONE3000's own recipe is the point), same 450 devices and holdout as `data/` so runs are
   comparable. NAM defines no test split, so `sweep_test` is the *same* 9 s as `sweep_val` —
   test re-measures val there, by construction.
5. **The A1/A2 render split** (found 2026-08-10 by a `nam_sha256` cross-check, the hard
   way): `data/renders` was rendered 2026-07-04 from **A1** captures; `migrate-a2` then
   re-pointed the manifest; `data_sweep/renders` (2026-08-08) used **A2** captures for
   408/450 devices. An A1 and an A2 capture of the same amp are *different models* — mixing
   them puts two target functions behind one device_id. This also means the two `*_sweep`
   runs differ from their EGDB controls in capture architecture as well as corpus, so their
   headers' "the corpus is the only variable" claim is quietly wrong. Decision: leave those
   configs/runs alone; note it here.
6. **`data_sweep_a1/`** (`scripts/stage_sweep_a1.py`): resolves every device's original A1
   capture by hashing `data/captures/**/*.nam` against the render manifest's `nam_sha256`
   (450/450 resolved) and re-renders just the 3 sweep files — the A1-consistent sweep
   render set.
7. **`data_mixed/`** (`scripts/build_mixed_corpus.py`): EGDB + sweep in one corpus,
   **without re-rendering** — a render is exactly `raw × output_scale` and the scalar is in
   the manifest, so the combined per-device scale is recomputed arithmetically and only the
   19 devices whose peak moved get their files rewritten (decode → multiply → re-encode);
   everything else is symlinks into `data/` and `data_sweep_a1/`. Device 486 appended to its
   holdout (404-device table). Verification lessons that now live in the scripts:
   `out_peak_db` is stored to 3 dp (≈5.8e-5 relative error on a recovered peak — never
   quote it back as an achieved peak), and rescaled files are checked as
   `|y − y0·ratio| ≤ one 24-bit LSB` — a *ratio* test divides by near-silent samples and
   turns one quantum into a meaningless 3.7e-2 "error".

## 9. Device 486, and what pooled ESR does to quiet amps

"Lel 3000 1993 — output_amp2" (device 486) renders ~20 dB below the corpus median and scores
test ESR > 1.0 in every run with per-device output — worse than predicting silence. It's a
broken/near-silent *capture*, not a modelling failure, yet it contributed ~7% of
`nam_a2_256`'s headline mean from 1/405 devices. It's now excluded in `data_mixed` via the
holdout file (chosen over adding an `exclude_devices` config knob — no new code path, and
`data/`'s holdout stays untouched so old runs still resume). Remember the flip side from §5:
it thereby became a default enrollment target; pass `--devices` explicitly.

The general form of the problem: pooled ESR weights each device by its render energy
(4,335× spread, quiet quartile 1.7–2.7× worse ESR, spearman −0.23…−0.43 across five runs —
and the confound checks say it's real: the eval metric is per-device normalized, and loud
high-gain amps are intrinsically *harder* yet score better). Fixing it means a per-device
reference energy in the loss denominator; parked as the next experiment.

## 10. Operational lore (the cluster, the GPU, the filesystem)

- Runs span machines on a **shared NFS home**: no local PID ≠ dead run. Never kill, resume,
  or relaunch a run this session didn't start.
- NFS can delay a log write ~1.5 h: **mtime is not a liveness signal** — date the last line
  by its `smp/s` content instead.
- Never launch through `conda run` — it buffers stdout for the whole job and
  `train_stdout.log` stays empty. Use the env's binaries directly
  (`~/anaconda3/envs/open-amp3000/bin/openamp`), under `nohup … >> <run>/train_stdout.log`.
  **`>>` not `>`**: a `>` on a resume truncated 100 epochs of log history once (2026-08-07).
- The base python has no torch (a bare `pytest` exits 5, silently); use the
  `open-amp3000` env. `--overfit` OOMs at batch 32 on the 12 GB card — call
  `overfit_one_batch()` with batch 8 instead. Enrolling against a batch-32 run is fp32 and
  needs ~2× that run's activation memory — drop batch to 8–16.
- `scripts/publish_run_dashboard.py` renders every run's curves + log tail into one static
  HTML file under `~/public_html/<unguessable-token>` — readable from anywhere, no SSH; the
  web root won't follow symlinks into the home volume, so it writes a real file.

## 11. Run ledger (best pooled pre-emph val ESR, 405 devices on `data/` unless noted)

| Run | Arch / knobs | Params | best_val_esr | What it established |
|---|---|---|---|---|
| `paper`, `channels32`, `embed256` | film_tcn | 72k–248k | (test means 0.15–0.71) | TCN line abandoned; WaveNet wins outright |
| `wavenet_a2` | film_wavenet E128 | 111k | 0.1173 | capture-native topology beats TCN at half the params |
| `wavenet_a2_256` | E256 | 210k | 0.1083 | more embedding helps a little |
| `nam_a2_256` | + NAM training recipe | 210k | 0.0876 | NAM's own numbers beat the paper recipe |
| `nam_a2_512` | E512 | 408k | 0.0923 | **embedding 512 hurt at ch8** |
| `nam_a2_tanh_256` | tanh activation | 210k | 0.0796 | tanh > leakyrelu, consistently |
| `nam_a2_tanh_ch16_256` | channels 16 | 340k | **0.0503** | width is the strongest lever |
| `mlpfilm_a2_h16_256` | MLP-FiLM, param-matched | 217k | 0.1045 | nonlinear FiLM generator: **negative** |
| `mlpfilm_a2_256` / `h32_tanh` | 2× params | 317k | 0.0872 / 0.0894 | at best a wash vs plain FiLM |
| `delta_a2_256` (r8) | weight-delta hook | 246k | 0.0639 | conditioning in weight space beats FiLM |
| `delta_a2_r16_256` | rank 16 | 376k | 0.0469 | rank scales well |
| `delta_a2_r32_512` | rank 32, E512 | 928k | 0.0374 | still scaling |
| `delta_a2_tanh_r32_512` | + tanh | 928k | **0.0334** | best *enrollable* model |
| `table_a2_256` | full-rank control | 4.20M | 0.0322 | rank-32 basis ≈ full rank at 4.5× fewer params; table can't enroll |
| `nam_a2_256_sweep` | sweep corpus (`data_sweep`) | 210k | 0.0832 | headline NOT comparable (different val audio); ⚠ A2-capture confound (§8.5) |
| `delta_a2_tanh_r32_512_sweep` | sweep corpus | 928k | still training (e≈115) | same caveats |
| `nam_a2_tanh_ch16_512` | E512 + `data_mixed`, 404 devices | ~600k | in progress (0.0744 @ e46) | three variables change at once — see its config header for the comparison protocol |

Reading notes: `val_esr_shuffled` lands at 1.85–1.92 on every run (conditioning check).
`train_hours` in `metrics.json` covers only the final session of a resumed run. The current
`ch16_512` run's number can only be compared to `ch16_256` by re-validating **both** against
`data/` on shared devices, excluding 486 and the 19 rescaled ids in
`data_mixed/manifests/rescaled_devices.txt`.

## 12. The tone encoder (new direction, 2026-08-11)

The question enrollment answers slowly: *what is this amp's conditioning vector?* `enroll_pair`
fits it by gradient descent against a frozen network, up to 30 epochs per amp. The encoder asks
whether it can be **read off the capture audio** in one forward pass instead. Built standalone and
gated by its own diagnostics — **nothing in the emulators consumes its output**, and wiring it into
a generator is deliberately a separate decision that `encode-eval` exists to inform.

- **`capture_id` and `amp_id` already existed, unnamed.** `device_id` (450) is the capture — one
  `.nam` is one amp at one knob setting — and is the contrastive grouping key. **`tone_id` (204
  groups) is the amp**: a TONE3000 upload holds several captures of the same amp at different
  settings, so 375 of 450 devices sit in groups of 2–4 (devices 0/1/2/6 are one Orange Micro Terror
  at `V10T10G10`, `V12T10G12`, `V12T10G9`, `V9G10G12`). That two-level structure is what the amp
  probe classifies. **`(make, model)` is not usable as amp identity** — `model` is derived from the
  settings string, so it is 449-unique over 450 devices and the probe would just restate the
  contrastive task.
- **Sweep-only, on `data_mixed`.** At inference you hold a capture-signal recording, so training on
  the same content sidesteps the 16–70× EGDB→sweep domain gap (§8.3). `sweep_train` is 171 s = 31
  segments of 5.46 s per device, which is where the architecture's segment length comes from. Cost:
  171 s of train audio per device instead of 31.9 min. `sources: [egdb, sweep]` trades back.
- **Two orthogonal splits, both needed.** The device axis was already spoken for by
  `emulate_holdout.txt` (46 unseen amps → the retrieval test), so train/val is a *time* split of the
  same captures, taken inside `sweep_train` — `sweep_val` (9 s) is too short to give the ≥4 segments
  per capture SupCon needs.
- **A tail split of the capture signal is a trap, and the first full run fell into it.**
  `sweep_train`'s level falls across its 171 s: the first 85% averages −20.98 dBFS, the last 15%
  averages −35.30 dBFS, and that tail's first two windows sit at −57 dBFS, near the render noise
  floor. Holding out the tail therefore does not validate on held-out audio — it validates at an
  operating point 14.3 dB quieter, where a level-dependent nonlinearity is closest to linear and
  every amp most resembles every other. The 2026-08-11 `sweep_base` run was split that way, which is
  the most likely single explanation for its val SupCon sitting above its train SupCon, its amp
  probe plateauing, and its crest probe reading 6× worse on val than train. Fixed by
  `time_split_mode: strided` (default): the file is cut into `time_split_blocks` blocks and
  evenly-spaced ones go to val, so both sides span the whole file. At the default 10 blocks the
  train/val gap is **0.98 dB** instead of 14.3. `encode` now measures and prints that gap at startup
  and warns past 3 dB, because the failure is invisible in the loss curve and depends on the corpus —
  don't re-derive it by hand, read the header. `time_split_blocks: 10` is not a round number chosen
  blind: 12 blocks lands a val block in the decayed part and gives a 3.94 dB gap, 8 gives 8.32.
- **A `--overfit` collapse is not conclusive, and this is where that was learned.** Switching to the
  strided split made the single-batch check collapse — every embedding on one point, SupCon pinned
  at `log(P*K−1)` = 3.4340 — on **4 of 5** torch seeds, where the tail split's batch collapsed on
  0 of 5. That looked damning and was not: the real 404-device run over the same regions descended
  **2.89 → 2.64 → 2.44** over three epochs, comfortably better than the tail run's own 3.23 at
  epoch 0. One fixed batch at full LR is an absorbing state that fresh batches escape.
  **The same thing happens transiently inside a real run and it is not a failure:** under the
  strided split the *dynamic* branch collapses partway through epoch 0 — `scd` pinned at 3.4340 for
  ~300 consecutive steps while `scs` descends normally — and then recovers, 3.4127 (epoch 0 mean) →
  2.7584 → 2.27. The epoch line prints only `supcon_s`, so this is visible only in the step lines or
  `encoder_log.csv`'s `train_supcon_d`. Do not kill a run over it before epoch 2. `--overfit`
  now warms the LR up over its first 10% of steps, matching `train()`, and reaches 1.1185 against
  the 1.0986 floor. Bounding the usable region away from the quiet end was tried first and rejected:
  it cut collapses to 1/5 but pushed the train/val level gap back up to 3.3–5.5 dB, trading the fix
  for the original bug.
- **P×K batches are built into the dataset index, not a Sampler.** Item `i` decodes to
  `(episode, device slot, repeat)`, so a plain sequential `DataLoader` at `batch_size = P*K` yields P
  distinct captures × K segments. Deterministic under any worker count, and it keeps the repo's
  seed-and-index idiom (items are a pure function of `(seed, i)`).
- **SupCon's floor is `log(K-1)`, not 0** — 1.0986 at K=4, measured and pinned by a test. The
  `--overfit` sanity check descends 3.4252 → 1.1041 against it. Anyone reading that check against
  zero will conclude the model is broken when it is converged.
- **Augmentation is almost empty, on purpose.** Additive noise on the *wet* channel only (a
  recording artifact) and time masking. EQ, gain, pitch, time-stretch and reverb are absent and must
  stay absent: they **are** tone, so invariance to them is invariance to the target. Time jitter is
  free — the random window draw shifts both channels identically.
- **The dry-channel ablation is the load-bearing check.** In sweep-only mode every device is fed the
  *same* dry file, so the dry channel carries zero capture identity. An encoder can therefore ignore
  it, solve SupCon from the wet channel alone, post a perfect retrieval score, and have learned "what
  this recording sounds like" rather than "what this amp does to a signal". `encode-eval` re-runs
  retrieval and the dynamics probe with dry zeroed; the dynamics probe must degrade. Same role
  `val_esr_shuffled` plays for the emulators.
- **The dynamics probe regresses a measured crest-factor delta**, `(crest_wet − crest_dry)` in dB / 20,
  not a gain knob — this corpus has no numeric knob values, only free text in `model_name`. It is
  label-free, always available, and measures on this corpus at −13.6…+9.2 dB (median −2.8: amps
  compress). `gain_bucket` ships as a configurable alternative at weight 0, because it is a keyword
  heuristic with 29/404 `unknown`.
- **Encoder runs live in `results/encoder/`, not `results/emulate/`.** They log a contrastive loss,
  not ESR, and §6 above is what happens when metrics that share a name get cross-compared. Flip
  side: `publish_run_dashboard.py` globs `results/emulate/*/`, so encoder runs are invisible to the
  dashboard until someone adds a `--results-dir` pass-through.
- **`output_scale` is left applied** (`undo_output_scale: false`). Renders are peak-normalized per
  device, which perturbs absolute level — but 435/450 devices sit at exactly 1.0 (min 0.747), and
  amplitude *is* gain/compression information, so removing it costs more than it fixes. The scalar is
  in `renders.parquet` if the knob is ever flipped.
- **Checkpoint bug found and fixed during the build:** `last.pt` was written *before* the best-val
  update, so it recorded the previous epoch's worse number. A `--resume` then came back believing its
  best was that stale value and the next mediocre epoch overwrote a genuinely better
  `checkpoint.pt`. Observed live (reported best 3.4329 when the true best was 3.3480, then clobbered
  it with 3.4000). Two scripted-val tests pin it; note the obvious version of that test — just train
  and compare — passes on the buggy code, because the bug only shows when the final epoch of a
  session is an improvement.

**Status: three full runs, gated. The direction is alive.** All three are 404 devices at `8x4`
(the config's `16x4` OOMs the 12 GB card), ~50 epochs of 60 on patience-10, ~0.85 h each, and all
three are kept under `results/encoder/`.

| | tail | strided, whole file | **bounded pool + strided** |
|---|---|---|---|
| pool / split | `(0,0.85)` / tail 15% | whole file / strided 10 | `(0,0.85)` / strided 12 |
| best val SupCon | 1.3344 @ 39 | 1.4562 @ 35 | **1.2500 @ 40** |
| retrieval top-1 (46 unseen) | 0.973 | 0.948 | **0.989** |
| ... dry zeroed | 0.582 | 0.524 | **0.614** |
| branch redundancy mean \|r\| | 0.277 | 0.299 | **0.253** |
| amp id ← static (dynamic) | 0.894 (0.512) | 0.661 (0.324) | 0.891 (0.461) |
| crest R² ← dynamic (static) | **0.762** (0.728) | 0.607 (0.662) | 0.662 (0.738) |

**Only the first three rows compare cleanly.** Retrieval, the ablation and redundancy are computed
over the whole file regardless of split. The probes are fit on each run's *own* val regions, and val
SupCon is defined by them — the tail run's is measured 14.3 dB below its training level, and the two
strided runs use slightly different regions. Never rank runs on val SupCon alone; that is the mistake
this whole section is about.

What the gate says, on the best run:

- **Retrieval top-1 0.989 against a 2.2% chance line.** One 5.46 s segment identifies an amp the
  encoder has never seen, against a gallery of 46. Top-5 is 1.000.
- **Zeroing the dry channel drops it to 0.614.** The encoder is genuinely using the (dry, wet)
  relation, not memorizing what each recording sounds like — the one thing sweep-only training could
  not otherwise rule out.
- **Branch redundancy 0.253**, so the two-branch split is not decorative, and **amp id reads 0.891
  from static vs 0.461 from dynamic** — the intended specialization is real on that axis.
- **But neither dynamics target favours the dynamic branch.** Gain is `dynamic < static` in all
  three runs (0.615 vs 0.731 here) and crest is too in two of three (0.662 vs 0.738). The spec's
  premise is that dynamics live in the dynamic branch; measured, the static branch is at least as
  good at both. Gain is a weak test — `gain_probe_w` is 0.0, so that head is never trained — but
  crest *is* trained and still loses. This is the open question the architecture has to answer.

## 13. The joint model (new direction, 2026-08-13)

§12's encoder is standalone: it learns amp identity from a contrastive objective and nothing
consumes its output. The joint model asks the same question from the other end — **train an
encoder and a generator together, so the embedding is whatever makes reconstruction work.**
A new amp then costs one forward pass instead of a 30-epoch enrollment fit. Spec is
`src/openamp/joint_model/guidelines.md`; the package is `openamp.joint_model`, run with
`openamp joint` / `openamp joint-eval` into `results/joint/`.

Deliberately **not** the §12 `ToneEncoder`: this builds the guidelines' own WaveNet-tap
encoder (non-causal A2 schedule, no FiLM, no table, residual stream → attentive stats
pooling → MLP). Both halves train from scratch; there is **no contrastive term** for now.

- **The generator did not need a new architecture, only a new door.** `FiLMWaveNet.forward`
  looked its embedding up internally, so nothing could backprop into a vector. It is now
  `forward_emb(x, e)` with `forward(x, idx) = forward_emb(x, embedding(idx))`, routed through
  the existing `layer_conditioning` hook — so `delta_wavenet` and `tabledelta_wavenet` got a
  working vector path for free, `export.forward_with_embedding` became a 3-line no-grad
  wrapper instead of a parallel copy of the forward body, and **`TableDeltaWaveNet.forward`
  was deleted**: it was that same body with `layer_conditioning`'s slicing inlined. No new
  parameters, and every existing checkpoint still loads.
- **The A/B barrier is a different *file*, not a time gap.** The embedding is computed from a
  reference window B and the generator reconstructs a different window A; whatever survives
  that gap is amp, not content. `data_mixed` has 71 train / 10 val clean files, so B is simply
  drawn from another file — minutes of different music away, far harder than a within-file gap.
  The gapped same-file draw is the fallback for single-file (`sources: [sweep]`) corpora and is
  strictly weaker. Note the dry side is **shared across every device**, so a reference is only
  informative through its wet channel; the encoder cannot cheat by recognizing the music.
- **Two collapse modes, both found in the sanity ladder, both invisible in the loss.** With no
  contrastive term nothing supervises `e` directly, so a respectable ESR is not evidence of
  anything — the corpus mean is not a terrible model of any single amp. Hence `val_esr_shuffled`
  (embeddings permuted within the batch) and `emb_spread` (across-batch std of `e`) are logged
  **every epoch**, not just at the end.
  - **Unbounded `‖e‖`.** Inflating the embedding is a cheap early way to grow FiLM gain, and it
    ends by saturating the generator's tanh into an amp-independent solution. At `enc_lr` 4e-3
    a fixed batch diverged at ~step 500 with `‖e‖` = 123, spread 0.0000 and a shuffled ratio of
    **1.0** — true and wrong embeddings rendering *identically*. Init is not the problem (`‖e‖`
    starts at 3.7, against 1.6 for the table it replaces); the runaway is learned. Fixed by
    `enc_normalize: true` — guidelines §2's own toggle — which removes the direction rather
    than staying one learning rate away from it, at equal quality (0.0116 vs 0.0100 ESR).
  - **Encoder LR.** This is guidelines §9's "encoder/generator LR imbalance → dead encoder", and
    it is the difference between a model and nothing. Over 32 devices on real varying
    references, 600 steps, generator lr fixed at 4e-3 (train ESR / spread / shuffled ratio):
    `enc_lr 4e-3 → 0.976 / 0.0000 / 1.00x` (collapsed), `enc_lr 5e-4 → 0.285 / 0.0396 / 7.11x`.
    At 4e-3 the encoder collapses first and the generator follows it into the output-silence
    solution. A 200-step encoder warmup measured 0.279 / 0.0419 / 7.33x — no better than the
    plain lower LR, so it was not added. **The generator's own 4e-3 was fine throughout.**
- **A `--overfit` *pass* is not conclusive either.** §12 established that a single-batch
  collapse can be a false alarm; the joint model shows the converse and it is worse. The fixed
  batch fit happily (ESR 0.0116, shuffled ratio 38x) at the exact `enc_lr` that collapses to
  silence on real data within 100 steps. Memorizing 8 reference clips is a different problem
  from reading a reference never seen before, and only the second one is the task. **For the
  joint model the real gate is a short multi-device run watching `emb_spread`,** which costs
  ~10 minutes and would have caught both modes.
- **Sizes and the width that actually matters.** `embedding_dim: 256` is the generator's FiLM
  input, but pooling emits `2 * enc_channels` = **32** numbers, so that is the real bottleneck
  on what can be said about an amp — 256 is how wide it is written, not how much it says.
  Guidelines §7 puts `C_enc` at 16–32 and its own default `De` at 16, so 32 pooled is already
  generous; `enc_channels` is the knob to raise if amps underfit. Encoder 86,704 params against
  the generator's 236,353.
- **Batch 16, not 32.** Encoder plus generator peak at 8.7 GB of the card's 11.5 at batch 16
  (measured with MRSTFT in the loop); 32 OOMs. `enc_channels 16` without gradient checkpointing
  runs at ~35 smp/s — same memory as `enc_channels 32` *with* checkpointing but 35% faster.
  `pairs_per_epoch` is halved to 25,000 to keep a ~13 min epoch, and `plateau_patience` /
  `early_stop_patience` are raised to 5 / 15 so a half-size epoch does not give up on half
  the data.
- **The baseline is `nam_a2_tanh_ch16_512`** (val ESR 0.05188, shuffled 1.881 = 36x, 404/46
  devices, manifest `c97d68…`, 21.3 h). Same corpus, same arch and width of network, also from
  scratch — a learned table *is* guidelines Phase 0's "fixed code per amp", so that run is this
  one's control and the comparison is fair. The headline number to beat is not that one though:
  it is **held-out-device ESR conditioned on a correct reference**, which is the zero-shot claim
  and the thing enrollment currently costs 30 epochs to buy.

### 13.1 First results (`results/joint/proto`, epoch 12 of a run that needs 100+)

Undertrained on purpose — these are the *mechanism* numbers, not converged ones. Seen-amp ESR
was 0.439 against the table baseline's 0.0519, so nothing here is a quality claim yet.

- **The leakage test passes, which is the result the whole A/B design exists for.** Holding
  window A fixed and changing only the reference: a different segment of the **same** amp moves
  the output 0.078, a **different** amp's reference moves it 0.916 — a 11.7x separation. The
  embedding carries amp identity, not content. Drawing B from a different corpus file rather
  than a gapped same-file window is what bought that, and it was free.
- **The generator genuinely depends on `e`:** zeroing it costs 10.9x ESR, permuting it 3.4x.
- **The encoder generalizes to unseen amps; the generator does not (yet).** On the 46 held-out
  devices, k-NN retrieval is **top-1 0.978 / top-5 1.000 against 0.0217 chance** — the encoder
  separates amps it has never seen almost perfectly. But zero-shot ESR on those same amps is
  0.805 against 0.439 on trained amps. So the embedding *space* transfers and the generator's
  ability to render an arbitrary point in it does not. That is a generator capacity/training
  question, not an encoder one.
- **Reconstruction alone taught it amp identity.** That 0.978 retrieval is essentially §12's
  purpose-built contrastive `ToneEncoder` result (0.989) reached **with no contrastive term in
  the loss**. The strongest argument yet that the contrastive term is a stabilizer to reach for
  on collapse, not a requirement for the embedding to mean something — and a reason to fix the
  generator side before adding loss terms.

## 14. Open threads

- The device-energy weighting fix (per-device reference denominator) — proposed, not run.
- `NAM_SWEEP_FILENAME` root fix + region-sliced sweep ingestion in `corpus/build.py`.
- Duration-weighted (or otherwise re-balanced) file sampling, if the 1.4% sweep share turns
  out to matter.
- A fixed-window "audio budget" mode for enrollment (current `--pairs` is an optimization
  budget over fresh random windows).
- The corpus-vs-sweep comparison deserves a clean re-run without the A1/A2 confound
  (`data_sweep_a1/` exists precisely so a sweep-only corpus could be rebuilt A1-consistent).
- **The dynamic branch does not own dynamics.** Crest reads better off the *static* head (0.738 vs
  0.662) and so does gain (0.731 vs 0.615), in every run. Redundancy is low, so the branches have
  genuinely learned different things — the dynamic one just is not the one holding the dynamics.
  Worth trying before accepting it: raise `dyn_probe_w` above 0.2, and check whether the static
  branch's 155 ms receptive field is already enough for crest (it is a within-window statistic, so
  it may simply not need the 5.45 s view the dynamic branch was built for).
- **The encoder→emulator bridge is now the live question**, not a hypothetical. Retrieval at 0.989
  on unseen amps says the 512-d vector carries amp identity; whether it lands anywhere useful in a
  trained emulator's embedding space is untested and is the next real experiment.
- **`--overfit` at `8x4` still OOMs the 12 GB card at the config's own `16x4`.** The 2026-08-11 run
  hit this twice before someone lowered `devices_per_batch` to 8, which silently changed what its
  SupCon numbers mean. Either the config default should be 8, or the trainer should fail fast with a
  suggested batch size instead of a raw CUDA OOM traceback.
- **The encoder→emulator bridge**, if the gate passes: map the 512-d tone vector into a run's
  `[N, embedding_dim]` table so a new amp is enrolled by one forward pass instead of a 30-epoch fit.
  `export.forward_with_embedding` / `folded_model` already accept an arbitrary vector rather than a
  table row, so that is the seam. Deliberately not built yet.
- `tests/test_train.py` is dead: it imports `train.objectives` (`nt_xent_loss`, `retrieval_top1`)
  from the Phase-3 encoder module deleted in `ea6e83e`, so it has been a silent collection error for
  months. Now actively confusing next to the real `tests/test_encoder.py` — delete it.
- `publish_run_dashboard.py` only globs `results/emulate/*/` and charts `train_esr`/`val_esr`, so
  encoder runs never appear. A `--results-dir` pass-through plus a `SERIES` override would fix it.
