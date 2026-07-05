# Phase 3 — Contrastive guitar-effects encoder

Phase 3 trains the paper's SimCLR-style **audio-effects encoder** on the Phase 2
rendered dataset, then evaluates it as a **frozen** feature extractor with simple
probes (KNN, MLP) on external guitar-effect classification datasets, reproducing
the direction and rough magnitude of Open-Amp's Table I/II results
(arXiv:2411.14972).

Full spec: [docs/phase3-spec.md](docs/phase3-spec.md). Consumes Phase 2 outputs
(`data/renders/`, `data/manifests/renders.parquet`, `ContrastivePairDataset`);
see [OA3K.md](OA3K.md) for Phase 2 and the [README](README.md) for the whole
project. Same `open-amp3000` conda env.

```
python -m train.contrastive           # train the encoder (SimCLR / NT-Xent)
python -m eval.embed  audio.wav        # audio -> 64-d effect embedding
python -m eval.probes egfxset          # frozen-probe one dataset
python -m eval.report                  # full eval + figures -> results/phase3_report.md
```

---

## What it does

- **Model** (`src/models/encoder.py`): a ~109 K-param 1-D residual conv encoder
  over raw 48 kHz audio (2 s crops) → 64-d embedding, plus a train-only MLP
  projection head. Two crops of the **same device** are a positive pair; other
  devices in the batch are negatives.
- **Training** (`src/train/contrastive.py`): NT-Xent loss, Adam, cosine schedule,
  AMP, single GPU. Best checkpoint chosen by in-batch retrieval on the `val`
  split. All randomness derives from one `seed`.
- **Eval** (`src/eval/`): embed every clip (mean over crops), then KNN (k=5,
  cosine) and MLP probes on the frozen **pre-projection** embedding, against a
  **random-init** encoder control and an **MFCC** baseline. Leak-free grouped
  splits. Report with training curves, probe tables vs. the paper, a confusion
  matrix, and a UMAP of device embeddings colored by gain bucket.

---

## Setup

```bash
pip install -e ".[phase3]"        # or: pip install -r requirements-phase3.txt
```

Reuses the Phase 2 torch/soundfile stack and adds scikit-learn, umap-learn,
librosa, matplotlib. Training needs a CUDA GPU; eval runs on CPU or GPU.

---

## 1. Train

```bash
python -m train.contrastive                     # full run (configs/phase3.yaml)
python -m train.contrastive --iterations 2000   # sanity milestone
python -m train.contrastive --resume            # continue from last.pt
```

All tunables live in [configs/phase3.yaml](configs/phase3.yaml) (model, training,
eval). Data paths come from the Phase 2 config so both phases agree on where the
renders live.

**Crop pool (performance).** The renders live on NFS; random 2 s FLAC seek-reads
cap a training loop at ~100 samples/s while the GPU can do ~3000/s. On first run,
`src/train/cropcache.py` decodes a fixed pool of crops per device **once** into a
local int16 memmap (`/tmp/oa3k_phase3_cache_<user>/`), then trains from RAM
(~25× faster; GPU-bound). Set `train.use_crop_pool: false` to read renders
directly. Delete the cache dir to rebuild.

Outputs land under `results/phase3/`: `checkpoints/best.pt` + `last.pt` (weights
**and** model config, so the encoder reconstructs without the yaml),
`train_log.jsonl` (curves), `train_stdout.log`.

The sanity milestone (spec §3): by ~2 K iterations, train in-batch retrieval must
clearly beat chance (1 / 2·batch). If not, suspect a pairing bug or collapse.

---

## 2. Evaluation datasets (external, one-time)

Downloaded into `data/raw/` (git-ignored). Each is optional — the eval skips any
that are absent.

| Dataset | Get it | Adapter |
|---|---|---|
| **EGFxSet** (12 real-hardware effects) | `python scripts/download_egfxset.py` | `eval.adapters.egfxset_adapter` |
| **GFX / GUITAR-FX-DIST** (13 distortion effects, paper Table I) | `python scripts/download_gfx.py` (~27 GB split zip) | `eval.adapters.gfx_adapter` |
| **EGDB amp renders** (5 amp classes) | download EGDB amp folders to `data/raw/egdb_rendered/` | `eval.adapters.egdb_amp_adapter` |

An **in-domain** amp-classification set is also derived from our own renders
(`internal_amp_adapter`) for the UMAP / sanity check — reported separately since
the encoder trained on those devices.

---

## 3. Embed / probe / report

```bash
python -m eval.embed  path/to/audio.wav -o vec.npy      # 64-d embedding
python -m eval.probes egfxset --no-mfcc                 # quick probe on one set
python -m eval.report --datasets egfxset gfx internal   # full report + figures
```

`eval.report` runs the probe suite on every available dataset, builds figures
under `results/phase3/figures/`, and writes `results/phase3_report.md` (the only
tracked output — everything else is derived from license-restricted audio).

---

## Testing

Core logic is unit-tested without the GPU stack (torch/sklearn tests are
`importorskip`-guarded):

```bash
pytest tests/test_encoder.py tests/test_objectives.py tests/test_probes.py \
       tests/test_eval_adapters.py tests/test_cropcache.py tests/test_phase3_config.py -q
```

Key proofs: NT-Xent retrieval is 1.0 for perfectly-aligned pairs and near chance
for random; the encoder hits the ~113 K param budget; probes reach >90 % on
separable synthetic embeddings; the crop-pool dataset yields two distinct crops
per device from the memmap.
