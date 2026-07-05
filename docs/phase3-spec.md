# Phase 3 Spec — Contrastive Guitar Effects Encoder

**Deliverable:** train the paper's contrastive audio-effects encoder on the Phase 2 rendered dataset, then evaluate it as a frozen feature extractor on external guitar-effect classification datasets, reproducing the paper's Table I/II experiments.

Standalone doc; consumes Phase 2 outputs (`devices_final.parquet`, `data/renders/`, `data/clean/`, `ContrastivePairDataset`). Same `openamp` conda env; this phase is deliberately specified **looser** than Phases 1–2 — training code is cheap to redo, so prefer simple over robust. Where this spec and the paper/reference code disagree on a hyperparameter, follow the paper (arXiv:2411.14972 Sec. III-A) / Alec Wright's OpenAmp repo and note the choice in the results report.

## 1. What we're building (context)

SimCLR-style self-supervised training: two different audio clips processed by the **same device** are a positive pair; clips from different devices in the batch are negatives. The encoder learns to represent "what amp/effect is this," independent of what's being played. Success is measured by how well simple probes (KNN, tiny MLP) on the **frozen** embeddings classify effects in datasets the encoder never saw.

## 2. Model

- 1-D convolutional encoder on raw audio (48 kHz mono, 2 s crops): ~6 residual conv blocks, kernel 5, channels growing 16→64, global temporal pooling → 64-d embedding; ~113K parameters. Match the paper/reference implementation where details differ.
- Projection head for training only (2-layer MLP, e.g. 64→64→64); probes and evals use the **pre-projection** embedding.
- Loss: NT-Xent (normalized temperature-scaled cross-entropy), temperature per paper/reference (default 0.1 if unspecified).

## 3. Training

- Data: `ContrastivePairDataset('train')` — two random 2 s crops per device, different positions/files. All ~400 devices.
- Batch 128 pairs, Adam (lr 1e-3 default, cosine or step decay per reference), ~200K iterations. Log NT-Xent loss + in-batch retrieval accuracy (top-1: does each view rank its pair first?).
- Checkpoint every N steps, keep best-by-val (val = in-batch retrieval on `ContrastivePairDataset('val')`).
- Single GPU, plain PyTorch training loop (no Lightning needed). Seeded. One config file `configs/phase3.yaml`.
- Sanity milestone before long training: after ~2K iterations, in-batch retrieval accuracy should be clearly above chance (1/128). If not, stop and debug (most likely a pairing bug or collapsed embeddings).

## 4. Evaluation (the actual point of this phase)

### 4.1 Datasets (external, download once into `data/raw/`)
1. **GFX / GUITAR-FX-DIST** — distortion-effect classification set built on IDMT clean recordings. Primary benchmark (paper Table I).
2. **EGFxSet (EGFX)** — Zenodo record 7044411; real hardware effects, 12 classes.
3. **EGDB rendered tones** — the amp-rendered folders of the EGDB download we already have (5 amp classes); the clean DI we trained on is fine as the "clean" class but note the overlap in the report.

Each gets a small adapter that yields (2 s crop @ 48 kHz mono, label); resample/normalize consistently with Phase 2 conventions.

### 4.2 Probes (frozen encoder)
- Embed every clip (mean over crops if clip > 2 s).
- **KNN probe** (k=5, cosine distance) and **MLP probe** (single hidden layer, e.g. 128 units, trained on embeddings only).
- Baselines: identical probes on a **randomly initialized** encoder (paper's control), and optionally on simple spectral features (MFCC mean/std) for context.
- Standard train/test splits per dataset (or seeded 80/20 by underlying clean source where none defined — avoid leaking the same clean phrase into both sides).

### 4.3 Report
`results/phase3_report.md`: training curves, probe accuracy tables vs. the paper's Tables I & II, confusion matrices for the main GFX result, brief notes on deltas (expected: our EGDB/NAM training data vs. their IDMT/Proteus). Also export a 2-D UMAP/t-SNE of device embeddings colored by gain_bucket — a nice qualitative check that the space organizes by tone character.

## 5. Deliverables & done-when

- `src/models/encoder.py`, `src/train/contrastive.py`, `src/eval/probes.py`, dataset adapters, config.
- Best checkpoint + embedding export script (`embed.py audio_in.wav → 64-d vector`), reused in later phases.
- Done when: training converged (retrieval accuracy plateaued), trained encoder beats the random-init baseline by a wide margin on GFX with both probes, and the report exists. Exact paper-number matching is not required — direction and rough magnitude are.

## 6. Out of scope

Foundation model training (Phase 4), enrollment (Phase 5), hyperparameter searches beyond small lr/temperature nudges, data re-rendering.