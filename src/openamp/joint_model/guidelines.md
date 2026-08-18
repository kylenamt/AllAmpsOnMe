# Conditional NAM — Encoder & Joint-Training Plan

*Assumes you **already have** a FiLM NAM generator that accepts an input embedding. This plan covers only the new parts: the **encoder**, the **embedding interface** to your generator, the **joint training**, and the **data**. Generator internals are intentionally left out.*

---

## 0. Goal and end state

**What you're adding.** An encoder that turns a short *reference* clip of an amp into a fixed-size embedding `e`, which you feed to your existing FiLM generator so it renders that amp's sound. Encoder and generator are trained jointly.

**The core trick.** `e` is computed from a *different* segment of the recording than the one the generator reconstructs, so it can only carry what's shared across segments — the amp/tone identity — not the content.

**Definition of done (research):**
- Held-out ESR conditioned on a *correct* reference approaches a per-amp NAM trained on that amp alone.
- Ablations confirm the model *uses* `e` (wrong/zero `e` degrades output) but does *not* leak content (swapping to a different reference segment of the same amp barely changes output).
- Embedding space clusters by amp.

---

## 1. System overview

```
recording (one amp)
   ├── Segment B (reference, disjoint from A) ──► Encoder ──► e
   └── Segment A (dry) ──► [your FiLM generator]  ◄── e ──► ŷ  ──► loss vs Segment A (wet)
```

Reconstruction gradient flows from the generator's output, through its FiLM path, back into the encoder. An auxiliary contrastive loss shapes `e` directly.

---

## 2. Embedding interface contract

The only coupling point between the new encoder and your existing generator. Nail this first.

- The encoder emits `e ∈ R^De` per example (one vector per clip).
- **Match `De`** to whatever embedding width your generator's FiLM input expects.
- **Match normalization** to your generator's expectation (e.g. L2-normalized to the unit sphere, or raw). The encoder has a toggle for this.
- **Time handling is your generator's job.** Your generator already broadcasts/consumes the embedding internally (constant across time for a fixed reference). Consequence to be aware of: a constant `e` makes its FiLM a *time-invariant* per-channel affine (device identity); nothing to build here, just expect that behavior.
- **Gradient must flow** from the generator back into `e` (don't `detach` the embedding on the conditioning path), or the encoder never learns.

Everything below produces and trains `e` against this contract.

---

## 3. Encoder spec

- Backbone: a WaveNet feature extractor returning the `(B, C_enc, L)` residual stream (no head collapse). See `wavenet_encoder.py`.
- Input: reference **(dry, wet) pair**, 2 channels (recommended over wet-only — lets it observe the transfer function directly). Set the encoder WaveNet `input_size = 2`; its own `condition_size` can stay 1.
- Causality: **non-causal** is fine and usually better (the reference is offline) — use centered convs / the `ConvNet` variant.
- Pooling: attentive stats pooling `(B, C_enc, L) → (B, 2·C_enc)`, with masking for batched variable-length references (masked mean/std; trim the mask to the post-conv feature length, right-aligned).
- Projection: MLP `2·C_enc → De`, optional L2-normalize (per the interface contract).
- Feature tap: run the layer arrays and return the residual stream instead of applying the head; or expose the pre-`head_rechannel` summed skip for a multi-layer aggregate.

---

## 4. Data pipeline

### 4.1 Per-amp records
Each training amp/recording provides a time-aligned **(dry, wet)** pair at 48 kHz. Reuse NAM's latency calibration to align dry↔wet first — misalignment poisons both the target and the encoder pair.

### 4.2 Sampling one training example
From a single amp's recording draw:
- **Window A:** dry `x_A` (generator input) + wet `y_A` (target), length `L_train` (e.g. 8k–32k samples).
- **Reference window B:** a **disjoint** segment (dry_B, wet_B), length `L_ref` (2–4 s — long enough that pooling averages out content). Draw B **far** from A (enforce a minimum gap; ideally a different take/section).

### 4.3 Batching
- Every example's A and B come from the **same** amp.
- A batch mixes **many different amps**.
- For the contrastive loss, include ≥2 examples per amp per batch (two different B segments of the same amp = a positive pair).

### 4.4 Augmentation
Apply identity-preserving, content-altering augmentation to the reference (pitch/time perturbation, level jitter) so `e` can't latch onto content. Leave the amp coloration untouched.

---

## 5. Training objective

Per example: `ŷ_A = Generator(x_A, e_B)`, where `e_B = Encoder(dry_B, wet_B)`.

**Reconstruction** (reuse NAM's losses):
- `L_ESR = Σ(y_A − ŷ_A)² / Σ y_A²`
- `L_MRSTFT` (auraloss), small weight
- optional pre-emphasis on ESR/MSE, optional DC loss

**Contrastive / consistency on embeddings** (add): pull together embeddings from two segments of the same amp, push apart different amps (InfoNCE or triplet over the batch's `e`s). Directly supervises "`e` = amp identity" and stabilizes joint training.

**Total:**
```
L = L_ESR + λ_stft · L_MRSTFT + λ_contrastive · L_contrastive
```
Start `λ_stft = 5e-4`, `λ_contrastive ≈ 0.1`, then tune.

---

## 6. Implementation phases

Each phase has a check. Don't advance until it passes.

**Phase 0 — Generator embedding sanity.** With your *existing* generator, feed a **fixed random code per amp** (no encoder yet). *Check:* it fits several amps when given distinct fixed codes, and degrades on a wrong/zero code. Confirms the generator's embedding pathway works and gradients reach the embedding input.

**Phase 1 — Encoder shapes.** Implement encoder + pooling; run on dummy tensors. *Check:* `(B, 2, L_ref) → (B, De)` for several `L_ref`; masking works; `De`/normalization match the contract (§2).

**Phase 2 — Handoff, single amp.** Replace the fixed code with the encoder's `e`; reconstruction loss only; train on **one** amp. *Check:* ESR drops to baseline; end-to-end training is stable; gradient reaches the encoder.

**Phase 3 — Multi-amp + disjoint sampling.** Add A/B sampling and multi-amp batches. *Check:* one model fits several amps; the *correct* reference beats a *wrong* one.

**Phase 4 — Contrastive loss.** Add the embedding loss. *Check:* embedding space clusters by amp (UMAP); more stable training; leakage diagnostics improve.

**Phase 5 — Ablations + sweeps.** Run diagnostics (§8); sweep `De`. *Check:* no collapse in either direction; pick `De`.

**Phase 6 — Inference.** Freeze `e` per amp; wire the offline-reference → stream path. *Check:* frozen-`e` inference matches training-time conditioned output.

---

## 7. Hyperparameters (starting points)

*Generator config = your existing FiLM NAM; not listed here.*

| Thing | Start | Notes |
|---|---|---|
| Encoder channels `C_enc` | 16–32 | can be richer than the generator |
| Encoder causality | non-causal | offline reference |
| Embedding dim `De` | **16** (sweep 8/16/32/64) | the master bottleneck knob; must match generator input |
| Embedding normalization | match generator | L2 or raw |
| Pooling | attentive mean+std | fall back to masked mean if unstable |
| `L_train` | 8k–32k samples | |
| `L_ref` | 2–4 s | long enough to average content |
| A–B gap | large | avoid temporal correlation |
| `λ_stft` | 5e-4 | |
| `λ_contrastive` | ~0.1 | tune |
| Optimizer | Adam, lr 4e-3, wd 3.17e-7 | consider a lower lr for the encoder |
| LR schedule | ExponentialLR γ=0.994 | |
| Batch | many amps, ≥2 segments/amp | needed for contrastive |

---

## 8. Evaluation and collapse diagnostics

- **Uses `e`?** Feed **zero** `e` and **shuffled** `e` (swap across the batch). ESR should worsen clearly. If not → collapse mode A (generator ignores `e`; bottleneck too tight or `e` uninformative).
- **Leaks content?** Swap the reference to a *different segment of the same amp*. Output should barely change. Then swap to a *different amp*'s reference — output should change a lot. If a same-amp swap changes output → leakage; tighten `De` or increase A–B gap/augmentation (collapse mode B).
- **Held-out ESR** conditioned on correct reference, vs per-amp NAM baseline.
- **Embedding geometry:** UMAP/t-SNE of `e` colored by amp → clean clusters = good.
- **Cross-amp transfer:** condition amp X's dry with amp Y's reference; should sound like Y.

---

## 9. Risks and mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| Bottleneck too large | content leaks; same-amp-segment swap changes output | shrink `De`; add augmentation |
| Bottleneck too small | "average amp", ignores `e` | grow `De`; check contrastive weight; verify generator FiLM capacity |
| Segments too correlated | subtle content leak | larger A–B gap, different takes, augment |
| Encoder/generator LR imbalance | dead encoder / one branch dominates | separate LRs; warm up encoder |
| Embedding detached on the conditioning path | encoder never learns | ensure gradient flows into `e` |
| Loss weighting off | tone right but noisy, or clean but wrong EQ | tune `λ_stft`; add pre-emphasis |
| Latency misalignment | poor ESR floor everywhere | re-run NAM latency calibration |
| Batch not amp-grouped | contrastive meaningless | enforce ≥2 same-amp examples/batch |

---

## 10. Inference

Compute `e` once from a reference (offline), then run your generator with `e` frozen. Because `e` is constant, its FiLM path is a fixed per-channel affine — no per-sample overhead beyond the embedding your generator already ingests. (If you later want a static per-amp `.nam` for the stock C++ engine, that's a generator-side export concern — folding the constant-`e` FiLM into the weights — outside this plan.)

---

## 11. Optional extensions

- **VQ bottleneck:** quantize `e` to a codebook for a discrete amp vocabulary and cleaner disentanglement.
- **Multi-reference averaging:** average `e` over several reference segments for a more stable fingerprint at inference.
- **Adversarial content scrubbing:** a content classifier on `e` trained adversarially to remove residual content info.

---

## Appendix — file/class reference map (new/shared parts only)

| Concern | Where |
|---|---|
| Encoder module | `wavenet_encoder.py` |
| Feature tap (residual stream / pre-rechannel skip) | `nam/models/wavenet/_layer_array.py` (`LayerArray`) |
| Losses (ESR, MRSTFT, pre-emphasis, DC) | `nam/train/lightning_module.py` |
| Dataset, latency alignment, 48 kHz | `nam/data.py` (`Dataset`) |
| Generator | your existing FiLM NAM (external to this plan) |