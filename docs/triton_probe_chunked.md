# Running probe_chunked on Triton

This is the checklist for running `openamp.probes.probe_chunked` (the chunked
complex-STFT Conformer encoder, contrastive phase) on Triton. Use the checkout and
Python environment that ran c1 (`probe_conformer`). This probe needs nothing new
from either of them.

## 1. Code

```bash
git fetch && git checkout joint_models && git pull
```

You need commit `10174e5` or later:

- `src/openamp/probes/probe_chunked.py`: the probe.
- `scripts/triton_probe_chunked.sbatch`: the job script.
- `scripts/triton_stage_chunked.sh`: copies the data across.

The GitHub repo moved to `git@github.com:kylenamt/AllAmpsOnMe.git`. If `git pull`
complains, update the remote.

## 2. Data

c1 read every input this probe reads, so they should already be on Triton. All of
them are git-ignored: a pull does not bring them, and a missing one fails at
startup.

| Path (relative to the repo root) | Size | What it is |
|---|---:|---|
| `data_sweep_a1/clean/train/sweep_train.flac` | 12 MB | the reference: the T3K capture signal's train region, 171 s |
| `data/captures/` | 2.1 GB | the `.nam` files |
| `data/manifests/renders.parquet` | 6 MB | each amp's `nam_sha256` and `output_scale` |
| `results/emulate/delta_a2_tanh_r32_512_v2_n10/` | 28 MB | the frozen generator, its `embedding.pt`, and its config |

If any of them is missing, run `scripts/triton_stage_chunked.sh` on the local box. It
reads `TRITON=user@host` and `DEST=<repo root on Triton>` from the environment.

The probe finds each amp's `.nam` by hashing the files in `data/captures/` and
matching `nam_sha256`. Never match by `file_path`: the `data_sweep_a1` manifest
points 366 amps at the wrong files.

## 3. GPU and job settings

- **A100, H100 or H200 only.** torch 2.9.1 (cu128) has no Volta kernels, so a V100
  fails on its first kernel launch.
- **Partitions and environment** are the ones c1 ran with (`conformer_c1.sbatch`):
  the same four GPU partitions, `module load mamba` + `source activate
  $WRKDIR/envs/openamp`, `OPENAMP_DATA_DIR=data` and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **GPU memory.**
  - A 32 + 32 step at full length is about 12 GB.
  - Gradient checkpointing (`--ckpt-group 112`) is what keeps it that small. It
    changes memory only, never the result.
  - Without it, the same step needs about 90 GB.
- **CPU memory.** About 10 GB holds the val audio (real and made-up). The job asks
  for 48 GB.
- **Time.** About 2–3 h for 1,000 steps on an A100 or H100. The job asks for 6 h.
- **Paths.** `PY` (the python to run) and `OUT` (the run folder) can be set when you
  submit. The defaults are the activated env's `python` and
  `results/encoder/encoder_probe_chunked_p3`.

## 4. Launch

From the repo root:

```bash
# optional 2-step smoke test (about 10 min, most of it setup)
OUT=results/encoder/smoke sbatch --time=00:30:00 scripts/triton_probe_chunked.sbatch --steps 2 --eval-every 1

# the run
sbatch scripts/triton_probe_chunked.sbatch
```

- **Extra arguments go to the probe.** When a flag is repeated, the last one wins.
- **Submitting the same command again resumes from `$OUT/last.pt`.** It continues
  the exact same random stream and the same learning-rate curve.
- **Resume refuses changed settings.** It stops if `--steps`, any model setting, or
  any data setting differs from the original run (the list is `RESUME_FIXED`). To
  change one of them, start a new `OUT`.
- **The log is appended, never overwritten.** It is
  `$OUT/train_stdout.log`. Never redirect it with `>`.

## 5. The settings this run is meant to have

p3 differs from p2 (local, `results/encoder/encoder_probe_chunked_p2`) **only** by
adding the made-up amps. Do not change anything else without asking.

| Setting | Value | Note |
|---|---|---|
| `--batch` / `--synth-per-batch` | 32 / 32 | 32 real + 32 made-up amps per step |
| `--synth-shape` | 0.5 | Made-up draws spread more evenly than the real amps. Scaling the Gaussian up does nothing, because each draw is rescaled to the table length. |
| `--synth-val` | 76 | Fixed made-up val amps. They are reported, but never used to pick the checkpoint. |
| `--steps` | 1,000 | same as p2 |
| `--lr` / `--warmup-steps` / `--min-lr-frac` | 3e-4 / 50 / 0.01 | **Must keep.** A fixed 1e-3 with no warmup collapsed p1. |
| `--tau` | 0.1 | |
| `--seed` | 1234 | Same train/val split (684 / 76) and the same length draws as p2. |
| model | 2 Conformer blocks, width 512, 8 heads, 8 pool heads, 15,320,960 params | |

## 6. Reading the log

There is one line every 21 steps, roughly one pass over the training amps:

```
s00168  train loss … top1 …  val loss … top1 A>B … B>A …  made-up val loss … top1 …  best …  lr …  * = new best
```

- **Chance values.** Top-1 at chance is 1/76 = 0.013. A loss at chance is
  ln 76 = 4.3307 on val and ln N on train, with N = 64 amps per step here
  (ln 64 = 4.1589).
- **Collapse.** Val loss stuck at exactly 4.3307 means every amp has the same
  embedding. That was p1's failure, and it did not recover. Stop the run and report
  it.
- **Normal noise.** Train loss jumps in single windows are normal. Each step draws
  its own length, 1 to 56 chunks, and a window with many short crops scores worse.
  Recompute a window's lengths before blaming instability: each step's length is
  `step_rng(1234, 3, step).integers(1, 57)`.
- **The real val top-1 tops out.** p2 reached 100% / 98.7% by step 168. Compare
  runs by **val loss** (p2's best: 0.380 at step 798) and by short-crop retrieval,
  not by top-1 at 84.4 s.
- **What `render` means.** The `render` percentage is the share of time spent
  playing the crops through the `.nam` files and the generator. About 70% is
  expected.

## 7. Afterwards

`$OUT/probe_chunked.pt` holds the best weights (`best_state`), the val amp ids and
the report. `$OUT/report.json` holds the summary.

Retrieval by crop length (2 s up to 84.4 s) is a scratch script on the local box.
To compare with p2, copy the run folder back and run it there.
