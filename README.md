# AllAmpsOnMe

A general-purpose guitar amp model trained on data captured from [TONE3000](https://www.tone3000.com).

- Follows the [Open-Amp paper](https://arxiv.org/abs/2411.14972): one FiLM-conditioned TCN emulates hundreds of amps at once.
- Each amp is a diverse **amp-only (DI)** [NAM](https://www.neuralampmodeler.com/) capture from the TONE3000 community — one "device" the model learns to emulate.
- Everything lives in one Python package and CLI (`openamp`), run as a sequence of verbs:

```
acquire :  auth · discover · select · download · validate · dedup · finalize · status
corpus  :  corpus · subset · render · verify
train   :  emulate · joint
evaluate:  emulate-compare · emulate-validate · emulate-demo · joint-eval
enroll  :  emulate-enroll   (+ `enroll.enroll_pair` for a raw capture pair, from a notebook)
export  :  emulate-export-bundle · emulate-export-profile · emulate-export-profiles
```

See [docs/architecture.md](docs/architecture.md) for the code layout and [docs/pipeline.md](docs/pipeline.md) for the full operator guide.

## Setup

```bash
conda create -n open-amp3000 python=3.10
conda activate open-amp3000
pip install -e ".[dev]"      # deps + pytest
cp .env.example .env         # add your TONE3000 key (OPENAMP_API_KEY)
```

- Requires `neural-amp-modeler>=0.13` — the first release that can parse an **A2** capture. Older versions fail validation on every A2 capture.
- Tunables: [`configs/openamp.yaml`](configs/openamp.yaml) (pipeline), [`configs/emulate/`](configs/emulate/) and [`configs/joint/`](configs/joint/) (one file per experiment).
- API access + data root: `.env` (`OPENAMP_*`).

## Pipeline

- `openamp --help` for the full command list, `openamp <command> --help` for any one.
- Every stage reads/writes the parquet manifests under `data/manifests/` and is idempotent — stop and resume anytime.

```bash
# 1 — Acquire diverse DI amp captures from TONE3000 into data/
openamp auth        # one-time browser OAuth
openamp discover    # build the candidate pool
openamp select      # diversity selection -> manifest
openamp download    # fetch the selected .nam files
openamp validate    # load + render-probe each capture
openamp dedup       # drop duplicate captures
openamp finalize    # assign device_ids -> final manifest

# 2 — Build the clean corpus and render it through each device (GPU)
openamp corpus                          # 48 kHz clean corpus + splits
openamp subset --size 450               # optional: gain-balanced device subset
openamp render --devices @data/manifests/render_subset.txt
openamp verify                          # completeness + QA -> devices_final.parquet

# 3 — Train one conditioned model over all devices. The config file *is* the
# experiment: it names the corpus, so no OPENAMP_DATA_DIR prefix to remember.
openamp emulate --config configs/emulate/nam_a2_256_mixed_dlw1.yaml
openamp emulate-compare results/emulate/*     # test-split comparison
openamp emulate-validate nam_a2_256_mixed_dlw1   # per-amp test ESR
openamp emulate-demo results/emulate/nam_a2_256_mixed_dlw1   # listening WAVs

# 4 — Fit an embedding for an amp the model never trained on.
openamp emulate-enroll results/emulate/nam_a2_256_mixed_dlw1

# 5 — (experiment) Replace the lookup with an encoder, trained jointly.
openamp joint --config configs/joint/fp_dac_attn512.yaml
openamp joint-eval fp_dac_attn512             # the collapse + leakage gate
```

## How the training code fits together

Everything above the corpus is a pipeline. Everything below it is **five seams**,
and each one is the same three things: a registry, a documented contract, and a
config section that names which implementation to use.

```
configs/emulate/<name>.yaml
        │
        ▼
   spec.load_run_spec ──────────► RunSpec        the boundary: below this line,
        │                          (frozen)      nothing reads a file or an env var
        ├── model:     ──► models.MODELS      ──► the network
        ├── data:      ──► data.DATASETS      ──► training pairs
        ├── loss:      ──► losses.LOSSES      ──► the objective
        ├── joint.encoder: ─► encoders.ENCODERS ─► the conditioning (joint runs)
        └── training:  ──► the loop's own knobs
        │
        ▼
   training.emulate.train(spec)  or  training.joint.train(spec)
        │
        ▼
   results/<stage>/<name>/       checkpoints · resolved_config.yaml · metrics.json
```

**One config file is one experiment.** It names its own corpus (`data_dir:`) and run
name, and the fully resolved spec is written back to `<run>/resolved_config.yaml`, so a
finished run records what it actually did rather than depending on what you typed
around it.

**A section's keys are the constructor's parameter names.** `model:` and `loss:` reach
the registered class untouched, which is why adding an architecture needs no schema
change anywhere.

**The loops name no implementation.** They call `build_model`, `build_dataset`,
`build_loss`, `build_encoder` and never mention an arch, a dataset or an objective by
name — so a new one costs no edit there.

### Adding an architecture

Everything an architecture-aware decision needs lives *on the architecture*: the log
and CSV labels (`describe`), and the plugin bundle's descriptor (`export_descriptor`).
The `--resume` check needs nothing, because the `model:` section **is** the structural
set by construction.

**1. Write the class** in a new file under `src/openamp/models/`:

```python
# src/openamp/models/my_arch.py
from torch import nn
from openamp.models.blocks import NormalizedEmbedding, make_activation

class MyWaveNet(NormalizedEmbedding, nn.Module):
    """Constructor parameters are the config's `model:` keys, verbatim."""

    arch = "my_arch"                       # its own registry name

    def __init__(self, *, n_devices, embedding_dim=64, channels=8, my_knob=3):
        super().__init__()
        ...                                # self.embedding, self.layers, ...

    @property
    def receptive_field(self) -> int: ...
    def forward_emb(self, x, emb): ...     # ([B,1,T], [B,E]) -> [B,1,T], causal
    def forward(self, x, device_idx):
        return self.forward_emb(x, self.embed(device_idx))

    def describe(self) -> dict:            # the comparison CSV's columns
        return {"arch": self.arch, "channels": self.channels,
                "blocks_x_layers": f"mine-{self.my_knob}"}

    def export_descriptor(self) -> dict | None:
        return None                        # not plugin-exportable (yet)
```

`embed`, `embedding_table` and `project_embedding` come from the `NormalizedEmbedding`
mixin. The full contract is the table at the top of
[`models/__init__.py`](src/openamp/models/__init__.py).

**2. Register it** — one line in [`models/__init__.py`](src/openamp/models/__init__.py):

```python
MODELS = {..., "my_arch": MyWaveNet}
```

(The built-ins are listed explicitly rather than self-registering, so a name is never
missing because its module happened not to be imported. `@register_model("my_arch")`
works too, for an arch defined outside the package.)

**3. Write a config** — `configs/emulate/my_arch_256.yaml`:

```yaml
data_dir: data_mixed

model:
  arch: my_arch
  channels: 8
  embedding_dim: 256
  my_knob: 5            # your constructor's own parameter, no schema change needed

data:     {dataset: emulation, clip_seconds: 2.0, pairs_per_epoch: 50000, holdout_frac: 0.1}
loss:     {name: esr_mrstft, preemph_coeff: 0.85, stft_weight: 0.05}
training: {lr: 4.0e-3, epochs: 400, batch_size: 32, amp: true, amp_dtype: bf16}
```

**4. Run it.** No training-loop change, and `emulate-compare` picks up the new rows
against every existing run:

```bash
openamp emulate --config configs/emulate/my_arch_256.yaml --overfit          # sanity #1
openamp emulate --config configs/emulate/my_arch_256.yaml --limit-devices 10 # sanity #2
openamp emulate --config configs/emulate/my_arch_256.yaml                    # the run
openamp emulate-compare results/emulate/*
```

A **loss**, a **dataset** or an **encoder** is the same four steps against
[`losses.py`](src/openamp/losses.py), [`data/__init__.py`](src/openamp/data/__init__.py)
or [`encoders/__init__.py`](src/openamp/encoders/__init__.py) — each carries its
contract in the same place, at the top of the module.

### Checking you changed nothing you didn't mean to

[`tests/test_equivalence.py`](tests/test_equivalence.py) pins the current behaviour as
sha256 snapshots: dataset items at fixed `(seed, index)` (the RNG-order guard), every
parameter after seeded construction, and a whole tiny run of both trainers including
the checkpoint weights. It runs on CPU/fp32/`num_workers=0`, which is what makes exact
equality the bar rather than a tolerance.

```bash
python -m pytest tests/test_equivalence.py
OPENAMP_UPDATE_SNAPSHOTS=1 python -m pytest tests/test_equivalence.py   # intended change
```

Regenerate only when a behaviour change is intended — the snapshot diff is the review.

## Data & licensing

- Downloaded `.nam` captures, OAuth tokens, and API keys are **never committed**.
- Captures are **never redistributed** — only per-capture metadata and license terms go in the manifest.
- The clean corpus and rendered audio are derived data, stay under the git-ignored `data/` tree, never published.
- Acquisition uses the official TONE3000 API only (no scraping), rate-limited at all times.

## Development

```bash
python -m pytest              # run the test suite
python -m openamp --help      # the CLI (also the `openamp` console script)
```

Some `test_validate.py` / `test_finalize.py` tests are skipped unless small `.nam` fixtures exist under `tests/fixtures/` — see [`tests/fixtures/README.md`](tests/fixtures/README.md).

## License

See [LICENSE](LICENSE).
