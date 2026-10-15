"""The resolved experiment: one config file in, one :class:`RunSpec` out.

This is the boundary. Everything upstream of it reads YAML and the environment;
everything downstream takes a spec and never looks at a file, an env var, or a
global again. A spec is frozen, is written verbatim to
``<run>/resolved_config.yaml`` and into the checkpoint, and is enough on its own
to rebuild the run -- including which corpus it trained on, which the old
``emulate:`` config never recorded because ``OPENAMP_DATA_DIR`` supplied it from
the environment.

**Six sections, mirroring the things that get changed independently:**

=============  ===================================================================
``model``      arch name + whatever that arch's constructor takes
``data``       which dataset, how many pairs, how long a clip
``loss``       which objective + its knobs, and the per-device weighting
``training``   optimizer, schedule, precision, logging -- nothing model-specific
``joint``      encoder + reference window; absent for a table-conditioned run
``unfold``     the frozen generator/encoder runs + the unfolder's knobs; absent
               otherwise (a run is emulate, joint, or unfold -- never two)
=============  ===================================================================

``model.kwargs`` and ``loss.kwargs`` are passed straight to the registered class,
so an architecture or objective with new knobs needs **no change here at all** --
that is the point of keeping them opaque. The typed fields are only the ones the
training loop itself reads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from openamp.core import constants as C
from openamp.core.config import DEFAULT_SEED, ConfigError, _read_yaml

__all__ = ["ModelSpec", "DataSpec", "LossSpec", "TrainingSpec", "EncoderSpec",
           "RefSpec", "JointSpec", "UnfoldSpec", "RunSpec", "load_run_spec",
           "spec_from_checkpoint"]


def _split(values: dict, *named: str) -> tuple[dict, dict]:
    """``(the named keys present, everything else)`` -- typed fields vs kwargs."""
    known = {k: values[k] for k in named if k in values}
    return known, {k: v for k, v in values.items() if k not in named}


@dataclass(frozen=True)
class ModelSpec:
    """Which architecture, and its constructor's own arguments.

    ``kwargs`` reaches ``MODELS[arch](n_devices=..., **kwargs)`` untouched, so
    the yaml uses the constructor's parameter names (``channels``, not
    ``wn_channels``) and a new arch's knobs need no schema change.
    """

    arch: str = "film_wavenet"
    kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        named, kwargs = _split(dict(d), "arch")
        return cls(arch=str(named.get("arch", "film_wavenet")), kwargs=kwargs)

    def to_dict(self) -> dict:
        return {"arch": self.arch, **self.kwargs}


@dataclass(frozen=True)
class DataSpec:
    """Which dataset to draw training pairs from, and how many.

    ``clip_seconds`` must match the corpus's clip grid. ``holdout_frac`` is the
    device holdout, excluded from training and enrolled later.
    """

    dataset: str = "emulation"
    clip_seconds: float = 2.0
    pairs_per_epoch: int = 50_000
    holdout_frac: float = 0.1
    sources: tuple = ()
    kwargs: dict = field(default_factory=dict)

    _NAMED = ("dataset", "clip_seconds", "pairs_per_epoch", "holdout_frac", "sources")

    @classmethod
    def from_dict(cls, d: dict) -> "DataSpec":
        named, kwargs = _split(dict(d), *cls._NAMED)
        if "sources" in named:
            named["sources"] = tuple(named["sources"] or ())
        return cls(**named, kwargs=kwargs)

    def to_dict(self) -> dict:
        return {"dataset": self.dataset, "clip_seconds": self.clip_seconds,
                "pairs_per_epoch": self.pairs_per_epoch,
                "holdout_frac": self.holdout_frac, "sources": list(self.sources),
                **self.kwargs}


@dataclass(frozen=True)
class LossSpec:
    """Which objective, and its constructor's own arguments.

    ``device_loss_weighting`` is a named field rather than a kwarg because the
    *trainer* applies it -- it scales each item's contribution before the loss
    sees it, offsetting the corpus's ~37 dB output-level spread in the pooled
    ESR term. 0.0 = off, 1.0 = every device weighted equally. It applies to the
    train loss and to val ESR alike, so a non-zero value is not comparable with
    a strength-0 run's val ESR.
    """

    name: str = "esr_mrstft"
    device_loss_weighting: float = 0.0
    kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "LossSpec":
        named, kwargs = _split(dict(d), "name", "device_loss_weighting")
        return cls(**named, kwargs=kwargs)

    def with_kwargs(self, **changes) -> "LossSpec":
        """A copy with objective kwargs overridden -- how ``--stft-weight`` lands."""
        return replace(self, kwargs={**self.kwargs, **changes})

    def to_dict(self) -> dict:
        return {"name": self.name,
                "device_loss_weighting": self.device_loss_weighting, **self.kwargs}


@dataclass(frozen=True)
class TrainingSpec:
    """Optimizer, schedule, precision and logging. Nothing model-specific."""

    lr: float = 5e-4
    weight_decay: float = 0.0
    scheduler: str = "plateau"         # "plateau" | "cosine"
    plateau_patience: int = 3          # ReduceLROnPlateau patience, in epochs
    plateau_factor: float = 0.5
    # cosine only: the floor each group anneals to, as a fraction of its own lr
    cosine_min_lr_frac: float = 0.01
    early_stop_patience: int = 8       # stop after N epochs with no val-ESR gain
    epochs: int = 60                   # max; early stopping usually ends it sooner
    batch_size: int = 16
    num_workers: int = 6
    amp: bool = True                   # CUDA mixed precision (the STFT term stays fp32)
    amp_dtype: str = "fp16"            # "fp16" | "bf16"
    val_pairs: int = 2000              # pairs behind each val-ESR estimate
    log_every: int = 50
    # Train-time Gaussian jitter on the conditioning, as a fraction of the
    # embedding's own scale (0 = off; train steps only, never val/eval).
    embed_noise_frac: float = 0.0
    # Train-time nested dropout on the conditioning: each sample keeps only its
    # first k embedding dims, k drawn uniformly from these sizes, so the leading
    # dims learn the coarse amp and later ones refine it (empty = off; train
    # steps only). A sample keeps every dim with probability embed_prefix_full_frac.
    embed_prefix_sizes: tuple[int, ...] = ()
    embed_prefix_full_frac: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingSpec":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ConfigError(f"training: unknown keys {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


@dataclass(frozen=True)
class EncoderSpec:
    """Which conditioning encoder, and its own arguments.

    Like :class:`ModelSpec`, ``kwargs`` is opaque -- but an encoder's factory
    does real work with it (loading an npz, opening a frame store), so it goes
    through ``from_spec`` rather than straight to the constructor.
    """

    kind: str = "wavenet"
    kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "EncoderSpec":
        named, kwargs = _split(dict(d), "kind")
        return cls(kind=str(named.get("kind", "wavenet")), kwargs=kwargs)

    def to_dict(self) -> dict:
        return {"kind": self.kind, **self.kwargs}


@dataclass(frozen=True)
class RefSpec:
    """Window B: the reference segment the encoder reads, disjoint from window A.

    ``different_file`` is what makes the embedding carry *tone* rather than
    *audio* -- only what survives a change of source material is the amp.
    ``min_gap_seconds`` is the weaker same-file fallback.
    """

    seconds: float = 2.0
    different_file: bool = True
    min_gap_seconds: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "RefSpec":
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"joint.ref: unknown keys {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


@dataclass(frozen=True)
class JointSpec:
    """The encoder arm. Present only for a jointly-trained run.

    ``enc_lr`` is separate from ``training.lr`` deliberately: an encoder trained
    at the generator's rate collapses to silence, so the split is load-bearing,
    not a convenience.
    """

    encoder: EncoderSpec = field(default_factory=EncoderSpec)
    ref: RefSpec = field(default_factory=RefSpec)
    enc_lr: float = 4.0e-3
    enc_weight_decay: float = -1.0     # < 0 inherits training.weight_decay
    grad_checkpoint_encoder: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "JointSpec":
        d = dict(d)
        named, rest = _split(d, "enc_lr", "enc_weight_decay", "grad_checkpoint_encoder")
        unknown = set(rest) - {"encoder", "ref"}
        if unknown:
            raise ConfigError(f"joint: unknown keys {sorted(unknown)}")
        return cls(encoder=EncoderSpec.from_dict(rest.get("encoder", {})),
                   ref=RefSpec.from_dict(rest.get("ref", {})), **named)

    def to_dict(self) -> dict:
        return {"encoder": self.encoder.to_dict(), "ref": self.ref.to_dict(),
                "enc_lr": self.enc_lr, "enc_weight_decay": self.enc_weight_decay,
                "grad_checkpoint_encoder": self.grad_checkpoint_encoder}


@dataclass(frozen=True)
class UnfoldSpec:
    """Deep-unfolded enrollment (:mod:`openamp.enroll.unfold`). Present only for an unfold run.

    Both networks are *finished runs*, frozen, named by directory: the generator
    an emulate run (``delta_wavenet``), the first guess a joint run's encoder.
    The unfold run's ``model``/``data``/``loss`` sections are not read from its
    config -- the trainer copies them from ``generator_run``'s checkpoint, since
    that generator *is* the model, and records them in ``resolved_config.yaml``.

    ``training.lr`` drives ``V``; ``step_lr`` drives the log step sizes, which
    live on a different scale (a log, not a weight).
    """

    generator_run: str = ""
    encoder_run: str = ""
    frames_path: str = ""        # "" = the store the encoder's checkpoint recorded
    steps: int = 5               # K
    clips: int = 16              # per amp: fit clips, and as many again to score on
    # Where the training amps' fit and loss clips, and the held-out amps' fit
    # clips, come from: "" = the run's own corpus. Selection and the report
    # always score on the run's own corpus (its val / test splits).
    clips_data_dir: str = ""
    fit_split: str = "train"
    loss_split: str = "train"
    step_lr: float = 1.0e-2
    # Each step's starting length as a fraction of |e_0|. 0.0226 = 1e-3 * sqrt(512):
    # one Adam step at enrollment's lr 1e-3 on a 512-d vector.
    init_step_frac: float = 0.0226

    @classmethod
    def from_dict(cls, d: dict) -> "UnfoldSpec":
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"unfold: unknown keys {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


@dataclass(frozen=True)
class RunSpec:
    """One experiment, fully resolved. What a run writes down about itself."""

    name: str
    data_dir: Path
    results_dir: Path
    seed: int = DEFAULT_SEED
    sample_rate: int = C.SAMPLE_RATE
    output_format: str = "flac"
    model: ModelSpec = field(default_factory=ModelSpec)
    data: DataSpec = field(default_factory=DataSpec)
    loss: LossSpec = field(default_factory=LossSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    joint: JointSpec | None = None
    unfold: UnfoldSpec | None = None

    def __post_init__(self) -> None:
        # Frozen, so coerce through object.__setattr__: a spec read back from a
        # checkpoint or yaml arrives with strings where Paths belong.
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "results_dir", Path(self.results_dir))
        if self.joint is not None and self.unfold is not None:
            raise ConfigError("a run spec has a joint: or an unfold: section, not both")

    # --- Derived ------------------------------------------------------------------
    @property
    def is_joint(self) -> bool:
        return self.joint is not None

    @property
    def is_unfold(self) -> bool:
        return self.unfold is not None

    @property
    def stage(self) -> str:
        return "unfold" if self.is_unfold else "joint" if self.is_joint else "emulate"

    @property
    def run_dir(self) -> Path:
        return Path(self.results_dir) / self.stage / self.name

    @property
    def paths(self):
        """The corpus this run reads, as a :class:`~openamp.data.CorpusPaths`."""
        from openamp.data import CorpusPaths
        return CorpusPaths.from_data_dir(self.data_dir, self.output_format)

    @property
    def clip_samples(self) -> int:
        return int(round(self.data.clip_seconds * self.sample_rate))

    @property
    def ref_samples(self) -> int:
        return int(round(self.joint.ref.seconds * self.sample_rate))

    def replace(self, **changes) -> "RunSpec":
        """A copy with fields overridden -- how CLI flags reach the spec."""
        return replace(self, **changes)

    # --- Serialization ----------------------------------------------------------
    def to_dict(self) -> dict:
        d = {"name": self.name, "data_dir": str(self.data_dir),
             "results_dir": str(self.results_dir), "seed": self.seed,
             "sample_rate": self.sample_rate, "output_format": self.output_format,
             "model": self.model.to_dict(), "data": self.data.to_dict(),
             "loss": self.loss.to_dict(), "training": self.training.to_dict()}
        if self.joint is not None:
            d["joint"] = self.joint.to_dict()
        if self.unfold is not None:
            d["unfold"] = self.unfold.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunSpec":
        d = dict(d)
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"run spec: unknown keys {sorted(unknown)}")
        joint, unfold = d.get("joint"), d.get("unfold")
        return cls(
            name=str(d["name"]), data_dir=Path(d["data_dir"]),
            results_dir=Path(d["results_dir"]),
            seed=int(d.get("seed", DEFAULT_SEED)),
            sample_rate=int(d.get("sample_rate", C.SAMPLE_RATE)),
            output_format=str(d.get("output_format", "flac")),
            model=ModelSpec.from_dict(d.get("model", {})),
            data=DataSpec.from_dict(d.get("data", {})),
            loss=LossSpec.from_dict(d.get("loss", {})),
            training=TrainingSpec.from_dict(d.get("training", {})),
            joint=JointSpec.from_dict(joint) if joint is not None else None,
            unfold=UnfoldSpec.from_dict(unfold) if unfold is not None else None)


# --- Loading -------------------------------------------------------------------
def load_run_spec(path: Path, *, results_dir: Path | None = None,
                  name: str | None = None, **overrides) -> RunSpec:
    """Read one experiment's yaml into a :class:`RunSpec`. The single boundary call.

    ``name`` defaults to the file stem, as it always has, but the yaml may now
    set it explicitly. ``data_dir`` likewise comes from the yaml, with
    ``OPENAMP_DATA_DIR`` still winning so an existing shell habit keeps working
    -- but a config that names its corpus no longer depends on remembering it.
    """
    path = Path(path)
    y = _read_yaml(path)
    unknown = set(y) - {"name", "data_dir", "results_dir", "seed", "sample_rate",
                        "output_format", "model", "data", "loss", "training", "joint",
                        "unfold"}
    if unknown:
        raise ConfigError(f"{path}: unknown top-level keys {sorted(unknown)}")

    data_dir = os.environ.get("OPENAMP_DATA_DIR") or y.get("data_dir") or "./data"
    seed = int(os.environ.get("OPENAMP_SEED", y.get("seed", DEFAULT_SEED)))
    spec = RunSpec(
        name=str(name or y.get("name") or path.stem),
        data_dir=Path(data_dir).expanduser(),
        results_dir=Path(results_dir or y.get("results_dir")
                         or Path.cwd() / "results").expanduser(),
        seed=seed,
        sample_rate=int(y.get("sample_rate", C.SAMPLE_RATE)),
        output_format=str(y.get("output_format", "flac")),
        model=ModelSpec.from_dict(y.get("model", {})),
        data=DataSpec.from_dict(y.get("data", {})),
        loss=LossSpec.from_dict(y.get("loss", {})),
        training=TrainingSpec.from_dict(y.get("training", {})),
        joint=JointSpec.from_dict(y["joint"]) if "joint" in y else None,
        unfold=UnfoldSpec.from_dict(y["unfold"]) if "unfold" in y else None)
    return spec.replace(**overrides) if overrides else spec


def spec_from_checkpoint(ck: dict) -> RunSpec:
    """The :class:`RunSpec` a checkpoint was trained under.

    A checkpoint written before the v2 schema stores ``emulate_cfg`` instead,
    and cannot be read here -- say so pointing at the one-time migration rather
    than failing later on a missing key.
    """
    if "spec" not in ck:
        raise ConfigError(
            "this checkpoint predates the v2 config schema: it stores "
            "'emulate_cfg'/'joint_cfg' rather than 'spec'. Convert it once with "
            "`python scripts/migrate_checkpoint_v2.py <run dir>`, which adds the "
            "spec and leaves the weights untouched.")
    return RunSpec.from_dict(ck["spec"])
