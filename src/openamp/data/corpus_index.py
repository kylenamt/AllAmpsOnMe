"""Where the corpus lives, and which devices are in the embedding table.

:class:`CorpusPaths` is the one thing a dataset needs to know about the layout on
disk. It is built once at the boundary -- from a ``RunSpec``'s ``data_dir``
(:meth:`CorpusPaths.from_data_dir`) or from a pipeline ``Config``
(:meth:`CorpusPaths.from_config`) -- and nothing below this line reaches for a
global.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from openamp.core import constants as C
from openamp.core import manifest as manifests
from openamp.core.util import sha256_file

__all__ = ["CorpusPaths", "render_ok_devices", "load_or_create_holdout",
           "build_device_index", "manifest_signature", "device_names"]


@dataclass(frozen=True)
class CorpusPaths:
    """Resolved locations of everything a dataset reads."""

    clean_dir: Path              # clean/, the split is appended
    renders_dir: Path            # renders/, the zero-padded device id is appended
    corpus_manifest: Path
    renders_manifest: Path
    device_manifest: Path        # the working acquisition manifest (display names)
    devices_final: Path
    holdout: Path                # emulate_holdout.txt
    output_format: str = "flac"

    @classmethod
    def from_data_dir(cls, data_dir, output_format: str = "flac") -> "CorpusPaths":
        """Derive the whole layout from a corpus root -- what a ``RunSpec`` holds."""
        root = Path(data_dir)
        m = root / "manifests"
        return cls(clean_dir=root / "clean", renders_dir=root / "renders",
                   corpus_manifest=m / C.CORPUS_MANIFEST,
                   renders_manifest=m / C.RENDERS_MANIFEST,
                   device_manifest=m / C.ACQUISITION_MANIFEST,
                   devices_final=m / C.DEVICES_FINAL_MANIFEST,
                   holdout=m / C.EMULATE_HOLDOUT,
                   output_format=str(output_format))

    @classmethod
    def from_config(cls, config) -> "CorpusPaths":
        return cls.from_data_dir(config.data_dir, config.output_format)

    def clean_file(self, split: str, file_id) -> Path:
        return self.clean_dir / split / f"{file_id}.{self.output_format}"

    def render_file(self, device_id: int, file_id) -> Path:
        return self.renders_dir / f"{int(device_id):04d}" / f"{file_id}.{self.output_format}"


def render_ok_devices(paths: CorpusPaths) -> list[int]:
    """Sorted device ids with at least one ``render_ok`` render (any split)."""
    renders = manifests.read_manifest(paths.renders_manifest, manifests.RENDERS_COLUMNS)
    if renders.empty:
        return []
    ok = renders[renders["status"] == C.RENDER_OK]
    return sorted(int(d) for d in ok["device_id"].unique())


def load_or_create_holdout(paths: CorpusPaths, frac: float, seed: int) -> list[int]:
    """Sorted device ids held out of one-to-many training (Phase 5 enrollment set).

    Drawn once as a seeded sample of the render-ok set and persisted to
    ``emulate_holdout.txt``; the file, not the RNG, is the source of truth, so
    the holdout stays fixed even if more devices are rendered later. ``frac <= 0``
    disables the holdout (returns ``[]`` without touching the file).
    """
    if frac <= 0:
        return []
    path = Path(paths.holdout)
    if path.is_file():
        return sorted(int(line) for line in path.read_text(encoding="utf-8").splitlines()
                      if line.strip() and not line.lstrip().startswith("#"))
    pool = render_ok_devices(paths)
    n = max(1, round(frac * len(pool)))
    rng = np.random.default_rng(seed)
    ids = sorted(int(d) for d in rng.choice(pool, size=n, replace=False))
    header = [
        f"# Open-Amp3000 emulate holdout — {len(ids)} of {len(pool)} render-ok devices "
        f"(frac {frac:g}, seed {seed})",
        "# excluded from one-to-many training; Phase 5 enrolls these as unseen devices",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header + [str(d) for d in ids]) + "\n", encoding="utf-8")
    return ids


def build_device_index(paths: CorpusPaths, *, exclude: "set[int] | list[int]" = ()
                       ) -> tuple[list[int], dict[int, int]]:
    """Canonical ``(device_ids, {device_id: row})`` for the embedding table.

    The table spans every render-ok device not in ``exclude``, in a fixed sorted
    order.
    """
    drop = set(exclude)
    ids = [d for d in render_ok_devices(paths) if d not in drop]
    return ids, {d: i for i, d in enumerate(ids)}


def manifest_signature(paths: CorpusPaths) -> str:
    """SHA-256 of the device manifest the table indexes (provenance for Phase 5)."""
    path = paths.devices_final if paths.devices_final.is_file() \
        else paths.renders_manifest
    return sha256_file(path) if Path(path).is_file() else ""


def device_names(paths: CorpusPaths) -> dict[int, str]:
    """``{device_id: display name}`` from the working manifest (best-effort).

    The name is the human ``tone_title`` plus the capture's ``model_name`` variant
    — many devices share a ``tone_title``, so the variant is what tells them apart.
    Returns ``{}`` when the manifest is absent, letting callers fall back to a
    generic ``device <id>`` label.
    """
    df = manifests.read_manifest(paths.device_manifest, manifests.ALL_COLUMNS)
    if df.empty or "device_id" not in df.columns:
        return {}
    df = df.dropna(subset=["device_id"])
    titles = df["tone_title"].fillna("").astype(str).str.strip()
    variants = df["model_name"].fillna("").astype(str).str.strip()
    names: dict[int, str] = {}
    for did, title, variant in zip(df["device_id"].astype(int), titles, variants):
        name = f"{title} — {variant}" if title and variant else (title or variant)
        names[int(did)] = name or f"device {int(did)}"
    return names
