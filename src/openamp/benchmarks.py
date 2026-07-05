"""External / internal evaluation dataset adapters (Phase 3 spec §4.1).

Each adapter yields an :class:`EffectDataset` — a flat list of
``(path, label, group)`` clips plus the class vocabulary. ``group`` is the
underlying clean source (note / phrase / DI file) so the probe split can avoid
leaking the same clean content across train and test. Audio loading itself lives
in :mod:`eval.probes` (crop + embed); adapters only resolve paths and labels.

Adapters:
- :func:`egfxset_adapter`  — EGFxSet (Zenodo 7044411), a folder per effect class.
- :func:`folder_adapter`   — generic "one subdir per class" (GFX, EGDB amp dirs).
- :func:`internal_amp_adapter` — in-domain sanity/UMAP set from our own renders
  (encoder trained on these devices — labelled as such, not an external number).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_AUDIO_EXT = (".wav", ".flac", ".aif", ".aiff", ".mp3", ".ogg")


@dataclass
class EffectClip:
    path: str
    label: int
    group: str          # underlying clean-source key (for leak-free splits)


@dataclass
class EffectDataset:
    name: str
    clips: list = field(default_factory=list)
    class_names: list = field(default_factory=list)
    sample_rate: int = 48_000
    note: str = ""      # free-text caveat surfaced in the report (e.g. in-domain)

    def __len__(self) -> int:
        return len(self.clips)

    @property
    def labels(self):
        import numpy as np
        return np.asarray([c.label for c in self.clips], dtype=np.int64)

    @property
    def groups(self):
        import numpy as np
        return np.asarray([c.group for c in self.clips], dtype=object)


def cap_per_class(dataset: "EffectDataset", max_per_class: int, *, seed: int = 1234
                  ) -> "EffectDataset":
    """Return a balanced subsample with ≤ ``max_per_class`` clips per class.

    For big datasets (GFX has 100k+ clips across settings × clean recordings) a
    balanced per-class cap keeps probe runtime sane without biasing accuracy.
    Sampling prefers spreading across distinct groups so the split stays leak-free.
    """
    import numpy as np

    if max_per_class <= 0:
        return dataset
    rng = np.random.default_rng(seed)
    by_label: dict[int, list] = {}
    for c in dataset.clips:
        by_label.setdefault(c.label, []).append(c)
    kept = []
    for label, clips in by_label.items():
        if len(clips) <= max_per_class:
            kept.extend(clips)
        else:
            idx = rng.choice(len(clips), max_per_class, replace=False)
            kept.extend(clips[i] for i in idx)
    return EffectDataset(dataset.name, kept, dataset.class_names,
                         dataset.sample_rate, dataset.note)


def _class_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir()
                  and any(d.rglob(f"*{e}") for e in _AUDIO_EXT))


def _audio_files(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in _AUDIO_EXT)


# EGFxSet filenames encode the played note, e.g. "2-7-..." or "...-E2-...". We
# group by the leading "<string>-<fret>" token when present so the same physical
# note never straddles the split; otherwise each file is its own group.
_NOTE_RE = re.compile(r"(\d{1,2}-\d{1,2})")


def _note_group(path: Path) -> str:
    m = _NOTE_RE.search(path.stem)
    return m.group(1) if m else path.stem


def folder_adapter(root: str | Path, name: str, *, sample_rate: int = 48_000,
                   group_fn=None, exclude=None, note: str = "") -> EffectDataset:
    """Generic adapter: each immediate subdirectory of ``root`` is one class.

    ``exclude(dir)`` drops class dirs (e.g. GFX's ``_NoFX`` dry references);
    ``group_fn(path)`` sets the leak-free split group per clip.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{name}: dataset dir not found: {root}")
    class_dirs = _class_dirs(root)
    if exclude is not None:
        class_dirs = [d for d in class_dirs if not exclude(d)]
    if not class_dirs:
        raise FileNotFoundError(f"{name}: no class subdirs with audio under {root}")
    class_names = [d.name for d in class_dirs]
    group_fn = group_fn or (lambda p: p.stem)
    clips = []
    for label, d in enumerate(class_dirs):
        for p in _audio_files(d):
            clips.append(EffectClip(str(p), label, str(group_fn(p))))
    return EffectDataset(name, clips, class_names, sample_rate, note)


def egfxset_adapter(root: str | Path = "data/raw/egfxset",
                    sample_rate: int = 48_000) -> EffectDataset:
    """EGFxSet: real-hardware guitar effects, one folder per class (spec §4.1.2).

    NOTE: most EGFxSet classes are modulation/delay/reverb effects that our
    amp/distortion-trained encoder never saw — out of scope. See
    :func:`egfxset_drive_adapter` for the in-scope drive/distortion subset.
    """
    return folder_adapter(root, "EGFxSet", sample_rate=sample_rate,
                          group_fn=_note_group,
                          note="external, real hardware; mostly OUT-OF-SCOPE effect "
                               "families (mod/delay/reverb) for an amp-trained encoder")


# EGFxSet drive/distortion classes — the families our amp-trained encoder can be
# expected to represent (gain/tonal character), matched name-insensitively.
_EGFX_DRIVE = {"clean", "bluesdriver", "rat", "tubescreamer"}


def _norm_name(s: str) -> str:
    return s.lower().replace(" ", "").replace("-", "").replace("_", "")


def egfxset_drive_adapter(root: str | Path = "data/raw/egfxset",
                          sample_rate: int = 48_000) -> EffectDataset:
    """EGFxSet restricted to its drive/distortion classes (in-scope, spec §4.2).

    Clean + BluesDriver + RAT + TubeScreamer — the gain/distortion families an
    amp-trained encoder should separate. A fair external comparison to the full
    13-class set, which mixes in modulation/delay/reverb the encoder never saw.
    """
    return folder_adapter(root, "EGFxSet-drive", sample_rate=sample_rate,
                          group_fn=_note_group,
                          exclude=lambda d: _norm_name(d.name) not in _EGFX_DRIVE,
                          note="external, real hardware; IN-SCOPE drive/distortion "
                               "subset (Clean, BluesDriver, RAT, TubeScreamer)")


# GFX/GUITAR-FX-DIST filenames: <guitar>-<cleanID>-<effect>-<settings>-<sampleID>,
# e.g. "G61-40100-808-O10T0-20593". The clean source (for a leak-free split) is the
# guitar + clean-recording id (first two tokens); the effect is the folder name.
def _gfx_group(path) -> str:
    parts = path.stem.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else path.stem


def _find_gfx_audio_dir(root):
    """Locate the dir whose subfolders are the effect classes.

    GFX layout is ``<root>/Mono_Discrete/Audio/<effect>/`` with a sibling
    ``Features/`` dir, so prefer an ``Audio`` dir (depth-limited) that holds ≥2
    class subdirs; otherwise fall back to whichever dir has the most classes.
    """
    root = Path(root)
    for pat in ("Audio", "*/Audio", "*/*/Audio"):
        for audio in sorted(root.glob(pat)):
            if audio.is_dir() and len(_class_dirs(audio)) >= 2:
                return audio
    cands = [root] + ([d for d in root.iterdir() if d.is_dir()] if root.is_dir() else [])
    return max(cands, key=lambda d: len(_class_dirs(d)) if d.is_dir() else 0)


def gfx_adapter(root: str | Path = "data/raw/gfx",
                sample_rate: int = 48_000) -> EffectDataset:
    """GFX / GUITAR-FX-DIST distortion set — one folder per effect (spec §4.1.1).

    Classes are the distortion-effect folders; the dry ``_NoFX`` reference folders
    are excluded. Split groups by clean recording so no clean phrase leaks.
    """
    audio_dir = _find_gfx_audio_dir(root)
    return folder_adapter(audio_dir, "GFX", sample_rate=sample_rate,
                          group_fn=_gfx_group,
                          exclude=lambda d: d.name.startswith("_NoFX"),
                          note="external; distortion effects on IDMT clean recordings")


def egdb_amp_adapter(root: str | Path = "data/raw/egdb_rendered",
                     sample_rate: int = 48_000) -> EffectDataset:
    """EGDB amp-rendered folders — one folder per amp class (spec §4.1.3)."""
    return folder_adapter(root, "EGDB-amps", sample_rate=sample_rate,
                          group_fn=lambda p: p.stem,
                          note="external; EGDB amp renders (overlaps our clean DI class)")


def internal_amp_adapter(p2_config, *, split: str = "test", label_by: str = "make",
                         max_devices_per_class: int = 6, min_class_size: int = 3,
                         max_classes: int = 12, max_clips: int = 4000,
                         seed: int = 1234) -> EffectDataset:
    """In-domain amp-classification set from our renders (sanity/UMAP, not external).

    Labels rendered clips by their device's ``make`` (or ``gain_bucket``), groups
    by clean ``file_id`` so no clean phrase leaks across the split. Keeps the
    ``max_classes`` best-populated makes (each with ≥ ``min_class_size`` devices)
    so it is a tractable ~12-class amp-ID probe, mirroring the paper's EGDB amp
    classification. The encoder trained on these devices, so it is reported as an
    in-domain sanity number, separately from the external datasets.
    """
    import numpy as np
    import pandas as pd

    from data import constants as C
    from data import manifests

    renders = manifests.read(p2_config.renders_manifest_path, manifests.RENDERS_COLUMNS)
    man = pd.read_parquet(p2_config.phase1_manifest_path,
                          columns=["device_id", "make", "model", "gain_bucket", "gear_type"])
    ok = renders[(renders["split"] == split) & (renders["status"] == C.RENDER_OK)].copy()
    meta = man.set_index("device_id")
    ok = ok.join(meta, on="device_id")
    ok = ok[ok["gear_type"] == "amp"]
    ok = ok[ok[label_by].notna() & (ok[label_by].astype(str).str.len() > 0)]

    # Rank classes by distinct-device count; keep the largest `max_classes` that
    # clear the floor, capping devices per class for balance.
    rng = np.random.default_rng(seed)
    dev_counts = ok.groupby(label_by)["device_id"].nunique().sort_values(ascending=False)
    eligible = [c for c, n in dev_counts.items() if n >= min_class_size][:max_classes]
    if not eligible:
        raise ValueError("internal_amp_adapter: no classes met the size floor")
    keep_rows = []
    for cls in eligible:
        grp = ok[ok[label_by] == cls]
        devs = sorted(grp["device_id"].unique())
        if len(devs) > max_devices_per_class:
            devs = list(rng.choice(devs, max_devices_per_class, replace=False))
        keep_rows.append(grp[grp["device_id"].isin(devs)])
    sel = pd.concat(keep_rows, ignore_index=True)

    class_names = sorted(sel[label_by].astype(str).unique())
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    clips = [EffectClip(str(r.path), cls_to_idx[str(getattr(r, label_by))], str(r.file_id))
             for r in sel.itertuples(index=False)]
    if len(clips) > max_clips:
        idx = rng.choice(len(clips), max_clips, replace=False)
        clips = [clips[i] for i in idx]
    return EffectDataset(f"internal-amp-by-{label_by}", clips, class_names,
                         p2_config.sample_rate,
                         note="IN-DOMAIN (encoder trained on these devices)")
