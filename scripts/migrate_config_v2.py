#!/usr/bin/env python3
"""Rewrite a v1 ``emulate:``/``joint:`` config into the v2 five-section schema.

    python scripts/migrate_config_v2.py configs/emulate/*.yaml           # in place
    python scripts/migrate_config_v2.py --stdout configs/emulate/x.yaml  # preview

**Line-based, not a YAML round-trip.** These files carry more research prose
than YAML -- derivations, run histories, why a knob is where it is -- and
``yaml.safe_dump`` would delete every comment. So this moves each ``key: value``
line to its new section and takes the key's own comment block (the comment lines
immediately above it, and any trailing ``#`` on the line itself) with it. The
file header, blank-line-separated from the first key, stays at the top untouched.

Verified by :func:`check`, which loads both files and asserts the resolved specs
are field-for-field equal.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# v1 key -> (v2 section, v2 key). A rename means the v2 name is the *constructor's*
# parameter name, so a section's contents pass straight through as kwargs.
EMULATE_MAP = {
    "arch":                  ("model", "arch"),
    "wn_channels":           ("model", "channels"),
    "wn_activation":         ("model", "activation"),
    "cond_hidden":           ("model", "cond_hidden"),
    "cond_activation":       ("model", "cond_activation"),
    "delta_rank":            ("model", "delta_rank"),
    "embedding_dim":         ("model", "embedding_dim"),
    "embed_normalize":       ("model", "embed_normalize"),

    "clip_seconds":          ("data", "clip_seconds"),
    "pairs_per_epoch":       ("data", "pairs_per_epoch"),
    "holdout_frac":          ("data", "holdout_frac"),

    "stft_weight":           ("loss", "stft_weight"),
    "preemph":               ("loss", "preemph_coeff"),
    "device_loss_weighting": ("loss", "device_loss_weighting"),

    "lr":                    ("training", "lr"),
    "weight_decay":          ("training", "weight_decay"),
    "plateau_patience":      ("training", "plateau_patience"),
    "plateau_factor":        ("training", "plateau_factor"),
    "early_stop_patience":   ("training", "early_stop_patience"),
    "epochs":                ("training", "epochs"),
    "batch_size":            ("training", "batch_size"),
    "num_workers":           ("training", "num_workers"),
    "amp":                   ("training", "amp"),
    "amp_dtype":             ("training", "amp_dtype"),
    "val_pairs":             ("training", "val_pairs"),
    "log_every":             ("training", "log_every"),
    "embed_noise_frac":      ("training", "embed_noise_frac"),
}

# The joint section nests one level: encoder knobs lose the enc_ prefix and
# become the encoder constructor's own names; the window-B knobs become `ref`.
JOINT_MAP = {
    "enc_kind":                ("joint.encoder", "kind"),
    "fp_path":                 ("joint.encoder", "path"),
    "fp_preprocess":           ("joint.encoder", "preprocess"),
    "fp_layernorm":            ("joint.encoder", "layernorm"),
    "fp_frames_path":          ("joint.encoder", "frames_path"),
    "enc_pool_queries":        ("joint.encoder", "n_queries"),
    "enc_pool_dim":            ("joint.encoder", "d_model"),
    "enc_frame_window":        ("joint.encoder", "frame_window"),
    "enc_channels":            ("joint.encoder", "channels"),
    "enc_activation":          ("joint.encoder", "activation"),
    "enc_attn_hidden":         ("joint.encoder", "attn_hidden"),
    "enc_proj_hidden":         ("joint.encoder", "proj_hidden"),
    "enc_normalize":           ("joint.encoder", "normalize"),
    "enc_head":                ("joint.encoder", "head"),
    "enc_taps":                ("joint.encoder", "taps"),
    "enc_tap_stride":          ("joint.encoder", "tap_stride"),
    "enc_expand_dim":          ("joint.encoder", "expand_dim"),
    "enc_n_heads":             ("joint.encoder", "n_heads"),
    "enc_expand_norm":         ("joint.encoder", "expand_norm"),
    "enc_norm_groups":         ("joint.encoder", "norm_groups"),

    "ref_seconds":             ("joint.ref", "seconds"),
    "ref_different_file":      ("joint.ref", "different_file"),
    "ref_min_gap_seconds":     ("joint.ref", "min_gap_seconds"),

    "sources":                 ("data", "sources"),
    "enc_lr":                  ("joint", "enc_lr"),
    "enc_weight_decay":        ("joint", "enc_weight_decay"),
    "grad_checkpoint_encoder": ("joint", "grad_checkpoint_encoder"),
}

SECTIONS = ["model", "data", "loss", "training", "joint.encoder", "joint.ref", "joint"]
_MISSING = object()

# v2 passes a section's keys straight to a constructor, so a key that arm does
# not take must not survive the move. v1 tolerated them: EmulateConfig held every
# arch's knobs at once and from_config picked out the relevant ones, which is why
# a dac_frames config still carries enc_channels. Those are dropped here -- the
# v1 code ignored them too, so nothing changes but the noise.
MODEL_KEYS = {
    "film_wavenet":    ("arch", "channels", "activation", "embedding_dim",
                        "embed_normalize"),
    "mlpfilm_wavenet": ("arch", "channels", "activation", "embedding_dim",
                        "embed_normalize", "cond_hidden", "cond_activation"),
    "delta_wavenet":   ("arch", "channels", "activation", "embedding_dim",
                        "embed_normalize", "delta_rank"),
}
ENCODER_KEYS = {
    "wavenet":     ("kind", "channels", "activation", "attn_hidden", "proj_hidden",
                    "normalize", "head", "taps", "tap_stride", "expand_dim",
                    "n_heads", "expand_norm", "norm_groups"),
    "fingerprint": ("kind", "path", "preprocess", "layernorm", "proj_hidden",
                    "activation", "normalize"),
    "dac_frames":  ("kind", "frames_path", "n_queries", "d_model", "proj_hidden",
                    "activation", "normalize", "frame_window"),
}

# Where a v1 dataclass default differs from the v2 constructor default it now
# falls back to, the migrated file must say the old value out loud or the run
# silently changes. Audited: these are the only four.
FORCED = {
    ("joint.encoder", "wavenet"):     {"activation": "tanh", "channels": 16},
    ("joint.encoder", "fingerprint"): {"normalize": False},
    ("joint.encoder", "dac_frames"):  {"normalize": False},
}
KEY_RE = re.compile(r"^(\s+)([a-z_][a-z0-9_]*):(.*)$")
DATA_DIR_RE = re.compile(r"OPENAMP_DATA_DIR=([\w./-]+)")
TOP_RE = re.compile(r"^([a-z_][a-z0-9_]*):\s*$")


# v1's own section dividers ("# --- Model ---"): they label groups this schema
# now expresses as real sections, so carrying them over would put a heading like
# "--- Training ---" inside `data:`. Dropped; every other comment is kept.
DIVIDER_RE = re.compile(r"^\s*#\s*-{2,}.*$")


class Entry:
    """One config key, with the comment lines that belong to it.

    ``lead`` is the block above the key, ``tail`` the wrapped continuation of its
    own trailing comment -- told apart by indentation, since a continuation is
    aligned under the ``#`` rather than under the key.
    """

    def __init__(self, key, value, comment, lead):
        self.key, self.value, self.comment, self.lead = key, value, comment, lead
        self.tail: list[str] = []

    def render(self, name, indent) -> list[str]:
        pad = " " * indent
        out = [pad + c.strip() if c.strip() else "" for c in self.lead]
        head = f"{pad}{name}: {self.value}"
        out.append(head + (f"  # {self.comment}" if self.comment else ""))
        col = " " * (len(head) + 2)               # align under the trailing '#'
        out += [col + "# " + c for c in self.tail]
        return out


def parse(text: str):
    """``(header_lines, {v1_section: [Entry]})``, keeping every comment."""
    header, sections, cur, lead = [], {}, None, []
    for raw in text.splitlines():
        line = raw.rstrip()
        top = TOP_RE.match(line)
        if top and top.group(1) in ("emulate", "joint", "enroll"):
            cur, lead = top.group(1), []
            sections[cur] = []
            continue
        if cur is None:
            header.append(line)
            continue
        m = KEY_RE.match(line)
        if m:
            indent, key, rest = m.groups()
            value, _, comment = rest.partition("#")
            sections[cur].append(Entry(key, value.strip(), comment.strip(), lead))
            sections[cur][-1].indent = len(indent)
            lead = []
        elif DIVIDER_RE.match(line):
            continue
        elif line.strip().startswith("#"):
            prev = sections[cur][-1] if sections[cur] else None
            wrapped = (prev is not None and prev.comment and not lead
                       and len(line) - len(line.lstrip()) > prev.indent)
            (prev.tail if wrapped else lead).append(
                line.strip().lstrip("#").strip() if wrapped else line)
        elif not line.strip():
            lead.append(line)
        else:
            sections[cur][-1].value += "\n" + line     # a list item
    return header, sections


def convert(text: str) -> tuple[str, list[str]]:
    header, sections = parse(text)
    arch = _value(sections.get("emulate"), "arch") or "film_wavenet"
    kind = _value(sections.get("joint"), "enc_kind") or "wavenet"

    buckets: dict[str, list[tuple[str, Entry]]] = {s: [] for s in SECTIONS}
    dropped = []
    for v1_section, entries in sections.items():
        table = EMULATE_MAP if v1_section == "emulate" else JOINT_MAP
        for e in entries:
            if e.key not in table:
                raise SystemExit(f"no v2 home for {v1_section}.{e.key}")
            dest, new_key = table[e.key]
            allowed = (MODEL_KEYS[arch] if dest == "model" else
                       ENCODER_KEYS[kind] if dest == "joint.encoder" else None)
            if allowed is not None and new_key not in allowed:
                dropped.append(f"{v1_section}.{e.key} (inert under "
                               f"{arch if dest == 'model' else kind})")
                continue
            buckets[dest].append((new_key, e))

    for (dest, which), forced in FORCED.items():
        if (dest == "joint.encoder" and which != kind) or not buckets[dest]:
            continue
        present = {k for k, _ in buckets[dest]}
        for key, value in forced.items():
            if key not in present and key in ENCODER_KEYS[kind]:
                buckets[dest].append((key, Entry(
                    key, "true" if value is True else
                    "false" if value is False else str(value),
                    "was the v1 JointConfig default; the constructor's differs",
                    [])))

    out = list(header)
    while out and not out[-1].strip():
        out.pop()
    # The corpus a run trained on lived in an OPENAMP_DATA_DIR shell prefix, which
    # only the header comment recorded -- so a config could silently train on the
    # wrong corpus, and has. If the header names exactly one, promote it to a key.
    corpora = {m for line in header for m in DATA_DIR_RE.findall(line)}
    if len(corpora) == 1:
        out += ["", f"data_dir: {corpora.pop()}   # was an OPENAMP_DATA_DIR prefix "
                    f"in the commands above"]
    out.append("")
    if not any(k == "dataset" for k, _ in buckets["data"]):
        buckets["data"].insert(0, ("dataset", Entry(
            "dataset", "joint" if sections.get("joint") else "emulation", "", [])))
    if not any(k == "name" for k, _ in buckets["loss"]):
        buckets["loss"].insert(0, ("name", Entry("name", "esr_mrstft", "", [])))

    for section in SECTIONS:
        items = buckets[section]
        if not items:
            continue
        if section == "joint.encoder":
            out += ["joint:", "  encoder:"]
            indent = 4
        elif section == "joint.ref":
            out += ["  ref:"] if buckets["joint.encoder"] else ["joint:", "  ref:"]
            indent = 4
        elif section == "joint":
            if not (buckets["joint.encoder"] or buckets["joint.ref"]):
                out.append("joint:")
            indent = 2
        else:
            out.append(f"{section}:")
            indent = 2
        for i, (new_key, e) in enumerate(items):
            if i == 0:                       # no blank line right under a header
                while e.lead and not e.lead[0].strip():
                    e.lead.pop(0)
            out += e.render(new_key, indent)
        out.append("")
    return "\n".join(out).rstrip() + "\n", dropped


def _value(entries, key):
    """The raw value of one v1 key, for picking the arch/kind before converting."""
    for e in entries or ():
        if e.key == key:
            return e.value.strip()
    return None


def check(v1_text: str, v2_text: str, stem: str) -> None:
    """Assert every value the v1 file sets reaches its v2 home, unchanged.

    Reads both files as plain YAML rather than through the config dataclasses:
    the v1 ones are deleted by the same change this script prepares for, and a
    one-time migration tool should not stop working the moment they go.
    """
    import yaml

    old = yaml.safe_load(v1_text) or {}
    new = yaml.safe_load(v2_text) or {}

    def get(section: str, key: str):
        node = new
        for part in section.split("."):
            if not isinstance(node, dict):
                return _MISSING
            node = node.get(part, _MISSING)
            if node is _MISSING:
                return _MISSING
        return node.get(key, _MISSING) if isinstance(node, dict) else _MISSING

    bad = []
    for v1_section, table in (("emulate", EMULATE_MAP), ("joint", JOINT_MAP)):
        for key, v1_value in (old.get(v1_section) or {}).items():
            section, new_key = table[key]
            v2_value = get(section, new_key)
            if v2_value is _MISSING:
                continue                    # dropped as inert under this arch/kind
            if isinstance(v1_value, (list, tuple)) or isinstance(v2_value, (list, tuple)):
                same = list(v1_value or []) == list(v2_value or [])
            else:
                same = v1_value == v2_value
            if not same:
                bad.append(f"{v1_section}.{key} -> {section}.{new_key}: "
                           f"{v1_value!r} became {v2_value!r}")
    if bad:
        raise SystemExit(f"{stem}: migration changed values:\n  " + "\n  ".join(bad))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = ap.parse_args()
    for path in args.paths:
        v1 = path.read_text(encoding="utf-8")
        if "\nemulate:" not in "\n" + v1:
            print(f"  skip {path} (no emulate: section)")
            continue
        v2, dropped = convert(v1)
        check(v1, v2, path.stem)
        if args.stdout:
            print(f"# ===== {path} =====\n{v2}")
        else:
            path.write_text(v2, encoding="utf-8")
            note = f"  (dropped {', '.join(dropped)})" if dropped else ""
            print(f"  migrated {path}{note}")


if __name__ == "__main__":
    main()
