"""Eval adapters: class discovery, labels/groups, note grouping, crop slicing."""

import numpy as np
import pytest

from eval.adapters import EffectClip, EffectDataset, _note_group, folder_adapter


def _make_tree(root, layout):
    """layout: {class_name: [relative_wav_paths]} -> touch empty .wav files."""
    for cls, files in layout.items():
        for rel in files:
            p = root / cls / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"RIFF")             # placeholder; adapters only resolve paths


def test_folder_adapter_discovers_classes_and_labels(tmp_path):
    _make_tree(tmp_path, {
        "Clean": ["Bridge/6-1.wav", "Neck/6-2.wav"],
        "RAT": ["Bridge/6-1.wav", "Bridge/6-3.wav"],
    })
    ds = folder_adapter(tmp_path, "toy", group_fn=_note_group)
    assert ds.class_names == ["Clean", "RAT"]
    assert len(ds) == 4
    assert set(ds.labels.tolist()) == {0, 1}
    # groups derive from the note token, not the class or pickup folder.
    assert set(ds.groups.tolist()) == {"6-1", "6-2", "6-3"}


def test_folder_adapter_errors_on_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        folder_adapter(tmp_path, "empty")


def test_note_group_extracts_string_fret():
    from pathlib import Path
    assert _note_group(Path("x/6-20.wav")) == "6-20"
    assert _note_group(Path("x/nogroup.wav")) == "nogroup"


def test_folder_adapter_exclude_drops_classes(tmp_path):
    _make_tree(tmp_path, {
        "808": ["G61-40100-808-O10T0-20593.wav"],
        "BD2": ["G61-40100-BD2-O5T5-20601.wav"],
        "_NoFX_mono": ["G61-40100-clean-00-20000.wav"],
    })
    ds = folder_adapter(tmp_path, "GFX", exclude=lambda d: d.name.startswith("_NoFX"))
    assert ds.class_names == ["808", "BD2"]            # dry reference excluded


def test_gfx_group_is_guitar_plus_clean_id():
    from pathlib import Path
    from eval.adapters import _gfx_group
    # same clean recording across effects/settings -> same group (leak-free)
    assert _gfx_group(Path("G61-40100-808-O10T0-20593.wav")) == "G61-40100"
    assert _gfx_group(Path("G61-40100-BD2-O5T5-20601.wav")) == "G61-40100"
    assert _gfx_group(Path("G61-41101-808-O2T0-20777.wav")) == "G61-41101"


def test_effect_dataset_labels_groups_arrays():
    ds = EffectDataset("d", [EffectClip("a.wav", 0, "g0"),
                             EffectClip("b.wav", 1, "g1")], ["x", "y"])
    assert ds.labels.tolist() == [0, 1]
    assert ds.groups.tolist() == ["g0", "g1"]


def test_crops_from_signal_shapes():
    from eval.embed import crops_from_signal
    # short clip -> single zero-padded crop
    one = crops_from_signal(np.ones(1000, np.float32), 96_000, max_crops=8)
    assert one.shape == (1, 96_000)
    # long clip -> capped, non-overlapping crops
    many = crops_from_signal(np.ones(96_000 * 20, np.float32), 96_000, max_crops=8)
    assert many.shape == (8, 96_000)
