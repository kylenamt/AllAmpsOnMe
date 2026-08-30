"""Datasets and dataloaders for the amp models.

**The item contract.** Every dataset here yields a dict of tensors:

===============  ======================  ==========================================
key              shape / dtype           meaning
===============  ======================  ==========================================
``input``        ``[R + clip]`` fp32     clean audio, prefixed with ``R`` samples of
                                         real left-context so the causal model is
                                         warmed before the loss region starts
``target``       ``[clip]`` fp32         the device's render of the warmed region
``device_idx``   ``[]`` int64            the embedding-table row for that device
``device_id``    ``[]`` int64            grid only: the real device id
``ref``          ``[2, ref]`` fp32       joint only: a disjoint (dry, wet) reference…
                 or ``[]`` int64         …or the device id, when the encoder looks a
                                         precomputed fingerprint up instead
===============  ======================  ==========================================

A model consumes ``input`` and compares ``output[..., R:]`` against ``target``.

**Adding a dataset** = one new module with a class honouring that contract, plus
either an entry in :data:`DATASETS` or a ``@register_dataset("name")`` decorator
on the class. Nothing in a training loop changes.

The built-ins are listed explicitly rather than self-registering, so a name is
never missing because its module happened not to be imported.

Where the corpus lives is a :class:`~openamp.data.corpus_index.CorpusPaths`,
built from a ``Config`` at the boundary — no dataset holds a ``Config``.
"""

from __future__ import annotations

from openamp.data.corpus_index import (CorpusPaths, build_device_index, device_names,
                                       load_or_create_holdout, manifest_signature,
                                       render_ok_devices, training_device_index)
from openamp.data.emulation import EmulationDataset, read_with_retry
from openamp.data.grid import DeviceGridDataset
from openamp.data.joint import REF_MODES, JointDataset
from openamp.data.loaders import make_loader
from openamp.data.pair import WetDryDataset

__all__ = ["CorpusPaths", "build_device_index", "device_names",
           "load_or_create_holdout", "manifest_signature", "render_ok_devices",
           "training_device_index",
           "EmulationDataset", "read_with_retry", "DeviceGridDataset",
           "JointDataset", "REF_MODES", "WetDryDataset", "make_loader",
           "DATASETS", "register_dataset", "build_dataset"]

DATASETS: dict[str, type] = {
    "emulation": EmulationDataset,
    "grid": DeviceGridDataset,
    "joint": JointDataset,
    "pair": WetDryDataset,
}


def register_dataset(name: str):
    """Register a dataset class defined outside this package."""
    def wrap(cls):
        DATASETS[name] = cls
        return cls
    return wrap


def build_dataset(name: str, **kwargs):
    """Construct a registered dataset."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    return DATASETS[name](**kwargs)
