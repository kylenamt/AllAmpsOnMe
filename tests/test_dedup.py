import numpy as np
import pandas as pd

from openamp.dsp import audio as probe_mod
from openamp.core import manifest
from openamp.acquire import dedup
from openamp.core.manifest import STATUS_DUPLICATE, STATUS_VALIDATED


def _save_probe(settings, name, arr):
    path = settings.captures_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32))
    return str(path)


def test_dedup_esr_and_hash(settings):
    base = probe_mod.generate_probe(2000).signal
    rng = np.random.default_rng(0)

    rows = [
        # ESR group (fender/deluxe): r2 is a near-copy of r1 -> duplicate (lower downloads).
        dict(model_id=1, make="fender", model="deluxe", downloads=100, sha256="A",
             probe=base.copy()),
        dict(model_id=2, make="fender", model="deluxe", downloads=50, sha256="B",
             probe=base + 1e-4 * rng.standard_normal(base.shape).astype(np.float32)),
        dict(model_id=3, make="fender", model="deluxe", downloads=200, sha256="C",
             probe=-base.copy()),  # very different -> kept
        # Exact-hash pair (marshall): identical sha256 -> lower downloads dropped.
        dict(model_id=4, make="marshall", model="jcm800", downloads=10, sha256="H",
             probe=base.copy()),
        dict(model_id=5, make="marshall", model="jcm800", downloads=20, sha256="H",
             probe=(0.5 * base).astype(np.float32)),
    ]
    records = []
    for r in rows:
        pp = _save_probe(settings, f"{r['model_id']}.probe.npy", r.pop("probe"))
        r["probe_path"] = pp
        r["status"] = STATUS_VALIDATED
        records.append(r)
    df = manifest.ensure_schema(pd.DataFrame(records))

    out = dedup.dedup(df, settings)
    status = dict(zip(out["model_id"], out["status"]))

    assert status[2] == STATUS_DUPLICATE       # ESR near-duplicate of #1
    assert status[1] == STATUS_VALIDATED
    assert status[3] == STATUS_VALIDATED       # distinct probe survives
    assert status[4] == STATUS_DUPLICATE       # exact-hash, lower downloads
    assert status[5] == STATUS_VALIDATED
