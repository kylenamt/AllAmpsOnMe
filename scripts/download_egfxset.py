"""Download + unzip EGFxSet (Zenodo record 7044411) into data/raw/egfxset/.

External eval dataset for Phase 3 (spec §4.1): real-hardware guitar effects, one
zip per effect class. Idempotent — skips files already downloaded/extracted.

    python scripts/download_egfxset.py [--dest data/raw/egfxset]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

RECORD = "7044411"
API = f"https://zenodo.org/api/records/{RECORD}"


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with tmp.open("wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {dest.name}: {done/1e6:7.1f}/{total/1e6:.1f} MB "
                          f"({pct:5.1f}%)", end="", flush=True)
    print()
    tmp.rename(dest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=Path("data/raw/egfxset"))
    args = ap.parse_args(argv)
    args.dest.mkdir(parents=True, exist_ok=True)

    rec = _fetch_json(API)
    files = rec.get("files", [])
    print(f"EGFxSet: {len(files)} files")
    for f in files:
        key = f["key"]
        url = f.get("links", {}).get("self") or f.get("links", {}).get("download")
        zpath = args.dest / key
        extracted_dir = args.dest / Path(key).stem
        if key.endswith(".zip") and extracted_dir.is_dir() and any(extracted_dir.rglob("*.wav")):
            print(f"  {key}: already extracted -> skip")
            continue
        if not zpath.is_file():
            print(f"  downloading {key} ...")
            _download(url, zpath)
        if key.endswith(".zip"):
            print(f"  extracting {key} ...")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(args.dest)
            zpath.unlink()  # reclaim space after extraction
    print("EGFxSet ready at", args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
