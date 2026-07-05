"""Download + reassemble GUITAR-FX-DIST (GFX) Mono-Discrete into data/raw/gfx/.

GFX is the paper's primary distortion-effect benchmark (spec §4.1.1), published as
a ~27 GB multi-part split zip on Zenodo (Comunità et al.). This downloads every
segment, reassembles the split archive with 7z, and extracts it. Idempotent —
skips segments already present and the extract step if it already ran.

    python scripts/download_gfx.py [--record 4298000] [--dest data/raw/gfx]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

DEFAULT_RECORD = "4298000"      # GUITAR-FX-DIST, Mono Discrete


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def _download(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"  {dest.name}: present -> skip")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=180) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with tmp.open("wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total and done % (64 << 20) < (1 << 20):
                    print(f"\r  {dest.name}: {done/1e9:5.2f}/{total/1e9:.2f} GB", end="", flush=True)
    print()
    tmp.rename(dest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=DEFAULT_RECORD)
    ap.add_argument("--dest", type=Path, default=Path("data/raw/gfx"))
    ap.add_argument("--download-dir", type=Path, default=Path("data/raw/gfx_download"))
    args = ap.parse_args(argv)
    args.dest.mkdir(parents=True, exist_ok=True)
    args.download_dir.mkdir(parents=True, exist_ok=True)

    if any(args.dest.rglob("*.wav")):
        print(f"GFX already extracted at {args.dest} -> nothing to do")
        return 0

    rec = _fetch_json(f"https://zenodo.org/api/records/{args.record}")
    files = rec.get("files", [])
    print(f"GFX record {args.record}: {len(files)} files, "
          f"{sum(f.get('size', 0) for f in files)/1e9:.1f} GB")
    main_zip = None
    for f in sorted(files, key=lambda f: f["key"]):
        key = f["key"]
        url = f.get("links", {}).get("self") or f.get("links", {}).get("download")
        dest = args.download_dir / key
        print(f"  downloading {key} ...")
        _download(url, dest)
        if key.lower().endswith(".zip") and "Discrete" in key:
            main_zip = dest

    if main_zip is None:
        print("!! could not identify the main .zip segment; downloaded files:",
              [f["key"] for f in files])
        return 1

    print(f"reassembling + extracting {main_zip.name} with 7z ...")
    # 7z reads the split .z01..zNN segments alongside the final .zip automatically.
    res = subprocess.run(["7z", "x", "-y", f"-o{args.dest}", str(main_zip)],
                         capture_output=True, text=True)
    sys.stdout.write(res.stdout[-2000:])
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-2000:])
        print("!! 7z failed; try manual: "
              f"zip -s- {main_zip} --out full.zip && unzip full.zip -d {args.dest}")
        return res.returncode
    n_wav = sum(1 for _ in args.dest.rglob("*.wav"))
    print(f"GFX ready at {args.dest} ({n_wav} wav files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
