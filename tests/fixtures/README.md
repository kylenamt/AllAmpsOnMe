# Test fixtures

The acquisition validation tests (`tests/test_validate.py`) can exercise the real
`neural-amp-modeler` load/parse path against two tiny local `.nam` files. These are
**not committed** (a `.nam` is a third-party capture format and we keep the repo free
of binary/model blobs), so those specific tests are `skipif`-guarded and skipped on a
fresh checkout — the rest of the suite runs normally.

To enable them, drop two files here:

- **`tiny_a2.nam`** — a valid NAM export whose config parses to
  `{"architecture": "WaveNet", "sample_rate": 48000}`. The smallest way to get one is
  to export any real capture from the NAM trainer / TONE3000, or to hand-write the
  minimal JSON envelope NAM accepts (`version`, `architecture: "WaveNet"`, a small
  `config`, and a `weights` array). It only needs to *load*; the render is injected in
  the tests, so the weights can be tiny.
- **`broken.nam`** — any file that is **not valid JSON** (e.g. a single line of
  garbage text). `test_parse_nam_fixtures` asserts it raises `NamBackendError`.

With both present, the guarded cases in `test_validate.py` run automatically.
