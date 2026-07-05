"""Global constants for the Phase 2 pipeline (spec §2, §3).

Fixed audio facts and vocabulary shared across corpus prep, rendering,
verification, and the dataset classes. Kept in one place so the clip grid, the
render slicing, and the datasets all agree by construction.
"""

from __future__ import annotations

# --- Fixed audio facts (spec §2) -----------------------------------------------
SAMPLE_RATE = 48_000            # Hz, mono, float32 everywhere
CLIP_SECONDS = 2.0
CLIP_SAMPLES = int(round(CLIP_SECONDS * SAMPLE_RATE))   # 96_000

# --- Corpus level normalization (spec §3.2.3) ----------------------------------
TARGET_RMS_DBFS = -18.0         # guitar-DI-typical level NAM captures expect
PEAK_LIMIT_DBFS = -1.0          # hard-limit true peaks after RMS normalization
MIN_SOURCE_SECONDS = 4.0        # reject EGDB files shorter than this
CLIP_FRACTION_MAX = 0.01        # reject files with >1% samples at full scale

# --- Rendering (spec §4) -------------------------------------------------------
DEFAULT_CHUNK_SECONDS = 10.0
OUTPUT_PEAK_CEILING = 0.999     # per-device output scalar targets this ceiling
DISK_SAFETY_FACTOR = 1.3        # refuse to start if free space < factor x estimate

# --- Verification thresholds (spec §5) -----------------------------------------
NOT_SILENT_MIN_DBFS = -70.0     # rendered EGDB must exceed this RMS
PASS_THROUGH_MIN_ESR = 0.001    # ESR(input, output) must exceed this (catch no-ops)

# --- Source labels (spec §3.2.6) -----------------------------------------------
SOURCE_EGDB = "egdb"
SOURCE_SWEEP = "sweep"

# --- Split labels (spec §3.2.5) ------------------------------------------------
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

# --- Render status (spec §4.3) -------------------------------------------------
RENDER_OK = "render_ok"
RENDER_FAILED = "render_failed"

# --- Manifest / output file names ----------------------------------------------
CORPUS_MANIFEST = "corpus.parquet"
CLIPS_MANIFEST = "clips.parquet"
RENDERS_MANIFEST = "renders.parquet"
DEVICES_FINAL_MANIFEST = "devices_final.parquet"
PHASE1_MANIFEST = "manifest.parquet"          # produced by Phase 1 finalize
RENDER_SUBSET = "render_subset.txt"           # device ids for `render run --devices @<path>`

# The NAM standardized reamp signal filename (spec §3.1.2).
NAM_SWEEP_FILENAME = "v3_0_0.wav"
