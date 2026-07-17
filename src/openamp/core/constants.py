"""Global constants for the pipeline (spec §2, §3).

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
RENDER_MIN_RMS_DBFS = -70.0     # rendered EGDB must exceed this RMS
PASS_THROUGH_MIN_ESR = 0.001    # ESR(input, output) must exceed this (catch no-ops)

# --- Source labels (spec §3.2.6) -----------------------------------------------
SOURCE_EGDB = "egdb"
SOURCE_SWEEP = "sweep"

# --- Split labels (spec §3.2.5) ------------------------------------------------
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

# --- Gain-style buckets (spec §5.3.3) ------------------------------------------
# Pipeline-wide vocabulary: the ``gain_bucket`` value on every capture/device.
# Shared by acquisition selection and the render subset, so it lives here rather
# than in the acquisition-only catalog. (The keyword heuristic that assigns these
# stays in ``openamp.acquire.catalog``.)
GAIN_CLEAN = "clean"
GAIN_CRUNCH = "crunch"          # crunch / mid-gain
GAIN_HIGH = "high_gain"
GAIN_UNKNOWN = "unknown"
GAIN_BUCKETS = (GAIN_CLEAN, GAIN_CRUNCH, GAIN_HIGH)   # the "known" (non-unknown) set

# --- Render status (spec §4.3) -------------------------------------------------
RENDER_OK = "render_ok"
RENDER_FAILED = "render_failed"

# --- Manifest / output file names ----------------------------------------------
CORPUS_MANIFEST = "corpus.parquet"
CLIPS_MANIFEST = "clips.parquet"
RENDERS_MANIFEST = "renders.parquet"
DEVICES_FINAL_MANIFEST = "devices_final.parquet"
ACQUISITION_MANIFEST = "manifest.parquet"     # produced by `openamp finalize`
RENDER_SUBSET = "render_subset.txt"           # device ids for `render run --devices @<path>`
EMULATE_HOLDOUT = "emulate_holdout.txt"       # device ids held out of emulate training (Phase 5 enrollment)

# The NAM standardized reamp signal filename (spec §3.1.2).
NAM_SWEEP_FILENAME = "v3_0_0.wav"
