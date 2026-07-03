"""Static lookup tables: search terms, gain-style buckets, make normalization,
and exclusion keywords. Kept in one place so selection is deterministic and the
heuristics are easy to audit/tweak.
"""

from __future__ import annotations

# --- Discovery keyword passes (spec §5.2) --------------------------------------
# ~40 brand/model terms to force brand diversity beyond ranking bias.
AMP_SEARCH_TERMS: list[str] = [
    "fender", "marshall", "vox", "mesa", "orange", "peavey", "soldano",
    "friedman", "bogner", "hiwatt", "engl", "evh", "diezel", "jcm800",
    "plexi", "twin reverb", "ac30", "rectifier", "5150", "jazz chorus",
    "tweed", "deluxe", "princeton", "bassman", "jtm45", "dsl", "sl0",
    "mark v", "supro", "matchless", "dumble", "roland", "laney", "randall",
    "carvin", "two rock", "revv", "victory", "silvertone", "gibson",
]

# --- Gain-style buckets (spec §5.3.3) ------------------------------------------
GAIN_CLEAN = "clean"
GAIN_CRUNCH = "crunch"          # crunch / mid-gain
GAIN_HIGH = "high_gain"
GAIN_UNKNOWN = "unknown"

# Ordered most-specific-first: high-gain wins over crunch wins over clean when
# multiple keywords appear (a "high gain lead" is high_gain).
GAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    GAIN_HIGH: (
        "high gain", "high-gain", "hi gain", "hi-gain", "metal", "djent",
        "brutal", "modern", "rectifier", "recto", "5150", "6505", "mark iv",
        "mark v", "diezel", "engl", "gain 8", "gain 9", "gain 10", "saturated",
        "tight", "chug", "scoop",
    ),
    GAIN_CRUNCH: (
        "crunch", "lead", "drive", "overdrive", "od", "rock", "plexi",
        "jcm", "dsl", "hot", "gain 5", "gain 6", "gain 7", "mid gain",
        "mid-gain", "rhythm", "classic rock", "blues",
    ),
    GAIN_CLEAN: (
        "clean", "edge", "edge of breakup", "sparkle", "chime", "pristine",
        "twang", "jazz", "gain 1", "gain 2", "gain 0", "low gain", "fender clean",
    ),
}

# --- Make normalization --------------------------------------------------------
# Maps common aliases / abbreviations to a canonical lowercase make token so that
# per-(make, model) caps and brand round-robin behave. Values are already lower.
MAKE_ALIASES: dict[str, str] = {
    "fender": "fender",
    "marshall": "marshall",
    "vox": "vox",
    "mesa": "mesa boogie",
    "mesa boogie": "mesa boogie",
    "mesa/boogie": "mesa boogie",
    "boogie": "mesa boogie",
    "orange": "orange",
    "peavey": "peavey",
    "soldano": "soldano",
    "friedman": "friedman",
    "bogner": "bogner",
    "hiwatt": "hiwatt",
    "engl": "engl",
    "evh": "evh",
    "diezel": "diezel",
    "supro": "supro",
    "matchless": "matchless",
    "dumble": "dumble",
    "roland": "roland",
    "laney": "laney",
    "randall": "randall",
    "carvin": "carvin",
    "two rock": "two rock",
    "two-rock": "two rock",
    "revv": "revv",
    "victory": "victory",
    "silvertone": "silvertone",
    "gibson": "gibson",
}

# Model/brand tokens frequently embedded in a title that imply a make, used when
# the tone has no explicit `makes[]` entry.
MODEL_TO_MAKE: dict[str, str] = {
    "jcm800": "marshall",
    "jcm900": "marshall",
    "plexi": "marshall",
    "jtm45": "marshall",
    "dsl": "marshall",
    "twin reverb": "fender",
    "deluxe reverb": "fender",
    "princeton": "fender",
    "bassman": "fender",
    "tweed": "fender",
    "ac30": "vox",
    "ac15": "vox",
    "rectifier": "mesa boogie",
    "dual rectifier": "mesa boogie",
    "mark v": "mesa boogie",
    "mark iv": "mesa boogie",
    "5150": "evh",
    "6505": "peavey",
    "jazz chorus": "roland",
    "jc-120": "roland",
    "slo": "soldano",
    "sl0": "soldano",
}

# --- Exclusion keywords (spec §5.2) --------------------------------------------
# If any of these appear in the tone/gear metadata (and the item isn't clearly an
# amp), the candidate is dropped at discovery time.
EXCLUDE_KEYWORDS: tuple[str, ...] = (
    "cab", "cabinet", "ir ", " ir", "impulse", "full rig", "full-rig",
    "fullrig", "pedal", "stompbox", "overdrive pedal", "fuzz", "delay",
    "reverb pedal", "outboard", "experimental capture", "bass amp",
    "bass di", "acoustic",
)

# TONE3000 `gear` values we treat as amp-only DI captures. Everything else
# (full-rig, cab, pedal, ...) is excluded for this phase.
AMP_GEAR_VALUES: frozenset[str] = frozenset({"amp", "amps", "amplifier"})

# NAM format identifiers.
NAM_FORMAT_VALUES: frozenset[str] = frozenset({"nam", "neural amp modeler", "neural-amp-modeler"})
