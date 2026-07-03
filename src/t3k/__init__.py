"""t3k — Phase 1 TONE3000 amp-capture acquisition package.

See ``docs/phase1-spec.md`` (Phase 1 spec) for the full requirements. The public
surface used by the CLI lives in the per-stage modules:

    auth, client, discover, select, download, validate, dedup, finalize

plus supporting modules: config, constants, ratelimit, manifest, normalize, probe.
"""

__version__ = "0.1.0"
