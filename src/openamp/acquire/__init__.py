"""TONE3000 acquisition — everything specific to the TONE3000 API.

Stages, in run order: auth, discover, select, download, validate, dedup,
finalize. Supporting modules: client (HTTP + rate limit + retry), catalog
(search terms / gain buckets / make aliases), normalize (API payload ->
candidate rows).
"""
