"""Rate-limited, authenticated HTTP client + typed endpoint wrappers (spec §3).

Every request goes through the :class:`~t3k.ratelimit.TokenBucket` and the
retry/backoff helper. 401s trigger a single token refresh + retry; 429/5xx are
retried with exponential backoff honoring ``Retry-After``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

import requests

from . import auth
from .config import Settings
from .ratelimit import (
    RetryableHTTPError,
    RetryConfig,
    TokenBucket,
    retry_with_backoff,
)

# Statuses we retry with backoff.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _parse_retry_after(resp: requests.Response) -> float | None:
    value = resp.headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        # HTTP-date form is uncommon here; fall back to no explicit hint.
        return None


class APIError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class T3KClient:
    def __init__(self, settings: Settings, token: auth.TokenStore | None = None,
                 session: requests.Session | None = None,
                 bucket: TokenBucket | None = None,
                 retry: RetryConfig | None = None) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self.bucket = bucket or TokenBucket(settings.rate_limit_rpm)
        self.retry = retry or RetryConfig()
        self._token = token  # may be None until first authed call

    # --- low-level ------------------------------------------------------------
    def _token_header(self) -> dict[str, str]:
        if self._token is None:
            self._token = auth.load_or_refresh(self.settings)
        return {"Authorization": f"{self._token.token_type} {self._token.access_token}"}

    def _refresh_token(self) -> None:
        if self._token and self._token.refresh_token:
            self._token = auth.refresh_tokens(self.settings, self._token.refresh_token)
            self._token.save(self.settings.token_path)
        else:
            self._token = auth.load_or_refresh(self.settings)

    def _full_url(self, path_or_url: str) -> str:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return path_or_url
        return f"{self.settings.base_url}/{path_or_url.lstrip('/')}"

    def request(self, method: str, path_or_url: str, *, params: dict | None = None,
                authed: bool = True, stream: bool = False,
                timeout: float = 60.0) -> requests.Response:
        url = self._full_url(path_or_url)
        state = {"refreshed": False}

        def do() -> requests.Response:
            self.bucket.acquire()
            headers = self._token_header() if authed else {}
            resp = self.session.request(method, url, params=params, headers=headers,
                                        stream=stream, timeout=timeout)
            if resp.status_code == 401 and authed and not state["refreshed"]:
                state["refreshed"] = True
                self._refresh_token()
                raise RetryableHTTPError(401, retry_after=0.0, message="token refreshed")
            if resp.status_code in _RETRYABLE_STATUS:
                raise RetryableHTTPError(resp.status_code, _parse_retry_after(resp))
            if resp.status_code >= 400:
                raise APIError(resp.status_code, resp.text[:300])
            return resp

        return retry_with_backoff(do, self.retry)

    def get_json(self, path: str, params: dict | None = None) -> dict:
        return self.request("GET", path, params=params).json()

    # --- typed endpoints ------------------------------------------------------
    def get_user(self) -> dict:
        """GET /user — the authenticated operator's profile (spec §5.1)."""
        return self.get_json("user")

    def search_tones(self, *, query: str | None = None, gears: str | None = None,
                     architecture: str | int | None = None, sort: str | None = None,
                     page: int = 1, page_size: int = 50, **extra) -> dict:
        """One page of GET /tones/search (a PaginatedResponse[Tone])."""
        params: dict[str, object] = {"page": page, "page_size": page_size}
        if query:
            params["query"] = query
        if gears:
            params["gears"] = gears
        if architecture is not None:
            params["architecture"] = architecture
        if sort:
            params["sort"] = sort
        params.update({k: v for k, v in extra.items() if v is not None})
        return self.get_json("tones/search", params)

    def iter_search(self, *, max_pages: int = 40, page_size: int = 50, **filters) -> Iterator[dict]:
        """Yield individual Tone records across all pages (up to ``max_pages``)."""
        page = 1
        while page <= max_pages:
            payload = self.search_tones(page=page, page_size=page_size, **filters)
            rows = payload.get("data") or []
            for row in rows:
                yield row
            total_pages = int(payload.get("total_pages") or 0)
            if not rows or (total_pages and page >= total_pages):
                break
            page += 1

    def get_tone(self, tone_id: int | str) -> dict:
        """GET /tones/{id}."""
        return self.get_json(f"tones/{tone_id}")

    def list_models(self, tone_id: int | str, *, architecture: str | int | None = None,
                    page_size: int = 100, max_pages: int = 20) -> list[dict]:
        """GET /models?tone_id=... — auto-paginated list of Model records."""
        out: list[dict] = []
        page = 1
        while page <= max_pages:
            params: dict[str, object] = {"tone_id": tone_id, "page": page, "page_size": page_size}
            if architecture is not None:
                params["architecture"] = architecture
            payload = self.get_json("models", params)
            rows = payload.get("data") or []
            out.extend(rows)
            total_pages = int(payload.get("total_pages") or 0)
            if not rows or (total_pages and page >= total_pages):
                break
            page += 1
        return out

    def get_model(self, model_id: int | str) -> dict:
        """GET /models/{id}."""
        return self.get_json(f"models/{model_id}")

    def download_model(self, model_url: str, dest: Path,
                       chunk_size: int = 1 << 16) -> tuple[str, int]:
        """Download a model file with Bearer auth. Returns (sha256_hex, n_bytes).

        Writes atomically via a ``.part`` file so an interrupted download never
        leaves a truncated ``.nam`` behind.
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        sha = hashlib.sha256()
        n = 0
        resp = self.request("GET", model_url, stream=True)
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                fh.write(chunk)
                sha.update(chunk)
                n += len(chunk)
        tmp.replace(dest)
        return sha.hexdigest(), n
