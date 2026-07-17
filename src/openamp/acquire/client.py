"""Rate-limited, authenticated TONE3000 HTTP client.

Every request is spaced by :class:`RateLimiter` and retried with exponential
backoff honoring ``Retry-After`` (429/5xx, transient network errors, and
Vercel/WAF 403 blocks). 401s trigger a single token refresh + retry.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, TypeVar

import requests

from . import auth
from openamp.core.config import Config

T = TypeVar("T")

# Statuses we retry with backoff.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Transient transport failures worth retrying alongside 429/5xx.
_NETWORK_ERRORS = (requests.ConnectionError, requests.Timeout,
                   requests.exceptions.ChunkedEncodingError)


# --------------------------------------------------------------------------
# Rate limiting + retry/backoff
# --------------------------------------------------------------------------

class RateLimiter:
    """Spaces calls at least ``60 / rate_per_min`` seconds apart.

    The pipeline is single-threaded, so a plain min-interval sleeper is all
    the rate cap needs. ``time_fn``/``sleep_fn`` are injectable for tests.
    """

    def __init__(self, rate_per_min: int,
                 time_fn: Callable[[], float] = time.monotonic,
                 sleep_fn: Callable[[float], None] = time.sleep) -> None:
        if rate_per_min <= 0:
            raise ValueError("rate_per_min must be positive")
        self.min_interval = 60.0 / rate_per_min
        self._time = time_fn
        self._sleep = sleep_fn
        self._next_ok = self._time()  # first call passes immediately

    def acquire(self) -> None:
        now = self._time()
        if now < self._next_ok:
            self._sleep(self._next_ok - now)
            now = self._next_ok
        self._next_ok = now + self.min_interval


@dataclass
class RetryConfig:
    max_attempts: int = 5
    base_delay: float = 1.0       # seconds
    max_delay: float = 60.0
    jitter: float = 0.5           # +/- fraction of the computed delay


class RetryableHTTPError(Exception):
    """Signals a transient failure (429/5xx/network) worth retrying.

    Carries an optional ``retry_after`` (seconds) parsed from the response.
    """

    def __init__(self, status: int, retry_after: float | None = None, message: str = "") -> None:
        super().__init__(message or f"retryable HTTP {status}")
        self.status = status
        self.retry_after = retry_after


def compute_delay(attempt: int, cfg: RetryConfig, retry_after: float | None,
                  rng: random.Random) -> float:
    """Exponential backoff with jitter; ``Retry-After`` takes precedence."""
    if retry_after is not None and retry_after >= 0:
        return min(retry_after, cfg.max_delay)
    exp = cfg.base_delay * (2 ** (attempt - 1))
    exp = min(exp, cfg.max_delay)
    jitter = exp * cfg.jitter
    return max(0.0, exp + rng.uniform(-jitter, jitter))


def retry_with_backoff(fn: Callable[[], T], cfg: RetryConfig | None = None,
                       sleep_fn: Callable[[float], None] = time.sleep,
                       rng: random.Random | None = None) -> T:
    """Call ``fn`` with retries on :class:`RetryableHTTPError`.

    ``fn`` should raise :class:`RetryableHTTPError` for transient failures and
    any other exception for permanent ones (which propagate immediately).
    """
    cfg = cfg or RetryConfig()
    rng = rng or random.Random()
    last_exc: RetryableHTTPError | None = None
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            return fn()
        except RetryableHTTPError as exc:
            last_exc = exc
            if attempt >= cfg.max_attempts:
                break
            sleep_fn(compute_delay(attempt, cfg, exc.retry_after, rng))
    assert last_exc is not None
    raise last_exc


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

def _is_json(resp: requests.Response) -> bool:
    """True if the response advertises a JSON body (checked via Content-Type only,
    so it is safe for streamed responses where the body isn't read yet)."""
    return "application/json" in resp.headers.get("Content-Type", "").lower()


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
    def __init__(self, settings: Config, token: auth.TokenStore | None = None,
                 session: requests.Session | None = None,
                 limiter: RateLimiter | None = None,
                 retry: RetryConfig | None = None) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self.limiter = limiter or RateLimiter(settings.rate_limit_rpm)
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
            self.limiter.acquire()
            headers = self._token_header() if authed else {}
            try:
                resp = self.session.request(method, url, params=params, headers=headers,
                                            stream=stream, timeout=timeout)
            except _NETWORK_ERRORS as exc:
                raise RetryableHTTPError(0, message=f"network error: {exc}") from exc
            if resp.status_code == 401 and authed and not state["refreshed"]:
                state["refreshed"] = True
                self._refresh_token()
                raise RetryableHTTPError(401, retry_after=0.0, message="token refreshed")
            if resp.status_code in _RETRYABLE_STATUS:
                raise RetryableHTTPError(resp.status_code, _parse_retry_after(resp))
            # Edge/WAF rate-limit blocks (TONE3000 is behind Vercel) surface as a
            # 403 with a non-JSON body — e.g. "Forbidden\narn1::...". These are
            # transient and clear on backoff, unlike a genuine application 403
            # (which is JSON). Retry the former; treat the latter as a hard error.
            if resp.status_code == 403 and not _is_json(resp):
                raise RetryableHTTPError(403, _parse_retry_after(resp),
                                         message="edge rate-limit / WAF block")
            if resp.status_code >= 400:
                raise APIError(resp.status_code, resp.text[:300])
            return resp

        try:
            return retry_with_backoff(do, self.retry)
        except RetryableHTTPError as exc:
            # Retries exhausted. Surface as the client's public error type so
            # callers (which catch APIError to skip/continue) don't crash on the
            # internal retry signal.
            raise APIError(exc.status, str(exc)) from exc

    def get_json(self, path: str, params: dict | None = None) -> dict:
        return self.request("GET", path, params=params).json()

    # --- typed endpoints ------------------------------------------------------
    def get_user(self) -> dict:
        """GET /user — the authenticated operator's profile."""
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

    def download_model(self, model_url: str, dest: Path,
                       chunk_size: int = 1 << 16) -> tuple[str, int]:
        """Download a model file with Bearer auth. Returns (sha256_hex, n_bytes).

        Writes atomically via a ``.part`` file so an interrupted download never
        leaves a truncated ``.nam`` behind. A connection drop mid-stream is
        retried with backoff (fresh request each attempt).
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")

        def do() -> tuple[str, int]:
            sha = hashlib.sha256()
            n = 0
            resp = self.request("GET", model_url, stream=True)
            try:
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        sha.update(chunk)
                        n += len(chunk)
            except _NETWORK_ERRORS as exc:
                raise RetryableHTTPError(0, message=f"stream interrupted: {exc}") from exc
            return sha.hexdigest(), n

        try:
            result = retry_with_backoff(do, self.retry)
        except RetryableHTTPError as exc:
            raise APIError(exc.status, str(exc)) from exc
        tmp.replace(dest)
        return result
