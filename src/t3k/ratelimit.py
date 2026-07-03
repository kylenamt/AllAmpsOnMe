"""Client-side rate limiting and retry/backoff helpers (spec §3).

- :class:`TokenBucket` enforces ~80 req/min regardless of what the server
  tolerates.
- :func:`retry_with_backoff` runs a callable with exponential backoff + jitter,
  honoring ``Retry-After`` on 429/5xx.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


class TokenBucket:
    """Thread-safe token bucket.

    Refills continuously at ``rate_per_min / 60`` tokens per second up to
    ``capacity``. :meth:`acquire` blocks until a token is available.
    """

    def __init__(self, rate_per_min: int, capacity: int | None = None,
                 time_fn: Callable[[], float] = time.monotonic,
                 sleep_fn: Callable[[float], None] = time.sleep) -> None:
        if rate_per_min <= 0:
            raise ValueError("rate_per_min must be positive")
        self.rate_per_sec = rate_per_min / 60.0
        # Small burst capacity by default, but never more than a handful so we do
        # not blow the per-minute cap in a spike.
        self.capacity = float(capacity if capacity is not None else max(1, rate_per_min // 10))
        self._tokens = self.capacity
        self._time = time_fn
        self._sleep = sleep_fn
        self._last = self._time()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._time()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_sec)
            self._last = now

    def acquire(self, tokens: float = 1.0) -> None:
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait = deficit / self.rate_per_sec
            # Sleep outside the lock so other threads can make progress.
            self._sleep(max(wait, 0.0))


@dataclass
class RetryConfig:
    max_attempts: int = 5
    base_delay: float = 1.0       # seconds
    max_delay: float = 60.0
    jitter: float = 0.5           # +/- fraction of the computed delay


class RetryableHTTPError(Exception):
    """Signals a transient HTTP failure (429/5xx) worth retrying.

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
