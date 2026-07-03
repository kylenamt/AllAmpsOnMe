import random

import pytest

from t3k.ratelimit import (
    RetryableHTTPError,
    RetryConfig,
    TokenBucket,
    compute_delay,
    retry_with_backoff,
)


class Clock:
    def __init__(self):
        self.t = 0.0

    def time(self):
        return self.t

    def sleep(self, s):
        assert s >= 0
        self.t += s


def test_token_bucket_throttles_to_rate():
    clock = Clock()
    # 60/min = 1 token/sec, burst capacity 1.
    bucket = TokenBucket(rate_per_min=60, capacity=1, time_fn=clock.time, sleep_fn=clock.sleep)
    for _ in range(5):
        bucket.acquire()
    # First token is free; remaining 4 each wait ~1s.
    assert clock.t == pytest.approx(4.0, abs=1e-6)


def test_token_bucket_allows_burst_up_to_capacity():
    clock = Clock()
    bucket = TokenBucket(rate_per_min=60, capacity=3, time_fn=clock.time, sleep_fn=clock.sleep)
    for _ in range(3):
        bucket.acquire()
    assert clock.t == 0.0  # all within burst, no sleeping


def test_compute_delay_prefers_retry_after():
    cfg = RetryConfig(base_delay=1, max_delay=30)
    rng = random.Random(0)
    assert compute_delay(3, cfg, retry_after=5.0, rng=rng) == 5.0
    assert compute_delay(3, cfg, retry_after=999.0, rng=rng) == 30.0  # capped


def test_compute_delay_exponential_with_jitter_bounds():
    cfg = RetryConfig(base_delay=1, max_delay=100, jitter=0.5)
    rng = random.Random(0)
    for attempt in range(1, 5):
        d = compute_delay(attempt, cfg, retry_after=None, rng=rng)
        base = min(1 * 2 ** (attempt - 1), 100)
        assert base * 0.5 - 1e-9 <= d <= base * 1.5 + 1e-9


def test_retry_with_backoff_succeeds_after_transient():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableHTTPError(503)
        return "ok"

    out = retry_with_backoff(fn, RetryConfig(base_delay=0, jitter=0), sleep_fn=lambda s: None)
    assert out == "ok"
    assert calls["n"] == 3


def test_retry_with_backoff_exhausts():
    def fn():
        raise RetryableHTTPError(429, retry_after=0)

    with pytest.raises(RetryableHTTPError):
        retry_with_backoff(fn, RetryConfig(max_attempts=3, base_delay=0), sleep_fn=lambda s: None)
