import hashlib
import random

import pytest
import requests

from openamp.acquire.auth import TokenStore
from openamp.acquire.client import (
    APIError,
    RateLimiter,
    RetryableHTTPError,
    RetryConfig,
    T3KClient,
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


# --- rate limiting + retry/backoff ------------------------------------------

def test_rate_limiter_spaces_calls():
    clock = Clock()
    # 60/min = one call per second.
    limiter = RateLimiter(rate_per_min=60, time_fn=clock.time, sleep_fn=clock.sleep)
    for _ in range(5):
        limiter.acquire()
    # First call is free; remaining 4 each wait ~1s.
    assert clock.t == pytest.approx(4.0, abs=1e-6)


def test_rate_limiter_no_wait_when_slow():
    clock = Clock()
    limiter = RateLimiter(rate_per_min=60, time_fn=clock.time, sleep_fn=clock.sleep)
    for _ in range(3):
        limiter.acquire()
        clock.t += 10.0  # caller is much slower than the cap
    assert clock.t == 30.0  # no additional sleeping imposed


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


# --- client -------------------------------------------------------------------

class FakeResp:
    def __init__(self, status=200, json_data=None, headers=None, chunks=None, text=""):
        self.status_code = status
        self._json = json_data or {}
        self.headers = headers or {}
        self._chunks = chunks or []
        self.text = text

    def json(self):
        return self._json

    def iter_content(self, chunk_size=0):
        yield from self._chunks


class FakeSession:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def request(self, method, url, params=None, headers=None, stream=False, timeout=None):
        self.calls.append((method, url, params, headers, stream))
        return self.handler(method, url, params, headers, stream)


def _client(handler, settings):
    return T3KClient(
        settings,
        token=TokenStore(access_token="AT", refresh_token="RT", expires_at=9e18),
        session=FakeSession(handler),
        limiter=RateLimiter(rate_per_min=100000),
        retry=RetryConfig(base_delay=0, jitter=0, max_attempts=5),
    )


def test_retries_on_429_then_succeeds(settings):
    seq = [FakeResp(429, headers={"Retry-After": "0"}), FakeResp(200, {"ok": 1})]
    client = _client(lambda *a: seq.pop(0), settings)
    assert client.get_json("user") == {"ok": 1}
    assert len(client.session.calls) == 2


def test_retries_on_connection_error_then_succeeds(settings):
    # Transient network failures are retryable, same as 429/5xx.
    seq = [requests.ConnectionError("reset by peer"), FakeResp(200, {"ok": 1})]

    def handler(*a):
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client = _client(handler, settings)
    assert client.get_json("user") == {"ok": 1}
    assert len(client.session.calls) == 2


def test_connection_errors_exhaust_to_api_error(settings):
    def handler(*a):
        raise requests.ConnectionError("network down")

    client = _client(handler, settings)
    with pytest.raises(APIError):
        client.get_json("user")
    assert len(client.session.calls) == 5  # max_attempts


def test_401_triggers_refresh_and_retry(settings):
    seq = [FakeResp(401), FakeResp(200, {"user": "me"})]
    client = _client(lambda *a: seq.pop(0), settings)
    flag = {"refreshed": False}
    client._refresh_token = lambda: flag.__setitem__("refreshed", True)
    assert client.get_json("user") == {"user": "me"}
    assert flag["refreshed"] is True
    assert len(client.session.calls) == 2


def test_bearer_header_present(settings):
    captured = {}

    def handler(method, url, params, headers, stream):
        captured.update(headers or {})
        return FakeResp(200, {"ok": 1})

    _client(handler, settings).get_json("user")
    assert captured["Authorization"] == "Bearer AT"


def test_non_retryable_raises(settings):
    client = _client(lambda *a: FakeResp(404, text="nope"), settings)
    with pytest.raises(APIError):
        client.get_json("tones/999")


def test_edge_block_403_is_retried(settings):
    # Vercel/WAF rate-limit blocks are 403 with a non-JSON body; retry, don't drop.
    block = FakeResp(403, headers={"Content-Type": "text/plain"},
                     text="Forbidden\narn1::123-abc")
    seq = [block, block, FakeResp(200, {"data": []})]
    client = _client(lambda *a: seq.pop(0), settings)
    assert client.get_json("models") == {"data": []}
    assert len(client.session.calls) == 3


def test_json_403_is_not_retried(settings):
    # A genuine application 403 (JSON body) is a hard error — no retry.
    client = _client(lambda *a: FakeResp(403, {"error": "forbidden"},
                                         headers={"Content-Type": "application/json"}),
                     settings)
    with pytest.raises(APIError):
        client.get_json("models")
    assert len(client.session.calls) == 1


def test_download_model_hashes(tmp_path, settings):
    chunks = [b"abc", b"def", b"ghi"]
    client = _client(lambda *a: FakeResp(200, chunks=chunks), settings)
    dest = tmp_path / "out" / "1.nam"
    sha, n = client.download_model("https://x/1.nam", dest)
    assert dest.is_file()
    assert n == 9
    assert sha == hashlib.sha256(b"abcdefghi").hexdigest()


def test_download_model_retries_interrupted_stream(tmp_path, settings):
    # A connection drop mid-stream retries with a fresh request.
    class DropResp(FakeResp):
        def iter_content(self, chunk_size=0):
            yield b"abc"
            raise requests.ConnectionError("stream reset")

    seq = [DropResp(200), FakeResp(200, chunks=[b"abcdef"])]
    client = _client(lambda *a: seq.pop(0), settings)
    dest = tmp_path / "1.nam"
    sha, n = client.download_model("https://x/1.nam", dest)
    assert n == 6
    assert sha == hashlib.sha256(b"abcdef").hexdigest()
    assert len(client.session.calls) == 2


def test_iter_search_paginates(settings):
    def handler(method, url, params, headers, stream):
        if params.get("page") == 1:
            return FakeResp(200, {"data": [{"id": 1}, {"id": 2}], "total_pages": 2, "page": 1})
        return FakeResp(200, {"data": [{"id": 3}], "total_pages": 2, "page": 2})

    client = _client(handler, settings)
    ids = [t["id"] for t in client.iter_search(gears="amp", page_size=2)]
    assert ids == [1, 2, 3]


def test_list_models_paginates(settings):
    def handler(method, url, params, headers, stream):
        if params.get("page") == 1:
            return FakeResp(200, {"data": [{"id": 10}], "total_pages": 2})
        return FakeResp(200, {"data": [{"id": 11}], "total_pages": 2})

    client = _client(handler, settings)
    models = client.list_models(5, page_size=1)
    assert [m["id"] for m in models] == [10, 11]
