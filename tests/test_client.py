import hashlib

import pytest

from t3k.auth import TokenStore
from t3k.client import APIError, T3KClient
from t3k.ratelimit import RetryConfig, TokenBucket


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
        bucket=TokenBucket(rate_per_min=100000),
        retry=RetryConfig(base_delay=0, jitter=0, max_attempts=5),
    )


def test_retries_on_429_then_succeeds(settings):
    seq = [FakeResp(429, headers={"Retry-After": "0"}), FakeResp(200, {"ok": 1})]
    client = _client(lambda *a: seq.pop(0), settings)
    assert client.get_json("user") == {"ok": 1}
    assert len(client.session.calls) == 2


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


def test_download_model_hashes(tmp_path, settings):
    chunks = [b"abc", b"def", b"ghi"]
    client = _client(lambda *a: FakeResp(200, chunks=chunks), settings)
    dest = tmp_path / "out" / "1.nam"
    sha, n = client.download_model("https://x/1.nam", dest)
    assert dest.is_file()
    assert n == 9
    assert sha == hashlib.sha256(b"abcdefghi").hexdigest()


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
