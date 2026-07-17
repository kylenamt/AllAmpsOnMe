import base64
import hashlib
import os
import stat

from openamp.acquire import auth


def test_generate_pkce_valid_s256():
    verifier, challenge = auth.generate_pkce()
    assert 43 <= len(verifier) <= 128
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert challenge == expected


def test_token_store_roundtrip_and_permissions(tmp_path):
    store = auth.TokenStore(access_token="a", refresh_token="r", expires_at=9e18)
    path = tmp_path / ".openamp_tokens.json"
    store.save(path)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600
    loaded = auth.TokenStore.load(path)
    assert loaded.access_token == "a"
    assert loaded.refresh_token == "r"


def test_token_store_expiry():
    store = auth.TokenStore(access_token="a", refresh_token="r", expires_at=1000)
    assert store.is_expired(now=1000)
    assert store.is_expired(now=950)          # within skew
    assert not store.is_expired(now=800)


def test_from_token_response_computes_expiry():
    store = auth.TokenStore.from_token_response(
        {"access_token": "x", "refresh_token": "y", "expires_in": 3600}, now=0)
    assert store.expires_at == 3600
    assert store.token_type == "Bearer"


def test_build_authorize_url(settings):
    verifier, challenge = auth.generate_pkce()
    url = auth.build_authorize_url(settings, state="st8", challenge=challenge)
    assert "oauth/authorize" in url
    assert f"client_id={settings.publishable_key}" in url
    assert "code_challenge_method=S256" in url
    assert "state=st8" in url
    assert "response_type=code" in url
    # Standard Flow must omit `prompt` so the consent screen (not the tone-picker)
    # is shown.
    assert "prompt" not in url


def test_exchange_and_refresh(monkeypatch, settings):
    monkeypatch.setattr(auth, "_post_token",
                        lambda s, data: {"access_token": "AT", "refresh_token": "RT",
                                         "expires_in": 100})
    store = auth.exchange_code(settings, code="c", verifier="v")
    assert store.access_token == "AT"

    # Refresh without a rotated refresh token keeps the old one.
    monkeypatch.setattr(auth, "_post_token",
                        lambda s, data: {"access_token": "AT2", "expires_in": 100})
    store2 = auth.refresh_tokens(settings, refresh_token="RT")
    assert store2.access_token == "AT2"
    assert store2.refresh_token == "RT"


def test_parse_pasted_redirect():
    result = auth._parse_pasted_redirect("http://localhost:3001/?code=abc&state=xyz")
    assert result["code"] == "abc"
    assert result["state"] == "xyz"


def test_load_or_refresh_requires_tokens(settings):
    try:
        auth.load_or_refresh(settings)
        assert False, "expected AuthError"
    except auth.AuthError:
        pass
