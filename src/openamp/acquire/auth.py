"""OAuth 2.0 + PKCE "Standard Flow" against the TONE3000 API (spec §3, §5.1).

Ported from the reference TypeScript client (github.com/tone-3000/api):

  authorize:  GET  /oauth/authorize  ?client_id&redirect_uri&response_type=code
                                       &code_challenge&code_challenge_method=S256&state
  token:      POST /oauth/token       grant_type=authorization_code | refresh_token

Tokens are persisted to ``OPENAMP_TOKEN_PATH`` with mode 0600 and auto-refreshed
on expiry / 401 by :mod:`openamp.acquire.client`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import requests

from openamp.core.config import Config

# Expiry safety margin: refresh if fewer than this many seconds remain.
EXPIRY_SKEW_SECONDS = 60


class AuthError(RuntimeError):
    pass


# --- PKCE -----------------------------------------------------------------------
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_pkce() -> tuple[str, str]:
    """Return ``(verifier, challenge)`` for the S256 method.

    Verifier is a 43–128 char URL-safe string; challenge is base64url(sha256).
    """
    verifier = _b64url(secrets.token_bytes(64))  # 86 chars, within [43, 128]
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


# --- Token storage --------------------------------------------------------------
@dataclass
class TokenStore:
    access_token: str
    refresh_token: str | None
    expires_at: float           # epoch seconds
    token_type: str = "Bearer"

    @classmethod
    def from_token_response(cls, payload: dict, now: float | None = None) -> "TokenStore":
        now = time.time() if now is None else now
        expires_in = float(payload.get("expires_in", 3600))
        return cls(
            access_token=payload["access_token"],
            refresh_token=payload.get("refresh_token"),
            expires_at=now + expires_in,
            token_type=payload.get("token_type", "Bearer"),
        )

    def is_expired(self, skew: float = EXPIRY_SKEW_SECONDS, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return now >= (self.expires_at - skew)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(path)
        os.chmod(path, 0o600)

    @classmethod
    def load(cls, path: Path) -> "TokenStore | None":
        path = Path(path)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token"),
                expires_at=float(data["expires_at"]),
                token_type=data.get("token_type", "Bearer"),
            )
        except (OSError, ValueError, KeyError) as exc:
            raise AuthError(f"Corrupt token file {path}: {exc}") from exc


# --- URL / endpoint helpers -----------------------------------------------------
def build_authorize_url(settings: Config, state: str, challenge: str) -> str:
    # The "Standard Flow" (full programmatic access) omits ``prompt`` entirely and
    # shows a consent/authorize screen (``prompt=select_tone`` would instead trigger
    # the tone-picker flow, which has nothing to confirm).
    params = {
        "client_id": settings.publishable_key,
        "redirect_uri": settings.redirect_uri,
        "response_type": "code",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return f"{settings.base_url}/oauth/authorize?{urllib.parse.urlencode(params)}"


def _post_token(settings: Config, data: dict) -> dict:
    resp = requests.post(
        f"{settings.base_url}/oauth/token",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if resp.status_code >= 400:
        raise AuthError(f"token endpoint {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def exchange_code(settings: Config, code: str, verifier: str) -> TokenStore:
    payload = _post_token(settings, {
        "grant_type": "authorization_code",
        "code": code,
        "code_verifier": verifier,
        "redirect_uri": settings.redirect_uri,
        "client_id": settings.publishable_key,
    })
    return TokenStore.from_token_response(payload)


def refresh_tokens(settings: Config, refresh_token: str) -> TokenStore:
    payload = _post_token(settings, {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": settings.publishable_key,
    })
    store = TokenStore.from_token_response(payload)
    # Some providers omit a rotated refresh token; keep the old one if so.
    if store.refresh_token is None:
        store.refresh_token = refresh_token
    return store


# --- Local redirect listener ----------------------------------------------------
class _CallbackHandler(BaseHTTPRequestHandler):
    result: dict = {}

    def do_GET(self):  # noqa: N802 (http.server API)
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        type(self).result = {k: v[0] for k, v in qs.items()}
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        ok = "code" in type(self).result and "error" not in type(self).result
        body = (
            "<h2>TONE3000 authorization "
            + ("complete" if ok else "failed")
            + "</h2><p>You can close this tab and return to the terminal.</p>"
        )
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, *args):  # silence the default stderr logging
        return


def _redirect_port(redirect_uri: str) -> int:
    port = urllib.parse.urlparse(redirect_uri).port
    if not port:
        raise AuthError(f"redirect_uri {redirect_uri!r} must include a port")
    return port


def _listen_for_code(port: int, timeout: float = 300.0) -> dict:
    _CallbackHandler.result = {}
    server = HTTPServer(("127.0.0.1", port), _CallbackHandler)
    server.timeout = timeout
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    deadline = time.time() + timeout
    while thread.is_alive() and time.time() < deadline:
        thread.join(timeout=0.2)
    server.server_close()
    if not _CallbackHandler.result:
        raise AuthError("Timed out waiting for the OAuth redirect.")
    return _CallbackHandler.result


def _parse_pasted_redirect(url: str) -> dict:
    qs = urllib.parse.urlparse(url.strip()).query
    return {k: v[0] for k, v in urllib.parse.parse_qs(qs).items()}


# --- Top-level flow -------------------------------------------------------------
def run_standard_flow(settings: Config, *, headless: bool = False,
                      open_browser: bool = True) -> TokenStore:
    """Run the full standard flow and persist tokens. Returns the store."""
    if not settings.publishable_key:
        raise AuthError("T3K_PUBLISHABLE_KEY is required for authentication.")

    verifier, challenge = generate_pkce()
    state = secrets.token_urlsafe(16)
    authorize_url = build_authorize_url(settings, state, challenge)

    if headless:
        print("\nOpen this URL in a browser (on any machine) and authorize:\n")
        print(authorize_url + "\n")
        pasted = input("Paste the full redirect URL you were sent to: ")
        result = _parse_pasted_redirect(pasted)
    else:
        print("\nOpening your browser to authorize TONE3000 access...")
        print("If it does not open, visit:\n" + authorize_url + "\n")
        if open_browser:
            try:
                webbrowser.open(authorize_url)
            except Exception:  # pragma: no cover - environment dependent
                pass
        result = _listen_for_code(_redirect_port(settings.redirect_uri))

    if result.get("error"):
        raise AuthError(f"Authorization denied: {result.get('error_description', result['error'])}")
    if result.get("state") and result["state"] != state:
        raise AuthError("OAuth state mismatch — possible CSRF, aborting.")
    code = result.get("code")
    if not code:
        raise AuthError("No authorization code returned.")

    store = exchange_code(settings, code, verifier)
    store.save(settings.token_path)
    return store


def load_or_refresh(settings: Config) -> TokenStore:
    """Load persisted tokens, refreshing if expired. Raises if no tokens exist."""
    store = TokenStore.load(settings.token_path)
    if store is None:
        raise AuthError("Not authenticated. Run `openamp auth` first.")
    if store.is_expired():
        if not store.refresh_token:
            raise AuthError("Access token expired and no refresh token. Re-run `openamp auth`.")
        store = refresh_tokens(settings, store.refresh_token)
        store.save(settings.token_path)
    return store
