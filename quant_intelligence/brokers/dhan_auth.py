"""Token keeper: keeps the Dhan access token valid without anyone pasting a new one.

Dhan access tokens last 24 hours. Two ways to get the next one without the Dhan website:

* RENEW (needs nothing extra): `RenewToken` swaps a still-active token for a fresh 24h one. The keeper does
  this when less than TOKEN_RENEW_BEFORE_HOURS (default 8) remain, so one working token keeps itself alive.
* GENERATE (optional): with DHAN_PIN and DHAN_TOTP_SECRET set, a new token is generated from a time-based
  one-time code (RFC 6238, computed here with the standard library). This also recovers from a token that
  has already expired. The PIN and TOTP secret are as sensitive as a password: keep them in Streamlit
  Secrets or .env, never in git, and never paste them in chat.

Safety: one attempt at a time (single-flight), at least 10 minutes between failed attempts (a wrong PIN
must not lock the account), and secrets/codes never reach logs or error messages. The new token is applied
through `config.credentials.update_dhan_credentials` (saved to .env where the filesystem allows it, always
applied in memory).
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import os
import struct
import threading
import time

import requests

from quant_intelligence.config.credentials import token_status, update_dhan_credentials
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.utils.logging_utils import log_event

RENEW_URL = "https://api.dhan.co/v2/RenewToken"
GENERATE_URL = "https://auth.dhan.co/app/generateAccessToken"
RETRY_AFTER_FAILURE_SEC = 600.0  # never hammer auth: a wrong PIN repeated could lock the account
_TIMEOUT = 15


def totp_now(secret: str, at: float | None = None, digits: int = 6, step: int = 30) -> str:
    """RFC 6238 time-based one-time password (HMAC-SHA1) from a base32 secret."""
    clean = secret.replace(" ", "").replace("-", "").upper()
    key = base64.b32decode(clean + "=" * (-len(clean) % 8))
    counter = int((time.time() if at is None else at) // step)
    digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code = (struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return str(code).zfill(digits)


def _json(resp):
    try:
        return resp.json()
    except ValueError:
        return None


def _extract_token(payload) -> str | None:
    if isinstance(payload, dict):
        for key in ("accessToken", "access_token", "token"):
            value = payload.get(key)
            if isinstance(value, str) and value.count(".") >= 2:
                return value
        data = payload.get("data")
        if isinstance(data, dict):
            return _extract_token(data)
    return None


def _apply(client_id: str, token: str) -> None:
    """Make `token` the live credential. Saving to .env is best effort (read-only on some hosts)."""
    try:
        update_dhan_credentials(client_id, token)
    except OSError:
        os.environ["DHAN_ACCESS_TOKEN"] = token
        object.__setattr__(SETTINGS, "dhan_access_token", token)
        object.__setattr__(SETTINGS, "dhan_client_id", client_id)


class TokenKeeper:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.last_action: str | None = None  # "renewed" | "generated"
        self.last_success_at: dt.datetime | None = None
        self.last_error: str | None = None
        self._next_attempt = 0.0  # monotonic

    @staticmethod
    def totp_configured() -> bool:
        return bool(SETTINGS.dhan_pin and SETTINGS.dhan_totp_secret)

    # ---- the two Dhan calls -------------------------------------------------------
    def _renew(self) -> str:
        try:
            resp = requests.get(RENEW_URL, headers={"access-token": SETTINGS.dhan_access_token,
                                                    "dhanClientId": SETTINGS.dhan_client_id}, timeout=_TIMEOUT)
        except requests.RequestException as e:
            raise RuntimeError(f"renew: could not reach Dhan ({type(e).__name__})") from None
        token = _extract_token(_json(resp)) if resp.ok else None
        if not token:
            raise RuntimeError(f"renew: Dhan answered HTTP {resp.status_code} without a token")
        return token

    def _generate(self) -> str:
        if not self.totp_configured():
            raise RuntimeError("generate: DHAN_PIN and DHAN_TOTP_SECRET are not set")
        try:
            code = totp_now(SETTINGS.dhan_totp_secret)
        except Exception:
            raise RuntimeError("generate: DHAN_TOTP_SECRET is not a valid base32 secret") from None
        try:
            resp = requests.post(GENERATE_URL, params={"dhanClientId": SETTINGS.dhan_client_id,
                                                       "pin": SETTINGS.dhan_pin, "totp": code}, timeout=_TIMEOUT)
        except requests.RequestException as e:  # message omitted on purpose: it would contain the URL with the PIN
            raise RuntimeError(f"generate: could not reach Dhan ({type(e).__name__})") from None
        token = _extract_token(_json(resp)) if resp.ok else None
        if not token:
            raise RuntimeError(f"generate: Dhan answered HTTP {resp.status_code} without a token (check PIN / TOTP secret)")
        return token

    # ---- policy -------------------------------------------------------------------
    def ensure_fresh(self, force: bool = False) -> str:
        """Renew or regenerate the token if it is due. Returns what happened:
        "ok" (nothing needed) | "renewed" | "generated" | "failed" | "manual" (expired, no TOTP set up) |
        "waiting" (a recent attempt failed) | "disabled" | "busy" (another thread is already doing it)."""
        if not SETTINGS.token_keeper_enabled:
            return "disabled"
        if not SETTINGS.dhan_client_id:
            return "ok"
        status = token_status(SETTINGS.dhan_access_token)
        state = status["state"]
        if state == "unknown":
            return "ok"  # not a JWT with an expiry: cannot judge, leave it alone
        due = state in ("missing", "expired") or status["delta"] < dt.timedelta(hours=SETTINGS.token_renew_before_hours)
        if not due and not force:
            return "ok"
        if state in ("missing", "expired") and not self.totp_configured():
            self.last_error = "Token expired and DHAN_PIN / DHAN_TOTP_SECRET are not set: enter a new token under API Keys."
            return "manual"
        if not self._lock.acquire(blocking=False):
            return "busy"
        try:
            if time.monotonic() < self._next_attempt and not force:
                return "waiting"
            attempts = []
            if state in ("expiring", "valid"):
                attempts.append(("renewed", self._renew))  # only an active token can be renewed
            if self.totp_configured():
                attempts.append(("generated", self._generate))
            errors = []
            for label, fn in attempts:
                try:
                    token = fn()
                except RuntimeError as e:
                    errors.append(str(e))
                    continue
                _apply(SETTINGS.dhan_client_id, token)
                self.last_action, self.last_success_at, self.last_error = label, dt.datetime.now(), None
                self._next_attempt = 0.0
                expires = token_status(token)["expires_at"]
                log_event("token_keeper", f"Dhan access token {label}"
                                          + (f" (valid until {expires:%d %b %H:%M} IST)" if expires else ""), level="INFO")
                return label
            self.last_error = "; ".join(errors) or "no way to renew the token"
            self._next_attempt = time.monotonic() + RETRY_AFTER_FAILURE_SEC
            log_event("token_keeper", f"Token refresh failed: {self.last_error}", level="WARNING")
            return "failed"
        finally:
            self._lock.release()

    def status(self) -> dict:
        wait = max(0.0, self._next_attempt - time.monotonic())
        return {"enabled": SETTINGS.token_keeper_enabled, "totp": self.totp_configured(),
                "last_action": self.last_action, "last_success_at": self.last_success_at,
                "last_error": self.last_error, "retry_in": wait}


TOKEN_KEEPER = TokenKeeper()
