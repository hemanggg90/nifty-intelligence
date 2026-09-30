"""
Thin REST client for the Dhan v2 API.

Centralizes auth headers, base URL, and error handling for every Dhan
endpoint the system uses. Deliberately dumb: no retries, no caching beyond a
simple in-process throttle for the option-chain endpoint (documented by Dhan
as one unique request per 3 seconds). Every method raises `DhanApiError` on
a non-2xx response or malformed body - callers decide how to degrade.

This client makes real network calls. It is never imported by anything in
`quant_intelligence/tests/` in a way that hits the network - tests mock
`requests.post`/`requests.get` and assert the constructed request.
"""
from __future__ import annotations

import threading
import time

import requests

from quant_intelligence.config.settings import SETTINGS

_OPTION_CHAIN_MIN_INTERVAL_SEC = 3.5  # Dhan's documented limit is 1 unique request / 3 s; keep a margin
# Dhan answers 429 (code 805, "may result in the user being blocked") when the account exceeds
# that, including requests from any other process using the same account. Back off and retry.
_RATE_LIMIT_MAX_ATTEMPTS = 3
_RATE_LIMIT_BACKOFF_SEC = 6.0
# Expiry dates change at most daily, so cache them and halve the option-chain request rate.
_EXPIRY_CACHE_TTL_SEC = 30 * 60
_expiry_cache: dict[tuple, tuple[float, list[str]]] = {}
# Process-wide, not per client: the NSE and commodity runners each build their own client, and
# Dhan's limit applies to the account, so the spacing must be shared between them.
_option_chain_lock = threading.Lock()
_last_option_chain_call = 0.0


class DhanApiError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def clear_expiry_cache() -> None:
    _expiry_cache.clear()


class DhanApiClient:
    def __init__(self, client_id: str | None = None, access_token: str | None = None, base_url: str | None = None):
        self.client_id = client_id or SETTINGS.dhan_client_id
        self.access_token = access_token or SETTINGS.dhan_access_token
        self.base_url = (base_url or SETTINGS.dhan_base_url).rstrip("/")

    def is_configured(self) -> bool:
        return bool(self.client_id and self.access_token)

    def _headers(self) -> dict:
        return {
            "access-token": self.access_token,
            "client-id": self.client_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _require_configured(self) -> None:
        if not self.is_configured():
            raise DhanApiError("DhanApiClient is not configured: set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in .env")

    def _get(self, path: str) -> dict:
        self._require_configured()
        resp = requests.get(f"{self.base_url}{path}", headers=self._headers(), timeout=15)
        return self._handle(resp)

    def _post(self, path: str, body: dict) -> dict:
        self._require_configured()
        resp = requests.post(f"{self.base_url}{path}", headers=self._headers(), json=body, timeout=15)
        return self._handle(resp)

    def _handle(self, resp: "requests.Response") -> dict:
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if not resp.ok:
            message = data.get("remarks") or data.get("message") or resp.text or f"HTTP {resp.status_code}"
            raise DhanApiError(f"Dhan API error ({resp.status_code}): {message}", status_code=resp.status_code)
        return data

    def _throttle_option_chain(self) -> None:
        global _last_option_chain_call
        with _option_chain_lock:  # concurrent callers queue up, each spaced 3 s from the last
            elapsed = time.monotonic() - _last_option_chain_call
            if elapsed < _OPTION_CHAIN_MIN_INTERVAL_SEC:
                time.sleep(_OPTION_CHAIN_MIN_INTERVAL_SEC - elapsed)
            _last_option_chain_call = time.monotonic()

    def _post_option_chain(self, path: str, body: dict) -> dict:
        """POST to an option-chain endpoint: shared throttle, and back off + retry on HTTP 429."""
        for attempt in range(1, _RATE_LIMIT_MAX_ATTEMPTS + 1):
            self._throttle_option_chain()
            try:
                return self._post(path, body)
            except DhanApiError as e:
                if e.status_code != 429 or attempt == _RATE_LIMIT_MAX_ATTEMPTS:
                    raise
                time.sleep(_RATE_LIMIT_BACKOFF_SEC * attempt)
        raise AssertionError("unreachable")

    def get_expiry_list(self, underlying_scrip: int, underlying_seg: str) -> list[str]:
        key = (self.base_url, int(underlying_scrip), underlying_seg)
        cached = _expiry_cache.get(key)
        if cached and time.monotonic() - cached[0] < _EXPIRY_CACHE_TTL_SEC:
            return list(cached[1])
        # UnderlyingScrip must be an integer: the scrip master yields string ids, which Dhan rejects (814).
        body = self._post_option_chain(
            "/optionchain/expirylist", {"UnderlyingScrip": int(underlying_scrip), "UnderlyingSeg": underlying_seg}
        )
        expiries = body.get("data", [])
        if expiries:  # never cache an empty answer (e.g. a future with no live options yet)
            _expiry_cache[key] = (time.monotonic(), list(expiries))
        return expiries

    def get_option_chain(self, underlying_scrip: int, underlying_seg: str, expiry: str) -> dict:
        return self._post_option_chain(
            "/optionchain",
            {"UnderlyingScrip": int(underlying_scrip), "UnderlyingSeg": underlying_seg, "Expiry": expiry},
        )

    def get_ltp(self, segment_map: dict[str, list[int]]) -> dict:
        return self._post("/marketfeed/ltp", segment_map)

    def get_fund_limit(self) -> dict:
        return self._get("/fundlimit")

    def get_positions(self) -> list[dict]:
        data = self._get("/positions")
        return data if isinstance(data, list) else data.get("data", [])

    def place_order(self, payload: dict) -> dict:
        return self._post("/orders", payload)

    def get_historical_daily(
        self, security_id: str, exchange_segment: str, instrument: str, from_date: str, to_date: str
    ) -> dict:
        """Daily OHLCV candles. `from_date`/`to_date` as 'YYYY-MM-DD'."""
        return self._post(
            "/charts/historical",
            {
                "securityId": security_id,
                "exchangeSegment": exchange_segment,
                "instrument": instrument,
                "expiryCode": 0,
                "oi": False,
                "fromDate": from_date,
                "toDate": to_date,
            },
        )

    def get_intraday_minute(
        self,
        security_id: str,
        exchange_segment: str,
        instrument: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> dict:
        """Intraday OHLCV candles. `interval` is one of '1'/'5'/'15'/'25'/'60' (minutes).
        `from_date`/`to_date` as 'YYYY-MM-DD HH:MM:SS'. Dhan caps this at 90 days per call."""
        return self._post(
            "/charts/intraday",
            {
                "securityId": security_id,
                "exchangeSegment": exchange_segment,
                "instrument": instrument,
                "interval": interval,
                "oi": False,
                "fromDate": from_date,
                "toDate": to_date,
            },
        )
