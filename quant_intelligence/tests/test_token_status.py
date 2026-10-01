"""Reading a Dhan token's expiry locally, and never sending requests with a token known to be expired."""
import base64
import datetime as dt
import json
from unittest.mock import MagicMock, patch

import pytest

from quant_intelligence.brokers.dhan_api_client import DhanApiClient, DhanApiError
from quant_intelligence.config.credentials import token_expiry, token_status

_IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def _jwt(exp: dt.datetime | None) -> str:
    """A token shaped like Dhan's: header.payload.signature, payload carrying an `exp` epoch (IST wall time given)."""
    def b64(obj) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    payload = {"iss": "dhan", "dhanClientId": "1000000001"}
    if exp is not None:
        payload["exp"] = int(exp.replace(tzinfo=_IST).timestamp())
    return f"{b64({'alg': 'HS512'})}.{b64(payload)}.signature"


NOW = dt.datetime(2026, 10, 1, 17, 0)


def test_expiry_is_read_from_the_token_in_ist():
    assert token_expiry(_jwt(dt.datetime(2026, 9, 30, 9, 5, 49))) == dt.datetime(2026, 9, 30, 9, 5, 49)


@pytest.mark.parametrize("token", ["", None, "plain-token", "a.b.c", "x.@@@.y"])
def test_unreadable_tokens_have_no_expiry_and_are_not_called_expired(token):
    assert token_expiry(token) is None
    assert token_status(token, NOW)["state"] in ("missing", "unknown")


def test_status_states():
    assert token_status(_jwt(NOW - dt.timedelta(hours=31)), NOW)["state"] == "expired"
    assert token_status(_jwt(NOW + dt.timedelta(minutes=50)), NOW)["state"] == "expiring"
    ok = token_status(_jwt(NOW + dt.timedelta(hours=20)), NOW)
    assert ok["state"] == "valid" and ok["delta"] == dt.timedelta(hours=20)
    gone = token_status(_jwt(NOW - dt.timedelta(hours=31)), NOW)
    assert gone["delta"] < dt.timedelta(0)
    assert token_status(_jwt(None), NOW)["state"] == "unknown"  # a JWT without an exp claim


def test_an_expired_token_sends_nothing_to_dhan_and_says_why():
    client = DhanApiClient("1000000001", _jwt(dt.datetime(2020, 1, 1)))
    with patch("quant_intelligence.brokers.dhan_api_client.requests.post") as post, \
            patch("quant_intelligence.brokers.dhan_api_client.requests.get") as get:
        for call in (lambda: client.get_ltp({"NSE_FNO": [1]}), client.get_fund_limit,
                     lambda: client.get_intraday_minute("1", "MCX_COMM", "FUTCOM", "5", "a", "b"),
                     lambda: client.get_option_chain(13, "IDX_I", "2026-10-06")):
            with pytest.raises(DhanApiError) as e:
                call()
            assert e.value.status_code == 401 and "expired on 01 Jan 00:00 IST" in str(e.value)
        assert post.call_count == 0 and get.call_count == 0  # not even one request


def test_a_non_expired_token_is_used_normally():
    client = DhanApiClient("id", _jwt(dt.datetime.now(_IST).replace(tzinfo=None) + dt.timedelta(hours=10)))
    resp = MagicMock(ok=True, status_code=200)
    resp.json.return_value = {"data": {}}
    resp.headers = {}
    with patch("quant_intelligence.brokers.dhan_api_client.requests.post", return_value=resp) as post:
        client.get_ltp({"NSE_FNO": [1]})
    assert post.call_count == 1


def test_test_connection_reports_acceptance_and_rejection():
    from quant_intelligence.ui import credentials_panel as panel

    with patch.object(panel, "DhanApiClient") as cls:
        cls.return_value.is_configured.return_value = True
        cls.return_value.get_fund_limit.return_value = {"availabelBalance": 123456.0}
        ok, msg = panel.test_connection()
        assert ok and "accepted" in msg and "1,23,456" in msg

        cls.return_value.get_fund_limit.side_effect = DhanApiError("Dhan API error (401): expired", status_code=401)
        ok, msg = panel.test_connection()
        assert not ok and "401" in msg

        cls.return_value.is_configured.return_value = False
        assert panel.test_connection() == (False, "No credentials are set.")
