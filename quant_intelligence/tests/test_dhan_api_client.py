import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from quant_intelligence.brokers.dhan_api_client import DhanApiClient, DhanApiError


def _settings_with_no_credentials():
    from quant_intelligence.brokers.dhan_api_client import SETTINGS

    return dataclasses.replace(SETTINGS, dhan_client_id="", dhan_access_token="")


def _client():
    return DhanApiClient(client_id="CID123", access_token="TOKEN123", base_url="https://api.dhan.co/v2")


def _mock_response(json_body, ok=True, status_code=200, text=""):
    resp = MagicMock()
    resp.ok = ok
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.text = text
    return resp


def test_is_configured_requires_both_credentials(monkeypatch):
    assert _client().is_configured()
    # Empty constructor args fall back to SETTINGS (so real .env credentials, if any,
    # can configure a no-args client) - swap SETTINGS for a credential-less copy so
    # this asserts the "truly no credentials anywhere" case regardless of .env contents.
    monkeypatch.setattr("quant_intelligence.brokers.dhan_api_client.SETTINGS", _settings_with_no_credentials())
    assert not DhanApiClient(client_id="", access_token="", base_url="x").is_configured()


def test_get_option_chain_sends_correct_payload_and_headers():
    client = _client()
    with patch("quant_intelligence.brokers.dhan_api_client.requests.post") as mock_post:
        mock_post.return_value = _mock_response({"data": {"last_price": 100.0, "oc": {}}})
        result = client.get_option_chain(13, "IDX_I", "2024-10-31")

    args, kwargs = mock_post.call_args
    assert args[0] == "https://api.dhan.co/v2/optionchain"
    assert kwargs["headers"]["access-token"] == "TOKEN123"
    assert kwargs["headers"]["client-id"] == "CID123"
    assert kwargs["json"] == {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I", "Expiry": "2024-10-31"}
    assert result["data"]["last_price"] == 100.0


def test_place_order_raises_on_non_ok_response():
    client = _client()
    with patch("quant_intelligence.brokers.dhan_api_client.requests.post") as mock_post:
        mock_post.return_value = _mock_response({"remarks": "Insufficient funds"}, ok=False, status_code=400)
        with pytest.raises(DhanApiError):
            client.place_order({"foo": "bar"})


def test_get_fund_limit_uses_get_with_headers():
    client = _client()
    with patch("quant_intelligence.brokers.dhan_api_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response({"availabelBalance": 50000.0})
        result = client.get_fund_limit()

    args, kwargs = mock_get.call_args
    assert args[0] == "https://api.dhan.co/v2/fundlimit"
    assert kwargs["headers"]["access-token"] == "TOKEN123"
    assert result["availabelBalance"] == 50000.0


def test_unconfigured_client_raises_before_network_call(monkeypatch):
    monkeypatch.setattr("quant_intelligence.brokers.dhan_api_client.SETTINGS", _settings_with_no_credentials())
    client = DhanApiClient(client_id="", access_token="", base_url="https://api.dhan.co/v2")
    with patch("quant_intelligence.brokers.dhan_api_client.requests.get") as mock_get:
        with pytest.raises(DhanApiError):
            client.get_fund_limit()
        mock_get.assert_not_called()
