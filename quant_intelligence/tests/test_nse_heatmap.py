from unittest.mock import MagicMock, patch

from quant_intelligence.data_adapters import nse_heatmap


def test_fetch_heatmap_universe_falls_back_on_network_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(nse_heatmap, "_CACHE_FILE", tmp_path / "nse_universe.json")

    with patch("quant_intelligence.data_adapters.nse_heatmap.requests.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session.get.side_effect = RuntimeError("simulated network failure")
        mock_session_cls.return_value = mock_session

        result = nse_heatmap.fetch_heatmap_universe(index="NIFTY 50", top_n=5, force_refresh=True)

    assert result["source"] == "fallback_static"
    assert len(result["constituents"]) == 5
    assert all(c["symbol"] for c in result["constituents"])


def test_fetch_heatmap_universe_parses_live_response_and_sorts_by_movement(tmp_path, monkeypatch):
    monkeypatch.setattr(nse_heatmap, "_CACHE_FILE", tmp_path / "nse_universe.json")

    fake_body = {
        "data": [
            {"symbol": "NIFTY 50", "pChange": 0.0, "lastPrice": 100},
            {"symbol": "A", "pChange": 1.0, "lastPrice": 10},
            {"symbol": "B", "pChange": -5.0, "lastPrice": 20},
            {"symbol": "C", "pChange": 2.0, "lastPrice": 30},
        ]
    }

    with patch("quant_intelligence.data_adapters.nse_heatmap.requests.Session") as mock_session_cls:
        mock_session = MagicMock()
        api_resp = MagicMock()
        api_resp.json.return_value = fake_body
        api_resp.raise_for_status.return_value = None
        mock_session.get.return_value = api_resp
        mock_session_cls.return_value = mock_session

        result = nse_heatmap.fetch_heatmap_universe(index="NIFTY 50", top_n=10, force_refresh=True)

    assert result["source"] == "nse_live"
    symbols = [c["symbol"] for c in result["constituents"]]
    assert symbols == ["B", "C", "A"]  # sorted by |pChange| descending, index itself excluded


def test_fetch_heatmap_universe_uses_cache_within_refresh_window(tmp_path, monkeypatch):
    cache_file = tmp_path / "nse_universe.json"
    monkeypatch.setattr(nse_heatmap, "_CACHE_FILE", cache_file)

    with patch("quant_intelligence.data_adapters.nse_heatmap.requests.Session") as mock_session_cls:
        mock_session = MagicMock()
        api_resp = MagicMock()
        api_resp.json.return_value = {"data": [{"symbol": "A", "pChange": 1.0, "lastPrice": 10}]}
        api_resp.raise_for_status.return_value = None
        mock_session.get.return_value = api_resp
        mock_session_cls.return_value = mock_session

        nse_heatmap.fetch_heatmap_universe(index="NIFTY 50", top_n=10, force_refresh=True)
        assert mock_session.get.call_count == 2  # homepage + api

        # Second call within the refresh window should not hit the network again.
        nse_heatmap.fetch_heatmap_universe(index="NIFTY 50", top_n=10, force_refresh=False)
        assert mock_session.get.call_count == 2
