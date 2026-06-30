from __future__ import annotations

import csv
import io
from unittest.mock import patch

import pytest

from src.data.universe import get_nordic_tickers, get_sector_from_universe
from src.data.watchlist import get_omxs30_tickers


def _write_universe(path, rows: list[dict]) -> None:
    fieldnames = ["yahoo_ticker", "exchange", "enabled", "sector", "name"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


@pytest.fixture(autouse=True)
def _clear_cache():
    from src.data.universe import _load_rows
    _load_rows.cache_clear()
    yield
    _load_rows.cache_clear()


class TestGetNordicTickersFromUniverse:
    """Tests for universe.get_nordic_tickers() — the core filtering logic."""

    def test_returns_enabled_omxs_tickers(self, tmp_path):
        _write_universe(tmp_path / "universe.csv", [
            {"yahoo_ticker": "ERIC-B.ST", "exchange": "OMXS", "enabled": "true"},
            {"yahoo_ticker": "VOLV-B.ST", "exchange": "OMXS", "enabled": "true"},
            {"yahoo_ticker": "HIDDEN.ST", "exchange": "OMXS", "enabled": "false"},
        ])
        with patch("src.data.universe.UNIVERSE_PATH", tmp_path / "universe.csv"):
            result = get_nordic_tickers()
        assert "ERIC-B.ST" in result
        assert "VOLV-B.ST" in result
        assert "HIDDEN.ST" not in result

    def test_excludes_non_main_board_exchanges(self, tmp_path):
        _write_universe(tmp_path / "universe.csv", [
            {"yahoo_ticker": "MAIN.ST", "exchange": "OMXS", "enabled": "true"},
            {"yahoo_ticker": "SMALL.ST", "exchange": "FIRST_NORTH", "enabled": "true"},
        ])
        with patch("src.data.universe.UNIVERSE_PATH", tmp_path / "universe.csv"):
            result = get_nordic_tickers()
        assert "MAIN.ST" in result
        assert "SMALL.ST" not in result

    def test_includes_all_supported_nordic_exchanges(self, tmp_path):
        _write_universe(tmp_path / "universe.csv", [
            {"yahoo_ticker": "SE.ST", "exchange": "OMXS", "enabled": "true"},
            {"yahoo_ticker": "NO.OL", "exchange": "OSLO", "enabled": "true"},
            {"yahoo_ticker": "FI.HE", "exchange": "OMXH", "enabled": "true"},
            {"yahoo_ticker": "DK.CO", "exchange": "OMXC", "enabled": "true"},
        ])
        with patch("src.data.universe.UNIVERSE_PATH", tmp_path / "universe.csv"):
            result = get_nordic_tickers()
        assert set(result) == {"SE.ST", "NO.OL", "FI.HE", "DK.CO"}

    def test_empty_csv_returns_empty(self, tmp_path):
        _write_universe(tmp_path / "universe.csv", [])
        with patch("src.data.universe.UNIVERSE_PATH", tmp_path / "universe.csv"):
            result = get_nordic_tickers()
        assert result == []


class TestGetSectorFromUniverse:
    def test_returns_sector_for_known_ticker(self, tmp_path):
        _write_universe(tmp_path / "universe.csv", [
            {"yahoo_ticker": "ERIC-B.ST", "exchange": "OMXS", "enabled": "true", "sector": "Technology"},
        ])
        with patch("src.data.universe.UNIVERSE_PATH", tmp_path / "universe.csv"):
            assert get_sector_from_universe("ERIC-B.ST") == "Technology"

    def test_returns_none_for_unknown_ticker(self, tmp_path):
        _write_universe(tmp_path / "universe.csv", [])
        with patch("src.data.universe.UNIVERSE_PATH", tmp_path / "universe.csv"):
            assert get_sector_from_universe("UNKNOWN.ST") is None


class TestGetOmxs30TickersFallback:
    """Tests for watchlist.get_omxs30_tickers() — specifically the fallback behaviour."""

    def test_falls_back_to_hardcoded_list_on_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent.csv"
        with patch("src.data.universe.UNIVERSE_PATH", missing):
            result = get_omxs30_tickers()
        assert len(result) >= 20
        assert all(isinstance(t, str) for t in result)

    def test_falls_back_when_universe_returns_fewer_than_20_tickers(self, tmp_path):
        _write_universe(tmp_path / "universe.csv", [
            {"yahoo_ticker": f"T{i}.ST", "exchange": "OMXS", "enabled": "true"}
            for i in range(5)
        ])
        with patch("src.data.universe.UNIVERSE_PATH", tmp_path / "universe.csv"):
            result = get_omxs30_tickers()
        # Falls back to hardcoded list
        assert len(result) >= 20

    def test_uses_universe_when_20_or_more_tickers(self, tmp_path):
        rows = [
            {"yahoo_ticker": f"T{i:02d}.ST", "exchange": "OMXS", "enabled": "true"}
            for i in range(25)
        ]
        _write_universe(tmp_path / "universe.csv", rows)
        with patch("src.data.universe.UNIVERSE_PATH", tmp_path / "universe.csv"):
            result = get_omxs30_tickers()
        assert len(result) == 25
        assert "T00.ST" in result
