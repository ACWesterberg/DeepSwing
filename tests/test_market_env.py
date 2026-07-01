from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import src.scheduler.scan_loop as sl
from src.data.news_fetcher import format_market_environment


class TestFormatMarketEnvironment:
    def test_empty_headlines(self):
        assert format_market_environment([]) == "No market-wide news available."

    def test_formats_headlines_with_source_and_time(self):
        out = format_market_environment([
            {"headline": "Riksbanken höjer räntan", "source": "DI", "published_at": "2026-07-01 08:00"},
            {"headline": "Oil spikes on Gulf tension", "source": "MFN", "published_at": "2026-07-01 07:30"},
        ])
        assert "Riksbanken höjer räntan" in out
        assert "Oil spikes on Gulf tension" in out
        assert "DI" in out and "MFN" in out
        assert out.startswith("Recent market-wide headlines")


def _cand(ticker: str):
    return SimpleNamespace(ticker=ticker)


class TestEarningsFilter:
    def test_drops_candidates_within_buffer(self):
        cands = [_cand("AAPL"), _cand("MSFT"), _cand("NVDA")]
        days = {"AAPL": 10, "MSFT": 1, "NVDA": None}  # MSFT reports tomorrow
        with patch.object(sl, "get_days_to_earnings", return_value=days):
            with patch.object(sl.settings, "earnings_buffer_days", 2):
                kept = sl._filter_earnings(cands)
        tickers = [c.ticker for c in kept]
        assert "MSFT" not in tickers          # within buffer → dropped
        assert "AAPL" in tickers              # 10 days out → kept
        assert "NVDA" in tickers              # unknown date → kept

    def test_buffer_zero_disables_filter(self):
        cands = [_cand("AAPL"), _cand("MSFT")]
        with patch.object(sl.settings, "earnings_buffer_days", 0):
            # get_days_to_earnings must not even be needed; nothing dropped
            kept = sl._filter_earnings(cands)
        assert len(kept) == 2

    def test_exact_boundary_is_dropped(self):
        cands = [_cand("AAPL")]
        with patch.object(sl, "get_days_to_earnings", return_value={"AAPL": 2}):
            with patch.object(sl.settings, "earnings_buffer_days", 2):
                kept = sl._filter_earnings(cands)
        assert kept == []  # days == buffer → dropped
