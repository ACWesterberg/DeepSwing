from __future__ import annotations

import pytest


class TestGetInsiderSummaryImport:
    def test_get_insider_summary_is_importable(self):
        from src.data.insider_fetcher import get_insider_summary
        assert callable(get_insider_summary)

    def test_get_insider_summary_is_callable_with_ticker_and_market(self):
        from src.data.insider_fetcher import get_insider_summary
        # With financedata stubbed, calling it returns a MagicMock — just verify it doesn't raise
        result = get_insider_summary("AAPL", "us")
        assert result is not None
