from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from config.settings import settings
from src.data import market_data


class TestChunkedBatchFetch:
    def test_chunks_large_watchlist(self):
        tickers = [f"T{i:04d}" for i in range(325)]
        frames = {
            t: pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2026-01-01"]))
            for t in tickers
        }
        calls: list[list[str]] = []

        def _fake_batch(batch: list[str], market: str, period: str = "1y"):
            calls.append(list(batch))
            return {t: frames[t] for t in batch if t in frames}

        with patch.object(settings, "ohlcv_batch_chunk_size", 150):
            with patch.object(market_data, "get_prices_batch", side_effect=_fake_batch):
                result = market_data.fetch_batch_us(tickers)

        assert len(result) == 325
        assert len(calls) == 3
        assert len(calls[0]) == 150
        assert len(calls[1]) == 150
        assert len(calls[2]) == 25

    def test_empty_watchlist_returns_empty(self):
        assert market_data.fetch_batch_us([]) == {}
