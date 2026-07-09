from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from config.settings import settings
from src.backtesting.news_history import _relative_age, company_query_name, format_headlines_block
from src.backtesting.optimize import dataset_stats, load_records
from src.backtesting.replay import _load_seen_keys, _trading_days_between, simulate_outcome


def _make_df(rows: list[dict], start: str = "2025-03-03") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(rows))
    return pd.DataFrame(rows, index=idx)


def _flat_bars(n: int, price: float = 100.0) -> list[dict]:
    return [
        {"Open": price, "High": price + 1, "Low": price - 1, "Close": price, "Volume": 1e6}
        for _ in range(n)
    ]


SCAN_DAY = date(2025, 3, 3)  # first bar; forward window starts on the 4th
ATR = 2.0  # stop = entry - 3.0 with the default 1.5x multiplier


class TestSimulateOutcome:
    def test_stop_hit_intra_bar(self):
        bars = _flat_bars(10)
        bars[3]["Low"] = 90.0
        out = simulate_outcome(_make_df(bars), SCAN_DAY, ATR, "nordic", 2.5, 20)
        assert out.exit_reason == "stop_loss"
        assert out.exit_price == pytest.approx(out.entry_price - settings.atr_stop_multiplier * ATR)
        assert out.net_pnl_pct < 0

    def test_gap_through_stop_exits_at_open(self):
        bars = _flat_bars(10)
        bars[4].update({"Open": 90.0, "High": 91.0, "Low": 89.0, "Close": 90.5})
        out = simulate_outcome(_make_df(bars), SCAN_DAY, ATR, "nordic", 2.5, 20)
        assert out.exit_reason == "stop_loss"
        assert out.exit_price == 90.0

    def test_target_hit(self):
        bars = _flat_bars(10)
        bars[5]["High"] = 120.0
        out = simulate_outcome(_make_df(bars), SCAN_DAY, ATR, "nordic", 2.5, 20)
        assert out.exit_reason == "take_profit"
        assert out.exit_price == pytest.approx(out.target)
        assert out.net_pnl_pct > 0

    def test_both_touched_same_bar_assumes_stop_first(self):
        bars = _flat_bars(10)
        bars[3].update({"High": 120.0, "Low": 90.0})
        out = simulate_outcome(_make_df(bars), SCAN_DAY, ATR, "nordic", 2.5, 20)
        assert out.exit_reason == "stop_loss"

    def test_timeout_exits_at_close(self):
        bars = _flat_bars(30)
        out = simulate_outcome(_make_df(bars), SCAN_DAY, ATR, "nordic", 2.5, 5)
        assert out.exit_reason == "timeout"
        assert out.exit_price == 100.0
        # flat price: gross 0, net negative by round-trip costs
        assert out.gross_pnl_pct == pytest.approx(0.0)
        assert out.net_pnl_pct < 0

    def test_us_costs_exceed_nordic(self):
        bars = _flat_bars(30)
        nordic = simulate_outcome(_make_df(bars), SCAN_DAY, ATR, "nordic", 2.5, 5)
        us = simulate_outcome(_make_df(bars), SCAN_DAY, ATR, "us", 2.5, 5)
        assert us.net_pnl_pct < nordic.net_pnl_pct

    def test_too_few_forward_bars_returns_none(self):
        bars = _flat_bars(4)
        assert simulate_outcome(_make_df(bars), SCAN_DAY, ATR, "nordic", 2.5, 20) is None

    def test_entry_is_next_bar_open(self):
        bars = _flat_bars(10)
        bars[1]["Open"] = 105.0
        out = simulate_outcome(_make_df(bars), SCAN_DAY, ATR, "nordic", 2.5, 20)
        assert out.entry_price == 105.0
        assert out.entry_date == date(2025, 3, 4)


class TestNewsFormatting:
    def test_relative_age(self):
        asof = date(2025, 3, 10)
        assert _relative_age("2025-03-10 09:00", asof) == "today"
        assert _relative_age("2025-03-07 12:00", asof) == "3d ago"
        assert _relative_age(None, asof) == "recent"
        assert _relative_age("garbage", asof) == "recent"

    def test_block_uses_relative_ages_not_dates(self):
        articles = [
            {"headline": "Volvo wins truck order", "published_at": "2025-03-08 08:00", "source": "DI"},
        ]
        block = format_headlines_block(articles, date(2025, 3, 10))
        assert "2d ago" in block
        assert "2025-03-08" not in block
        assert "Volvo wins truck order" in block

    def test_empty_block(self):
        assert format_headlines_block([], date(2025, 3, 10)) == "No recent relevant news found."

    def test_company_query_name_strips_share_class(self):
        assert company_query_name("VOLV-B.ST") == "Volvo"
        assert company_query_name("UNKNOWN.ST") == "UNKNOWN"


class TestTrainsetIO:
    def _record(self, ticker: str = "AAPL", scan_date: str = "2025-03-03", pnl: float = 0.05) -> dict:
        return {
            "schema": 1, "market": "us", "ticker": ticker, "scan_date": scan_date,
            "exit_reason": "take_profit", "pnl_pct": pnl,
            "inputs": {"technicals": "t", "regime": "r", "news_summary": "n",
                       "macro_context": "m", "heuristics": "h"},
        }

    def test_load_records_sorted_and_filters_malformed(self, tmp_path):
        path = tmp_path / "trainset.jsonl"
        lines = [
            json.dumps(self._record(scan_date="2025-05-01")),
            "not json",
            json.dumps({"pnl_pct": 0.1}),  # missing inputs
            json.dumps(self._record(scan_date="2025-03-03", pnl=-0.02)),
        ]
        path.write_text("\n".join(lines) + "\n")
        records = load_records(path)
        assert len(records) == 2
        assert records[0]["scan_date"] == "2025-03-03"

    def test_dataset_stats(self, tmp_path):
        path = tmp_path / "trainset.jsonl"
        path.write_text(
            json.dumps(self._record(pnl=0.05)) + "\n" + json.dumps(self._record(ticker="MSFT", pnl=-0.03)) + "\n"
        )
        stats = dataset_stats(load_records(path))
        assert stats["examples"] == 2
        assert stats["winners"] == 1
        assert stats["losers"] == 1
        assert stats["by_market"] == {"us": 2}

    def test_seen_keys_resume(self, tmp_path):
        path = tmp_path / "trainset.jsonl"
        path.write_text(json.dumps(self._record()) + "\nbroken line\n")
        assert _load_seen_keys(path) == {("us", "AAPL", "2025-03-03")}
        assert _load_seen_keys(tmp_path / "missing.jsonl") == set()


class TestGenerateTrainset:
    def _synthetic_setup(self, monkeypatch, tmp_path):
        import src.backtesting.replay as replay
        from src.analysis.screener import ScreenerCandidate

        prices = [100 * (1.002**i) for i in range(300)]
        rows = [
            {"Open": p, "High": p * 1.01, "Low": p * 0.99, "Close": p, "Volume": 1e6}
            for p in prices
        ]
        df = _make_df(rows, start="2024-06-03")

        monkeypatch.setattr(replay, "load_ohlcv_history", lambda *a, **k: {"TEST.ST": df})
        monkeypatch.setattr(replay, "get_omxs30_tickers", lambda: ["TEST.ST"])

        class FakeMacro:
            def vix(self, day):
                return 18.0

            def context(self, market, day):
                return "Macro snapshot (as of previous close):\n- VIX: 18.0 (normal)"

        monkeypatch.setattr(replay, "MacroHistory", lambda *a, **k: FakeMacro())
        # Screener quality gates are covered by test_screener.py — pass everything
        # through so this test exercises the replay plumbing deterministically
        monkeypatch.setattr(
            replay, "screen_candidates",
            lambda amap, market: [ScreenerCandidate(t, market, s, r) for t, (s, r) in amap.items()],
        )

        cfg = replay.ReplayConfig(
            start=df.index[220].date(),
            end=df.index[-30].date(),
            markets=["nordic"],
            news_mode="off",
            out_path=tmp_path / "trainset.jsonl",
        )
        return replay, cfg

    def test_end_to_end_synthetic(self, monkeypatch, tmp_path):
        replay, cfg = self._synthetic_setup(monkeypatch, tmp_path)
        stats = replay.generate_trainset(cfg)
        assert stats["examples"] >= 2
        assert stats["winners"] >= 1  # steady uptrend

        records = load_records(cfg.out_path)
        r = records[0]
        assert set(r["inputs"]) == {"technicals", "regime", "news_summary", "macro_context", "heuristics"}
        assert "Insider activity" in r["inputs"]["news_summary"]
        assert r["inputs"]["heuristics"] == "No relevant heuristics yet."
        assert "RSI(14)" in r["inputs"]["technicals"]
        assert r["inputs"]["regime"].startswith("Regime:")
        assert r["exit_reason"] in ("stop_loss", "take_profit", "timeout")
        assert r["entry_date"] > r["scan_date"]

    def test_rerun_is_idempotent(self, monkeypatch, tmp_path):
        replay, cfg = self._synthetic_setup(monkeypatch, tmp_path)
        first = replay.generate_trainset(cfg)
        second = replay.generate_trainset(cfg)
        assert second["examples"] == 0
        assert second["skipped_seen"] == first["examples"]


def test_trading_days_between():
    days = [date(2025, 3, d) for d in (3, 4, 5, 6, 7, 10, 11)]
    assert _trading_days_between(days, date(2025, 3, 3), date(2025, 3, 4)) == 1
    assert _trading_days_between(days, date(2025, 3, 3), date(2025, 3, 10)) == 5
    assert _trading_days_between(days, date(2025, 3, 7), date(2025, 3, 7)) == 0
