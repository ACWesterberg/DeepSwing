from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.analysis.regime import RegimeResult
from src.analysis.screener import ScreenerCandidate
from src.analysis.technical import TechnicalSignals
from src.portfolio.simulator import get_portfolio, reset_portfolios

# Must be imported before patch() resolves the target paths
import src.scheduler.scan_loop  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signals(ticker: str = "AAPL", price: float = 100.0) -> TechnicalSignals:
    return TechnicalSignals(
        ticker=ticker,
        ema_21=98.0, sma_50=95.0, sma_200=88.0,
        price_above_50sma=True, price_above_200sma=True, ema_21_above_50sma=True,
        atr_14=2.0, bb_upper=106.0, bb_middle=100.0, bb_lower=94.0, bb_pct_b=0.5,
        rsi_14=52.0, parabolic_sar=96.0, sar_is_bearish=False,
        ease_of_movement=0.01, obv=1_000_000.0,
        volume_ratio=2.0, volume_spike=True,
        fib_38_2=97.0, fib_61_8=93.0,
        current_price=price, current_volume=600_000.0,
    )


def _make_regime(regime: str = "trending") -> RegimeResult:
    return RegimeResult(
        regime=regime, hurst_exponent=0.65, autocorrelation=0.12,
        recommended_tactic="EMA crossover entries",
    )


def _make_candidate(ticker: str = "AAPL") -> ScreenerCandidate:
    return ScreenerCandidate(
        ticker=ticker, market="us",
        signals=_make_signals(ticker),
        regime=_make_regime(),
    )


def _buy_decision(entry: float = 100.0) -> dict:
    return {
        "action": "BUY",
        "stop_loss": entry - 3.0,
        "target": entry + 10.0,
        "reasoning": "Strong momentum",
        "confidence": 0.82,
    }


# ---------------------------------------------------------------------------
# Shared mock context
# ---------------------------------------------------------------------------

SCAN_PATCHES = [
    "src.scheduler.scan_loop.fetch_batch_us",
    "src.scheduler.scan_loop.get_macro_context",
    "src.scheduler.scan_loop.fetch_news_for_ticker",
    "src.scheduler.scan_loop.get_insider_summary",
    "src.scheduler.scan_loop.analyze_news",
    "src.scheduler.scan_loop.get_decision",
    "src.scheduler.scan_loop.get_sector",
    "src.scheduler.scan_loop.get_current_price",
    "src.scheduler.scan_loop._get_current_prices",
    "src.scheduler.scan_loop.run_erl",
    "src.scheduler.scan_loop.get_vix",
    "src.scheduler.scan_loop.screen_candidates",
    "src.scheduler.scan_loop.compute_signals",
    "src.scheduler.scan_loop.classify_regime",
    "src.scheduler.scan_loop._to_sek_price",
]


def _apply_patches(patches: list, **overrides):
    """Return a dict of started mock objects, with sensible defaults."""
    started = {}
    for target in patches:
        name = target.split(".")[-1]
        mock = patch(target).start()
        started[name] = mock

    # Defaults
    started["fetch_batch_us"].return_value = {"AAPL": MagicMock()}
    started["get_macro_context"].return_value = "No macro data."
    started["fetch_news_for_ticker"].return_value = []
    started["get_insider_summary"].return_value = "No insider activity."
    started["analyze_news"].return_value = "Neutral news."
    started["get_sector"].return_value = "Technology"
    started["get_current_price"].return_value = 100.0
    started["run_erl"].return_value = None
    started["get_vix"].return_value = 20.0
    started["screen_candidates"].return_value = [_make_candidate("AAPL")]
    started["compute_signals"].return_value = _make_signals("AAPL")
    started["classify_regime"].return_value = _make_regime()
    started["get_decision"].return_value = _buy_decision(100.0)
    # Pass prices through unchanged — FX conversion tested separately in TestToSekPrice
    started["_to_sek_price"].side_effect = lambda price, ticker, market: price
    # Use the mocked get_current_price value so tests can control stop-hit behaviour
    started["_get_current_prices"].side_effect = (
        lambda tickers, market: {t: started["get_current_price"].return_value for t in tickers}
    )

    for key, val in overrides.items():
        started[key].return_value = val

    return started


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunScanReturnShape:
    def setup_method(self):
        reset_portfolios()

    def teardown_method(self):
        patch.stopall()
        reset_portfolios()

    def test_returns_dict_with_expected_keys(self):
        _apply_patches(SCAN_PATCHES)
        from src.scheduler.scan_loop import run_scan
        result = run_scan("us")
        assert "market" in result
        assert "candidates" in result
        assert "decisions" in result
        assert result["market"] == "us"

    def test_empty_when_no_ohlcv_data(self):
        _apply_patches(SCAN_PATCHES, fetch_batch_us={})
        from src.scheduler.scan_loop import run_scan
        result = run_scan("us")
        assert result["candidates"] == []
        assert result["decisions"] == []

    def test_empty_when_no_candidates_pass_screener(self):
        _apply_patches(SCAN_PATCHES, screen_candidates=[])
        from src.scheduler.scan_loop import run_scan
        result = run_scan("us")
        assert result["candidates"] == []

    def test_vix_halt_returns_early(self):
        _apply_patches(SCAN_PATCHES, get_vix=35.0)
        from src.scheduler.scan_loop import run_scan
        result = run_scan("us")
        assert result.get("vix_halt") is True
        assert result["decisions"] == []

    def test_vix_below_threshold_proceeds(self):
        mocks = _apply_patches(SCAN_PATCHES, get_vix=34.9)
        from src.scheduler.scan_loop import run_scan
        result = run_scan("us")
        assert result.get("vix_halt") is None


class TestRunScanTradeExecution:
    def setup_method(self):
        reset_portfolios()

    def teardown_method(self):
        patch.stopall()
        reset_portfolios()

    def test_buy_decision_opens_position(self):
        _apply_patches(SCAN_PATCHES)
        from src.scheduler.scan_loop import run_scan
        run_scan("us")
        for track in ("claude", "gpt"):
            portfolio = get_portfolio(track)
            assert len(portfolio.open_positions) == 1
            pos = portfolio.open_positions[0]
            assert pos.ticker == "AAPL"
            assert pos.sector == "Technology"
            assert pos.regime == "trending"
            assert pos.technical_snapshot != ""

    def test_hold_decision_opens_no_position(self):
        _apply_patches(SCAN_PATCHES, get_decision={"action": "HOLD"})
        from src.scheduler.scan_loop import run_scan
        run_scan("us")
        for track in ("claude", "gpt"):
            assert len(get_portfolio(track).open_positions) == 0

    def test_buy_with_bad_stop_blocked_by_risk(self):
        # stop_loss above entry → rejected by risk.py
        bad_decision = {"action": "BUY", "stop_loss": 105.0, "target": 115.0,
                        "reasoning": "test", "confidence": 0.7}
        _apply_patches(SCAN_PATCHES, get_decision=bad_decision)
        from src.scheduler.scan_loop import run_scan
        result = run_scan("us")
        for track in ("claude", "gpt"):
            assert len(get_portfolio(track).open_positions) == 0
        blocked = [d for d in result["decisions"] if d.get("action") == "BLOCKED"]
        assert len(blocked) > 0

    def test_decisions_log_contains_buy_entry(self):
        _apply_patches(SCAN_PATCHES)
        from src.scheduler.scan_loop import run_scan
        result = run_scan("us")
        buy_decisions = [d for d in result["decisions"] if d["action"] == "BUY"]
        assert len(buy_decisions) > 0
        assert buy_decisions[0]["ticker"] == "AAPL"
        assert "entry_price" in buy_decisions[0]
        assert "rrr" in buy_decisions[0]


class TestRunScanStopHitAndERL:
    def setup_method(self):
        reset_portfolios()

    def teardown_method(self):
        patch.stopall()
        reset_portfolios()

    def test_stop_hit_triggers_erl(self):
        mocks = _apply_patches(SCAN_PATCHES)
        from src.scheduler.scan_loop import run_scan

        # Open a position first
        run_scan("us")

        # Reset mock call count, then update prices below stop
        mocks["run_erl"].reset_mock()
        # Price below stop (entry ~100, stop=97, current_price mock returns 95 → stop hit)
        mocks["get_current_price"].return_value = 85.0
        run_scan("us")

        # ERL should have been called once per track
        assert mocks["run_erl"].call_count == len(["claude", "gpt"])

    def test_erl_receives_technical_snapshot(self):
        mocks = _apply_patches(SCAN_PATCHES)
        from src.scheduler.scan_loop import run_scan

        run_scan("us")
        mocks["run_erl"].reset_mock()
        mocks["get_current_price"].return_value = 80.0
        run_scan("us")

        for call in mocks["run_erl"].call_args_list:
            technicals_str = call.kwargs.get("technicals_str") or call.args[2]
            assert technicals_str != "See trade entry data"
            assert len(technicals_str) > 10


class TestCurrencyForTicker:
    def test_us_market_is_always_usd(self):
        from src.scheduler.scan_loop import _currency_for_ticker
        assert _currency_for_ticker("AAPL", "us") == "USD"

    def test_swedish_suffix_is_sek(self):
        from src.scheduler.scan_loop import _currency_for_ticker
        assert _currency_for_ticker("ERIC-B.ST", "nordic") == "SEK"

    def test_norwegian_suffix_is_nok(self):
        from src.scheduler.scan_loop import _currency_for_ticker
        assert _currency_for_ticker("EQNR.OL", "nordic") == "NOK"

    def test_finnish_suffix_is_eur(self):
        from src.scheduler.scan_loop import _currency_for_ticker
        assert _currency_for_ticker("DIGIA.HE", "nordic") == "EUR"

    def test_danish_suffix_is_dkk(self):
        from src.scheduler.scan_loop import _currency_for_ticker
        assert _currency_for_ticker("NOVO-B.CO", "nordic") == "DKK"

    def test_legacy_sto_suffix_defaults_to_sek(self):
        from src.scheduler.scan_loop import _currency_for_ticker
        assert _currency_for_ticker("ERIC-B.STO", "nordic") == "SEK"


class TestToSekPrice:
    def teardown_method(self):
        import src.scheduler.scan_loop as sl
        sl._HAS_FX = False
        sl._to_sek_fn = None

    def test_swedish_price_passes_through_unchanged(self):
        from src.scheduler.scan_loop import _to_sek_price
        assert _to_sek_price(150.0, "ERIC-B.ST", "nordic") == 150.0

    def test_us_price_without_fx_passes_through(self):
        import src.scheduler.scan_loop as sl
        sl._HAS_FX = False
        assert sl._to_sek_price(100.0, "AAPL", "us") == 100.0

    def test_us_price_converted_via_fx(self):
        import src.scheduler.scan_loop as sl
        sl._HAS_FX = True
        sl._to_sek_fn = lambda price, currency: price * 10.5
        assert sl._to_sek_price(100.0, "AAPL", "us") == pytest.approx(1050.0)

    def test_us_price_falls_back_when_fx_returns_none(self):
        import src.scheduler.scan_loop as sl
        sl._HAS_FX = True
        sl._to_sek_fn = lambda price, currency: None
        assert sl._to_sek_price(100.0, "AAPL", "us") == 100.0

    def test_norwegian_price_converted_via_nok(self):
        import src.scheduler.scan_loop as sl
        sl._HAS_FX = True
        sl._to_sek_fn = lambda price, currency: price * 1.0 if currency == "NOK" else None
        assert sl._to_sek_price(100.0, "EQNR.OL", "nordic") == pytest.approx(100.0)

    def test_finnish_price_converted_via_eur(self):
        import src.scheduler.scan_loop as sl
        sl._HAS_FX = True
        sl._to_sek_fn = lambda price, currency: price * 11.0 if currency == "EUR" else None
        assert sl._to_sek_price(46.0, "DIGIA.HE", "nordic") == pytest.approx(506.0)

    def test_danish_price_converted_via_dkk(self):
        import src.scheduler.scan_loop as sl
        sl._HAS_FX = True
        sl._to_sek_fn = lambda price, currency: price * 1.5 if currency == "DKK" else None
        assert sl._to_sek_price(100.0, "NOVO-B.CO", "nordic") == pytest.approx(150.0)


class TestGetCurrentPrices:
    def teardown_method(self):
        patch.stopall()
        import src.scheduler.scan_loop as sl
        sl._HAS_LIVE = False
        sl._HAS_FX = False

    def test_empty_tickers_returns_empty(self):
        from src.scheduler.scan_loop import _get_current_prices
        assert _get_current_prices([], "us") == {}

    def test_uses_get_current_price_fallback(self):
        import src.scheduler.scan_loop as sl
        sl._HAS_LIVE = False
        sl._HAS_FX = False
        with patch("src.scheduler.scan_loop.get_current_price", return_value=55.0) as mock_p:
            result = sl._get_current_prices(["AAPL", "MSFT"], "us")
        assert result == {"AAPL": 55.0, "MSFT": 55.0}

    def test_fallback_applies_fx_conversion(self):
        import src.scheduler.scan_loop as sl
        sl._HAS_LIVE = False
        sl._HAS_FX = True
        sl._to_sek_fn = lambda price, currency: price * 10.0
        with patch("src.scheduler.scan_loop.get_current_price", return_value=50.0):
            result = sl._get_current_prices(["AAPL"], "us")
        assert result == {"AAPL": pytest.approx(500.0)}

    def test_live_path_maps_yf_tickers_back(self):
        import src.scheduler.scan_loop as sl
        sl._HAS_LIVE = True
        sl._HAS_FX = False
        sl._get_live_prices = lambda tickers: {t: 200.0 for t in tickers}
        result = sl._get_current_prices(["ERIC-B.STO"], "nordic")
        assert "ERIC-B.STO" in result
        assert result["ERIC-B.STO"] == pytest.approx(200.0)

    def test_live_path_falls_back_on_exception(self):
        import src.scheduler.scan_loop as sl
        sl._HAS_LIVE = True
        sl._HAS_FX = False

        def _raise(_):
            raise RuntimeError("network error")

        sl._get_live_prices = _raise
        with patch("src.scheduler.scan_loop.get_current_price", return_value=77.0):
            result = sl._get_current_prices(["AAPL"], "us")
        assert result == {"AAPL": 77.0}

    def test_live_path_applies_fx_conversion(self):
        import src.scheduler.scan_loop as sl
        sl._HAS_LIVE = True
        sl._HAS_FX = True
        sl._to_sek_fn = lambda price, currency: price * 10.0
        sl._get_live_prices = lambda tickers: {t: 100.0 for t in tickers}
        result = sl._get_current_prices(["AAPL"], "us")
        assert result == {"AAPL": pytest.approx(1000.0)}

    def test_live_path_converts_eur_nordic_ticker_not_skipped_as_sek(self):
        import src.scheduler.scan_loop as sl
        sl._HAS_LIVE = True
        sl._HAS_FX = True
        sl._to_sek_fn = lambda price, currency: price * 11.0 if currency == "EUR" else price
        sl._get_live_prices = lambda tickers: {t: 46.0 for t in tickers}
        result = sl._get_current_prices(["DIGIA.HE"], "nordic")
        assert result == {"DIGIA.HE": pytest.approx(506.0)}


class TestRunScanEventCallback:
    def setup_method(self):
        reset_portfolios()

    def teardown_method(self):
        patch.stopall()
        reset_portfolios()
        from src.scheduler.scan_loop import set_trade_event_handler
        set_trade_event_handler(None)

    def test_trade_opened_event_fires(self):
        _apply_patches(SCAN_PATCHES)
        from src.scheduler.scan_loop import run_scan, set_trade_event_handler

        events = []
        set_trade_event_handler(events.append)
        run_scan("us")

        opened = [e for e in events if e.get("event") == "trade_opened"]
        # One per track
        assert len(opened) == 2
        assert opened[0]["data"]["ticker"] == "AAPL"

    def test_trade_closed_event_fires_on_stop(self):
        mocks = _apply_patches(SCAN_PATCHES)
        from src.scheduler.scan_loop import run_scan, set_trade_event_handler

        events = []
        set_trade_event_handler(events.append)

        run_scan("us")
        events.clear()

        mocks["get_current_price"].return_value = 80.0
        run_scan("us")

        closed = [e for e in events if e.get("event") == "trade_closed"]
        assert len(closed) == 2
        assert all(c["data"]["exit_reason"] == "stop_loss" for c in closed)
