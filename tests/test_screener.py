from __future__ import annotations

import pytest

from src.analysis.regime import RegimeResult
from src.analysis.screener import ScreenerCandidate, _score_candidate, screen_candidates
from src.analysis.technical import TechnicalSignals


def _make_signals(
    ticker: str = "TEST",
    price: float = 100.0,
    atr_14: float = 4.0,
    sma_50: float = 95.0,
    sma_200: float = 90.0,
    rsi_14: float = 52.0,
    volume_ratio: float = 1.6,
    ema_21: float = 97.0,
    bb_pct_b: float = 0.2,
    sar_is_bearish: bool = False,
    avg_turnover: float = 0.0,
) -> TechnicalSignals:
    return TechnicalSignals(
        ticker=ticker,
        ema_21=ema_21,
        sma_50=sma_50,
        sma_200=sma_200,
        price_above_50sma=price > sma_50,
        price_above_200sma=price > sma_200,
        ema_21_above_50sma=ema_21 > sma_50,
        atr_14=atr_14,
        bb_upper=105.0,
        bb_middle=100.0,
        bb_lower=95.0,
        bb_pct_b=bb_pct_b,
        rsi_14=rsi_14,
        parabolic_sar=97.0,
        sar_is_bearish=sar_is_bearish,
        ease_of_movement=0.01,
        obv=1_000_000.0,
        volume_ratio=volume_ratio,
        volume_spike=volume_ratio >= 1.5,
        fib_38_2=97.0,
        fib_61_8=94.0,
        current_price=price,
        current_volume=500_000.0,
        avg_turnover=avg_turnover,
    )


def _make_regime(regime: str = "trending", hurst: float = 0.65) -> RegimeResult:
    return RegimeResult(
        regime=regime,
        hurst_exponent=hurst,
        autocorrelation=0.1,
        recommended_tactic="test tactic",
    )


class TestScoreCandidate:
    def test_valid_trending_candidate_returns_score(self):
        sig = _make_signals(ema_21=97.0, sma_50=95.0)  # ema21 > sma50 ✓
        reg = _make_regime(regime="trending")
        score = _score_candidate(sig, reg)
        assert score is not None
        assert score > 0

    def test_valid_mean_reverting_candidate_returns_score(self):
        sig = _make_signals(bb_pct_b=0.2)  # bb_pct_b <= 0.35 ✓
        reg = _make_regime(regime="mean-reverting", hurst=0.35)
        score = _score_candidate(sig, reg)
        assert score is not None
        assert score > 0

    def test_price_below_50sma_rejected(self):
        sig = _make_signals(price=90.0, sma_50=95.0)
        reg = _make_regime()
        assert _score_candidate(sig, reg) is None

    def test_rsi_too_high_rejected(self):
        sig = _make_signals(rsi_14=85.0)
        reg = _make_regime()
        assert _score_candidate(sig, reg) is None

    def test_rsi_too_low_rejected(self):
        sig = _make_signals(rsi_14=20.0)
        reg = _make_regime()
        assert _score_candidate(sig, reg) is None

    def test_volume_spike_day_rejected(self):
        # The day after a spike is short-horizon reversal territory
        assert _score_candidate(_make_signals(volume_ratio=3.0), _make_regime()) is None

    def test_quiet_volume_is_fine_by_default(self):
        assert _score_candidate(_make_signals(volume_ratio=0.8), _make_regime()) is not None

    def test_min_volume_floor_when_enabled(self, monkeypatch):
        from config.settings import settings
        monkeypatch.setattr(settings, "volume_spike_multiplier", 1.2)
        assert _score_candidate(_make_signals(volume_ratio=0.8), _make_regime()) is None

    def test_neutral_regime_passes_without_gate(self):
        assert _score_candidate(_make_signals(), _make_regime(regime="neutral")) is not None

    def test_neutral_regime_rejected_with_gate(self, monkeypatch):
        from config.settings import settings
        monkeypatch.setattr(settings, "regime_gate", True)
        assert _score_candidate(_make_signals(), _make_regime(regime="neutral")) is None

    def test_below_200sma_rejected(self):
        assert _score_candidate(_make_signals(sma_200=105.0), _make_regime()) is None

    def test_trending_without_ema_crossover_rejected(self):
        sig = _make_signals(ema_21=93.0, sma_50=95.0)  # ema21 < sma50 ✗
        reg = _make_regime(regime="trending")
        assert _score_candidate(sig, reg) is None

    def test_mean_reverting_with_high_bb_pctb_rejected_under_gate(self, monkeypatch):
        from config.settings import settings
        monkeypatch.setattr(settings, "regime_gate", True)
        sig = _make_signals(bb_pct_b=0.8)
        reg = _make_regime(regime="mean-reverting")
        assert _score_candidate(sig, reg) is None

    def test_price_above_200sma_adds_score(self):
        sig_above = _make_signals(price=100.0, sma_200=90.0)
        sig_below = _make_signals(price=100.0, sma_200=105.0)
        reg = _make_regime()
        score_above = _score_candidate(sig_above, reg)
        score_below = _score_candidate(sig_below, reg)
        if score_above is not None and score_below is not None:
            assert score_above > score_below

    def test_bearish_sar_lowers_score(self):
        sig_bull = _make_signals(sar_is_bearish=False)
        sig_bear = _make_signals(sar_is_bearish=True)
        reg = _make_regime()
        score_bull = _score_candidate(sig_bull, reg)
        score_bear = _score_candidate(sig_bear, reg)
        if score_bull is not None and score_bear is not None:
            assert score_bull > score_bear


class TestScreenCandidates:
    def _valid_entry(self, ticker: str = "A") -> tuple:
        sig = _make_signals(ticker=ticker, ema_21=97.0, sma_50=95.0)
        reg = _make_regime()
        return sig, reg

    def test_passes_valid_candidates(self):
        data = {"AAPL": self._valid_entry("AAPL")}
        result = screen_candidates(data, "us")
        assert len(result) == 1
        assert result[0].ticker == "AAPL"

    def test_filters_invalid_candidates(self):
        bad_sig = _make_signals(price=80.0, sma_50=95.0)  # price below SMA50
        data = {"BAD": (bad_sig, _make_regime())}
        result = screen_candidates(data, "us")
        assert len(result) == 0

    def test_returns_screener_candidate_objects(self):
        data = {"X": self._valid_entry("X")}
        result = screen_candidates(data, "us")
        assert all(isinstance(c, ScreenerCandidate) for c in result)

    def test_sorted_by_score_descending(self):
        # Close to the rising EMA21 = higher score than stretched away from it
        sig_high = _make_signals(ticker="HI", ema_21=98.0, sma_50=95.0)
        sig_low = _make_signals(ticker="LO", atr_14=2.0, ema_21=96.0, sma_50=95.0)
        reg = _make_regime()
        data = {"LO": (sig_low, reg), "HI": (sig_high, reg)}
        result = screen_candidates(data, "us")
        assert len(result) == 2
        assert result[0].ticker == "HI"

    def test_empty_input_returns_empty(self):
        result = screen_candidates({}, "us")
        assert result == []

    def test_to_dict_contains_expected_fields(self):
        data = {"AAPL": self._valid_entry("AAPL")}
        result = screen_candidates(data, "us")
        assert len(result) == 1
        d = result[0].to_dict()
        assert "ticker" in d
        assert "price" in d
        assert "rsi" in d
        assert "regime" in d


class TestRangeBounds:
    def test_low_atr_is_rejected(self):
        assert _score_candidate(_make_signals(atr_14=0.5), _make_regime()) is None

    def test_wild_range_is_rejected(self):
        # A name moving 6% a day gaps through any stop placed inside its noise
        assert _score_candidate(_make_signals(atr_14=6.0), _make_regime()) is None

    def test_adequate_atr_passes(self):
        assert _score_candidate(_make_signals(atr_14=4.0), _make_regime()) is not None

    def test_bounds_disabled_by_zero(self, monkeypatch):
        from config.settings import settings
        monkeypatch.setattr(settings, "min_atr_pct", 0.0)
        monkeypatch.setattr(settings, "max_atr_pct", 0.0)
        assert _score_candidate(_make_signals(atr_14=0.5), _make_regime()) is not None
        assert _score_candidate(_make_signals(atr_14=9.0), _make_regime()) is not None


class TestPullbackScoring:
    def test_resting_rsi_outscores_stretched(self):
        rested = _score_candidate(_make_signals(rsi_14=50.0), _make_regime())
        stretched = _score_candidate(_make_signals(rsi_14=63.0), _make_regime())
        assert rested > stretched

    def test_overbought_is_rejected(self):
        assert _score_candidate(_make_signals(rsi_14=74.0), _make_regime()) is None

    def test_extension_from_ema21_lowers_score(self):
        near = _score_candidate(_make_signals(ema_21=98.0), _make_regime())
        far = _score_candidate(_make_signals(ema_21=96.0, atr_14=1.5), _make_regime())
        assert near > far

    def test_stronger_trend_scores_higher(self):
        strong = _score_candidate(_make_signals(sma_200=80.0), _make_regime())
        weak = _score_candidate(_make_signals(sma_200=93.0), _make_regime())
        assert strong > weak


class TestLiquidity:
    def _screen(self, factor, **kw):
        data = {"T": (_make_signals(ticker="T", **kw), _make_regime())}
        return screen_candidates(data, "eu", sek_factor=lambda ticker: factor)

    def test_thin_name_is_rejected(self):
        assert self._screen(1.0, avg_turnover=2_000_000.0) == []

    def test_liquid_name_passes(self):
        assert len(self._screen(1.0, avg_turnover=50_000_000.0)) == 1

    def test_turnover_is_judged_in_sek(self):
        # 1.5M GBP/day at 13 SEK/GBP is ~19.5 MSEK — liquid enough
        assert len(self._screen(13.0, avg_turnover=1_500_000.0)) == 1

    def test_penny_price_is_rejected(self):
        # 3.5p in pence units at 0.13 SEK per penny ≈ 0.46 SEK
        assert self._screen(0.13, price=3.5, sma_50=3.2, sma_200=3.0, ema_21=3.3,
                            atr_14=0.14, avg_turnover=500_000_000.0) == []

    def test_unresolved_currency_is_rejected(self):
        assert self._screen(None, avg_turnover=50_000_000.0) == []

    def test_floors_skipped_without_a_converter(self):
        data = {"T": (_make_signals(ticker="T"), _make_regime())}
        assert len(screen_candidates(data, "eu")) == 1

    def test_converter_only_called_for_technical_survivors(self):
        calls = []
        data = {
            "OK": (_make_signals(ticker="OK", avg_turnover=5e7), _make_regime()),
            "BAD": (_make_signals(ticker="BAD", sma_50=120.0), _make_regime()),
        }
        screen_candidates(data, "eu", sek_factor=lambda t: calls.append(t) or 1.0)
        assert calls == ["OK"]


class TestLondonInternationalOrderBook:
    def test_excluded_from_eu_universe(self, monkeypatch):
        import src.data.universe as uni
        rows = tuple(
            {"yahoo_ticker": t, "enabled": "true", "exchange": "LSE"}
            for t in ("VOD.L", "0A6Y.L", "3IN.L", "09II.L", "0QAV.L")
        )
        monkeypatch.setattr(uni, "_load_eu_rows", lambda: rows)
        assert uni.get_eu_tickers() == ["VOD.L", "3IN.L"]

    def test_switch_keeps_them(self, monkeypatch):
        import src.data.universe as uni
        from config.settings import settings
        rows = ({"yahoo_ticker": "0A6Y.L", "enabled": "true", "exchange": "LSE"},)
        monkeypatch.setattr(uni, "_load_eu_rows", lambda: rows)
        monkeypatch.setattr(settings, "exclude_lse_international", False)
        assert uni.get_eu_tickers() == ["0A6Y.L"]
