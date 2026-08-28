from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from config.settings import settings
from src.analysis.event_model import EventContract
from src.portfolio.simulator import Portfolio

NOW = datetime(2026, 8, 27, 12, 0, 0)
RATE = 10.0


@pytest.fixture
def event_loop_module(monkeypatch):
    """Import the loop with a fixed FX rate and isolated portfolios."""
    from src.scheduler import event_loop

    portfolios: dict[str, Portfolio] = {}

    def fake_get_portfolio(track: str) -> Portfolio:
        if track not in portfolios:
            portfolios[track] = Portfolio(track)
        return portfolios[track]

    monkeypatch.setattr(event_loop, "get_portfolio", fake_get_portfolio)
    monkeypatch.setattr(event_loop, "_usd_sek_rate", lambda: RATE)
    monkeypatch.setattr(event_loop, "persist_portfolio", lambda p: None)
    monkeypatch.setattr(event_loop, "emit", lambda e: None)
    monkeypatch.setattr(event_loop, "_persist_event_decisions", lambda d: None)
    event_loop._portfolios_for_test = portfolios
    return event_loop, portfolios


def contract(
    ticker: str = "KXHIGHNY-26AUG28-B82",
    *,
    ask: float = 0.20,
    bid: float = 0.18,
    open_interest: int = 5000,
    floor_strike: float = 82,
    cap_strike: float = 83,
) -> EventContract:
    return EventContract(
        ticker=ticker,
        event_ticker="KXHIGHNY-26AUG28",
        series_ticker="KXHIGHNY",
        title="NYC high 82-83",
        yes_bid=bid,
        yes_ask=ask,
        last_price=(bid + ask) / 2,
        volume=2000,
        open_interest=open_interest,
        # Relative to real now: the loop prices against utcnow(), so a fixed
        # date would make lead time — and therefore sigma and the fair value —
        # depend on when the suite happens to run.
        close_time=datetime.utcnow() + timedelta(days=1),
        strike_type="between",
        floor_strike=floor_strike,
        cap_strike=cap_strike,
    )


_DEFAULT = object()


def install_pipeline(
    module,
    monkeypatch,
    *,
    contracts=None,
    forecasts=None,
    decision=_DEFAULT,
    discussion="Quiet pattern, high confidence.",
):
    contracts = [contract()] if contracts is None else contracts
    forecasts = {"KXHIGHNY-26AUG28": 82.0} if forecasts is None else forecasts
    if decision is _DEFAULT:
        decision = {
            "action": "TRADE", "confidence": 0.8, "reasoning": "Edge looks real.",
            "entry_inputs": {},
        }

    monkeypatch.setattr(module, "fetch_weather_markets", lambda: list(contracts))
    monkeypatch.setattr(module, "get_forecast_highs", lambda c: dict(forecasts))
    monkeypatch.setattr(module, "fetch_forecast_discussion", lambda s: discussion)
    monkeypatch.setattr(module, "fetch_markets_by_ticker", lambda t: {})
    monkeypatch.setattr(module, "get_event_decision", lambda **kw: decision)

    class _Store:
        def retrieve(self, **kw):
            return []

        def to_prompt_text(self, items):
            return ""

    monkeypatch.setattr(module, "get_store", lambda track: _Store())


class TestRunEventScan:
    def test_dry_run_reports_but_opens_nothing(self, event_loop_module, monkeypatch):
        module, portfolios = event_loop_module
        monkeypatch.setattr(settings, "event_dry_run", True)
        install_pipeline(module, monkeypatch)

        result = module.run_event_scan()

        assert result["dry_run"] is True
        actions = {d["action"] for d in result["decisions"]}
        assert actions == {"DRY_RUN"}
        assert all(not p.open_positions for p in portfolios.values())

    def test_live_run_opens_positions_for_both_tracks(self, event_loop_module, monkeypatch):
        module, portfolios = event_loop_module
        monkeypatch.setattr(settings, "event_dry_run", False)
        install_pipeline(module, monkeypatch)

        result = module.run_event_scan()

        opened = [d for d in result["decisions"] if d["action"] == "BUY"]
        assert {d["track"] for d in opened} == set(settings.event_tracks)
        for track in settings.event_tracks:
            assert len(portfolios[track].open_positions) == 1

    def test_position_records_the_entry_fx_rate(self, event_loop_module, monkeypatch):
        module, portfolios = event_loop_module
        monkeypatch.setattr(settings, "event_dry_run", False)
        install_pipeline(module, monkeypatch)
        module.run_event_scan()

        position = portfolios[settings.event_tracks[0]].open_positions[0]
        assert position.entry_inputs["usd_sek"] == RATE
        assert position.entry_inputs["event_ticker"] == "KXHIGHNY-26AUG28"
        assert position.market == "events"

    def test_cash_debited_by_cost_plus_fee(self, event_loop_module, monkeypatch):
        module, portfolios = event_loop_module
        monkeypatch.setattr(settings, "event_dry_run", False)
        install_pipeline(module, monkeypatch)
        module.run_event_scan()

        portfolio = portfolios[settings.event_tracks[0]]
        position = portfolio.open_positions[0]
        spent = settings.starting_capital_sek - portfolio.cash
        expected = (
            position.entry_inputs["cost_usd"] + position.entry_inputs["fee_usd"]
        ) * RATE
        assert spent == pytest.approx(expected)

    def test_pass_decision_opens_nothing(self, event_loop_module, monkeypatch):
        module, portfolios = event_loop_module
        monkeypatch.setattr(settings, "event_dry_run", False)
        install_pipeline(module, monkeypatch, decision={
            "action": "PASS", "confidence": 0.7,
            "reasoning": "Front moving through — sigma understated.",
            "entry_inputs": {},
        })

        result = module.run_event_scan()

        assert {d["action"] for d in result["decisions"]} == {"PASS"}
        assert all(not p.open_positions for p in portfolios.values())

    def test_model_error_is_recorded_not_traded(self, event_loop_module, monkeypatch):
        module, portfolios = event_loop_module
        monkeypatch.setattr(settings, "event_dry_run", False)
        install_pipeline(module, monkeypatch, decision=None)

        result = module.run_event_scan()

        assert {d["action"] for d in result["decisions"]} == {"ERROR"}
        assert all(not p.open_positions for p in portfolios.values())

    def test_no_markets_is_a_clean_noop(self, event_loop_module, monkeypatch):
        module, _ = event_loop_module
        install_pipeline(module, monkeypatch, contracts=[])
        result = module.run_event_scan()
        assert result["candidates"] == []
        assert result["decisions"] == []

    def test_no_forecasts_is_a_clean_noop(self, event_loop_module, monkeypatch):
        module, _ = event_loop_module
        install_pipeline(module, monkeypatch, forecasts={})
        result = module.run_event_scan()
        assert result["candidates"] == []

    def test_contract_without_edge_never_reaches_the_model(self, event_loop_module, monkeypatch):
        module, _ = event_loop_module
        monkeypatch.setattr(settings, "event_dry_run", False)
        asked: list = []

        # Priced at 0.90 against a fair value near 0.20 — deeply negative edge.
        install_pipeline(module, monkeypatch, contracts=[contract(ask=0.90, bid=0.88)])
        monkeypatch.setattr(
            module, "get_event_decision",
            lambda **kw: asked.append(kw) or {"action": "PASS", "confidence": 0.0,
                                              "reasoning": "", "entry_inputs": {}},
        )

        result = module.run_event_scan()
        assert asked == []
        assert result["decisions"] == []

    def test_concurrent_scan_is_refused(self, event_loop_module, monkeypatch):
        module, _ = event_loop_module
        install_pipeline(module, monkeypatch)
        module._event_scan_lock.acquire()
        try:
            assert module.run_event_scan()["busy"] is True
        finally:
            module._event_scan_lock.release()


class TestSettlement:
    def _open_one(self, module, monkeypatch):
        monkeypatch.setattr(settings, "event_dry_run", False)
        install_pipeline(module, monkeypatch)
        module.run_event_scan()

    def test_yes_result_settles_at_payout(self, event_loop_module, monkeypatch):
        module, portfolios = event_loop_module
        self._open_one(module, monkeypatch)
        portfolio = portfolios[settings.event_tracks[0]]
        position = portfolio.open_positions[0]

        monkeypatch.setattr(
            module, "fetch_markets_by_ticker",
            lambda t: {position.ticker: {"ticker": position.ticker, "result": "yes"}},
        )
        events = module._settle_and_mark()

        assert any(e["action"] == "SETTLED" for e in events)
        assert not portfolio.open_positions
        closed = portfolio.closed_trades[-1]
        assert closed.exit_reason == "settled_yes"
        assert closed.exit_price == pytest.approx(RATE)
        assert closed.pnl > 0

    def test_no_result_settles_worthless(self, event_loop_module, monkeypatch):
        module, portfolios = event_loop_module
        self._open_one(module, monkeypatch)
        portfolio = portfolios[settings.event_tracks[0]]
        position = portfolio.open_positions[0]

        monkeypatch.setattr(
            module, "fetch_markets_by_ticker",
            lambda t: {position.ticker: {"ticker": position.ticker, "result": "no"}},
        )
        module._settle_and_mark()

        closed = portfolio.closed_trades[-1]
        assert closed.exit_reason == "settled_no"
        assert closed.exit_price == 0.0
        assert closed.pnl < 0

    def test_loss_equals_the_full_stake_including_fee(self, event_loop_module, monkeypatch):
        module, portfolios = event_loop_module
        self._open_one(module, monkeypatch)
        portfolio = portfolios[settings.event_tracks[0]]
        position = portfolio.open_positions[0]
        staked = settings.starting_capital_sek - portfolio.cash

        monkeypatch.setattr(
            module, "fetch_markets_by_ticker",
            lambda t: {position.ticker: {"ticker": position.ticker, "result": "no"}},
        )
        module._settle_and_mark()

        assert portfolio.closed_trades[-1].pnl == pytest.approx(-staked)
        assert portfolio.cash == pytest.approx(settings.starting_capital_sek - staked)

    def test_unsettled_market_is_marked_not_closed(self, event_loop_module, monkeypatch):
        module, portfolios = event_loop_module
        self._open_one(module, monkeypatch)
        portfolio = portfolios[settings.event_tracks[0]]
        position = portfolio.open_positions[0]

        monkeypatch.setattr(module, "fetch_markets_by_ticker", lambda t: {
            position.ticker: {
                "ticker": position.ticker,
                "event_ticker": "KXHIGHNY-26AUG28",
                "close_time": "2026-08-28T23:59:00Z",
                "strike_type": "between", "floor_strike": 82, "cap_strike": 83,
                "yes_bid": 40, "yes_ask": 44, "last_price": 42,
                "volume": 10, "open_interest": 10,
                "result": "",
            }
        })
        module._settle_and_mark()

        assert len(portfolio.open_positions) == 1
        # mid of 0.40/0.44 is 0.42, marked into SEK at the entry rate
        assert portfolio.open_positions[0].current_price == pytest.approx(0.42 * RATE)

    def test_settlement_never_uses_the_stop_target_sweep(self, event_loop_module, monkeypatch):
        # A binary marked above its entry must not be closed by a trailing stop.
        module, portfolios = event_loop_module
        self._open_one(module, monkeypatch)
        portfolio = portfolios[settings.event_tracks[0]]
        position = portfolio.open_positions[0]

        portfolio.mark_positions({position.ticker: position.entry_price * 2})
        portfolio.mark_positions({position.ticker: position.entry_price * 0.5})
        assert len(portfolio.open_positions) == 1

    def test_nothing_open_is_a_noop(self, event_loop_module, monkeypatch):
        module, _ = event_loop_module
        assert module._settle_and_mark() == []

    def test_missing_market_data_leaves_position_alone(self, event_loop_module, monkeypatch):
        module, portfolios = event_loop_module
        self._open_one(module, monkeypatch)
        portfolio = portfolios[settings.event_tracks[0]]

        monkeypatch.setattr(module, "fetch_markets_by_ticker", lambda t: {})
        assert module._settle_and_mark() == []
        assert len(portfolio.open_positions) == 1
