from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from config.settings import settings
from src.agent.erl import run_erl
from src.agent.memory import get_store
from src.agent.news_analyzer import analyze_news
from src.agent.options_decision import get_option_decision
from src.agent.options_risk import validate_option_trade
from src.analysis.regime import classify_regime
from src.analysis.screener import screen_candidates
from src.analysis.technical import compute_signals
from src.analysis.vol_context import compute_vol_context
from src.data.insider_fetcher import get_insider_summary
from src.data.macro_data import get_macro_context
from src.data.market_data import fetch_batch_us, get_current_price, get_vix
from src.data.news_fetcher import fetch_market_headlines, fetch_news_for_ticker, format_market_environment
from src.data.options_chain import OptionContract, fetch_chain_shortlist, fetch_contract_quotes
from src.data.watchlist import get_us_tickers
from src.portfolio.options_simulator import ClosedOptionTrade, OptionsPortfolio, get_options_portfolio
from src.portfolio.simulator import persist_portfolio
from src.scheduler import scan_loop
from src.scheduler.scan_loop import _emit, _filter_earnings, _persist_decisions, _scan_lock, _to_sek_price

logger = logging.getLogger(__name__)

MARKET_LABEL = "options"


def run_options_scan() -> dict:
    """Options scan for the US watchlist — shares the scan lock with stock scans
    so yfinance + LLM load never doubles up."""
    if not _scan_lock.acquire(blocking=False):
        logger.info("Scan already in progress — skipping options scan")
        return {"market": MARKET_LABEL, "candidates": [], "decisions": [], "busy": True}
    try:
        return _run_options_scan()
    finally:
        _scan_lock.release()


def _run_options_scan() -> dict:
    logger.info("=== Scan started: options (US chains) ===")

    vix = get_vix()
    if vix is not None and vix >= settings.vix_halt_threshold:
        logger.warning("VIX=%.1f >= threshold %.1f — halting new option entries", vix, settings.vix_halt_threshold)
        _mark_and_persist([])
        return {"market": MARKET_LABEL, "candidates": [], "decisions": [], "vix_halt": True, "vix": vix}

    funded_tracks = [t for t in settings.options_tracks if get_options_portfolio(t).can_open_new_position]
    if not funded_tracks:
        logger.info("All cash allocated across options tracks — holdings-only monitor")
        decisions_log = _manage_holdings()
        _mark_and_persist(decisions_log)
        return {"market": MARKET_LABEL, "mode": "holdings_monitor", "candidates": [], "decisions": decisions_log}

    macro_context = get_macro_context("us")
    try:
        market_env = format_market_environment(
            fetch_market_headlines("us", limit=settings.market_news_max_headlines)
        )
        macro_context = f"{macro_context}\n\n{market_env}"
    except Exception as exc:
        logger.warning("Market-wide news fetch failed: %s", exc)

    ohlcv_map = fetch_batch_us(get_us_tickers())
    if not ohlcv_map:
        logger.warning("No OHLCV data returned for options scan")
        _mark_and_persist([])
        return {"market": MARKET_LABEL, "candidates": [], "decisions": []}

    analysis_map: dict[str, tuple] = {}
    for ticker, df in ohlcv_map.items():
        signals = compute_signals(ticker, df)
        if signals is None:
            continue
        analysis_map[ticker] = (signals, classify_regime(df))

    candidates = _filter_earnings(screen_candidates(analysis_map, "us"))
    if not candidates:
        logger.info("No candidates for options scan after screener/earnings filter")
        decisions_log = _manage_holdings()
        _mark_and_persist(decisions_log)
        return {"market": MARKET_LABEL, "candidates": [], "decisions": decisions_log}

    decisions_log: list[dict] = []

    for candidate in candidates:
        spot_usd = candidate.signals.current_price
        shortlist = fetch_chain_shortlist(candidate.ticker, spot_usd, "call", atr=candidate.signals.atr_14)
        if not shortlist:
            logger.info("No tradable contracts for %s — skipping", candidate.ticker)
            continue

        atm_iv = min(shortlist, key=lambda c: abs(abs(c.delta) - 0.5)).implied_vol
        vol_ctx = compute_vol_context(ohlcv_map[candidate.ticker], atm_iv)
        vol_ctx_str = vol_ctx.to_prompt_str() if vol_ctx else ""

        articles = fetch_news_for_ticker(candidate.ticker, "us")
        insider_summary = get_insider_summary(candidate.ticker, "us")
        news_summary = analyze_news(
            ticker=candidate.ticker,
            market="us",
            current_price=spot_usd,
            technicals_brief=f"Price {spot_usd:.4f}, RSI {candidate.signals.rsi_14:.1f}",
            articles=articles,
        )
        full_news = f"{news_summary}\nInsider activity: {insider_summary}"
        tech_snapshot = candidate.signals.to_prompt_str()

        for track in funded_tracks:
            portfolio = get_options_portfolio(track)

            store = get_store(track)
            heuristics_list = store.retrieve(
                ticker=candidate.ticker, regime=candidate.regime.regime, market="us",
            )
            decision = get_option_decision(
                candidate=candidate,
                track=track,
                shortlist=shortlist,
                news_summary=full_news,
                macro_context=macro_context,
                heuristics_text=store.to_prompt_text(heuristics_list),
                volatility_context=vol_ctx_str,
            )

            if decision is None or decision["action"] != "BUY":
                decisions_log.append({
                    "track": track,
                    "ticker": candidate.ticker,
                    "action": decision.get("action", "PASS") if decision else "ERROR",
                    "confidence": round(decision.get("confidence", 0.0), 2) if decision else 0.0,
                    "reasoning": decision.get("reasoning", "") if decision else "",
                    "regime": candidate.regime.regime,
                })
                continue

            contract: OptionContract = decision["contract"]
            premium_sek = _to_sek_price(contract.mid, contract.underlying, "us")
            risk = validate_option_trade(
                contract=contract,
                premium_sek_per_share=premium_sek,
                profit_target_pct=decision["profit_target_pct"],
                max_loss_pct=decision["max_loss_pct"],
                portfolio_equity=portfolio.equity,
                open_underlyings=portfolio.get_open_underlyings(),
                is_drawdown_mode=portfolio.is_drawdown_mode,
            )
            if not risk.approved:
                logger.info("[%s] %s option risk rejected: %s", track, contract.display_name(), risk.rejection_reason)
                decisions_log.append({
                    "track": track,
                    "ticker": contract.display_name(),
                    "action": "BLOCKED",
                    "confidence": round(decision["confidence"], 2),
                    "reasoning": decision["reasoning"],
                    "rrr": risk.reward_risk or None,
                    "regime": candidate.regime.regime,
                    "reason": risk.rejection_reason,
                })
                continue

            # Fill at mid + adverse half-spread — option spreads are the real friction
            fill_sek = _to_sek_price(contract.mid + (contract.ask - contract.bid) / 2, contract.underlying, "us")
            position = portfolio.open_option(
                contract_symbol=contract.contract_symbol,
                underlying=contract.underlying,
                right=contract.right,
                strike=contract.strike,
                expiry=contract.expiry,
                contracts=risk.contracts,
                entry_premium=fill_sek,
                profit_target_pct=decision["profit_target_pct"],
                max_loss_pct=decision["max_loss_pct"],
                time_stop_dte=decision["time_stop_dte"],
                regime=candidate.regime.regime,
                reasoning=decision["reasoning"],
                confidence=decision["confidence"],
                entry_underlying_price=spot_usd,
                iv_at_entry=contract.implied_vol,
                delta_at_entry=contract.delta,
                technical_snapshot=tech_snapshot,
                entry_inputs=decision.get("entry_inputs", {}),
            )
            if position:
                trade_event = {
                    "track": track,
                    "ticker": contract.display_name(),
                    "action": "BUY",
                    "entry_price": position.entry_premium,
                    "stop_loss": position.premium_stop_level,
                    "target": position.profit_target_level,
                    "confidence": round(decision["confidence"], 2),
                    "reasoning": decision["reasoning"],
                    "rrr": risk.reward_risk,
                    "regime": candidate.regime.regime,
                }
                decisions_log.append(trade_event)
                _emit({"event": "trade_opened", "data": trade_event})

    decisions_log.extend(_manage_holdings())

    logger.info("=== Scan complete: options | %d candidates | %d decisions ===",
                len(candidates), len(decisions_log))
    _mark_and_persist(decisions_log)
    return {
        "market": MARKET_LABEL,
        "candidates": [c.to_dict() for c in candidates],
        "decisions": decisions_log,
    }


def _manage_holdings() -> list[dict]:
    """Refresh premiums for all open option positions and run the mechanical exit
    sweep (profit target / premium stop / time stop). Returns close events."""
    events: list[dict] = []
    for track in settings.options_tracks:
        portfolio = get_options_portfolio(track)
        if not portfolio.open_positions:
            continue

        quotes = _refresh_quotes_sek(portfolio)
        for closed in portfolio.update_premiums(quotes):
            events.append(_emit_option_close(track, closed))
        persist_portfolio(portfolio)
    return events


def run_expiry_sweep() -> dict:
    """Daily post-close job: settle contracts at/past expiry at intrinsic value.
    Long-only, so this is pure bookkeeping — no assignment, no margin."""
    settled: list[dict] = []
    for track in settings.options_tracks:
        portfolio = get_options_portfolio(track)
        expired = portfolio.expired_positions()
        if not expired:
            continue

        for position in expired:
            spot_usd = get_current_price(position.underlying, "us")
            if spot_usd is None:
                logger.warning("[%s] No spot for %s — deferring expiry settlement", track, position.underlying)
                continue
            if position.right == "call":
                intrinsic_usd = max(spot_usd - position.strike, 0.0)
            else:
                intrinsic_usd = max(position.strike - spot_usd, 0.0)
            exit_sek = _to_sek_price(intrinsic_usd, position.underlying, "us") if intrinsic_usd > 0 else 0.0
            reason = "expired_itm" if intrinsic_usd > 0 else "expired_worthless"
            closed = portfolio.close_option(
                position.trade_id, exit_sek, reason, exit_underlying_price=spot_usd,
            )
            if closed:
                settled.append(_emit_option_close(track, closed))
        persist_portfolio(portfolio)

    if settled:
        logger.info("Expiry sweep settled %d position(s)", len(settled))
        _mark_and_persist(settled)
    return {"market": MARKET_LABEL, "settled": settled}


def _refresh_quotes_sek(portfolio: OptionsPortfolio) -> dict[str, float]:
    """One chain call per open (underlying, expiry, right) → {contract_symbol: SEK mid}."""
    groups: dict[tuple, set[str]] = {}
    for p in portfolio.open_positions:
        groups.setdefault((p.underlying, p.expiry, p.right), set()).add(p.contract_symbol)

    quotes_sek: dict[str, float] = {}
    for (underlying, expiry, right), symbols in groups.items():
        usd_quotes = fetch_contract_quotes(underlying, expiry, right, symbols)
        for symbol, usd in usd_quotes.items():
            quotes_sek[symbol] = _to_sek_price(usd, underlying, "us")
    return quotes_sek


def _emit_option_close(track: str, closed: ClosedOptionTrade) -> dict:
    _emit({
        "event": "trade_closed",
        "data": {
            "track": track,
            "ticker": closed.ticker,
            "exit_reason": closed.exit_reason,
            "pnl_pct": round(closed.pnl_pct * 100, 2),
            "pnl": round(closed.pnl, 2),
            "exit_price": closed.exit_premium,
        },
    })
    _trigger_options_erl(track, closed)
    return {
        "track": track,
        "ticker": closed.ticker,
        "action": "SELL",
        "confidence": closed.confidence,
        "reasoning": f"{closed.exit_reason}: P&L {closed.pnl_pct*100:+.1f}% of premium",
        "regime": closed.regime,
    }


def _trigger_options_erl(track: str, closed: ClosedOptionTrade) -> None:
    """ERL on a closed option trade — the underlying-move vs premium-P&L pair is
    the canonical options lesson (right direction, wrong timing/theta)."""
    try:
        trade_dict = closed.to_dict()
        trade_dict["id"] = closed.trade_id
        trade_dict["stop_hit"] = closed.exit_reason == "premium_stop"
        trade_dict["pnl_pct"] = closed.pnl_pct

        dte_consumed = (closed.exit_time - closed.entry_time).days
        option_context = (
            f"\nOPTION TRADE (long {closed.right}): strike ${closed.strike:g}, expiry {closed.expiry}, "
            f"IV at entry {closed.iv_at_entry*100:.1f}%, DTE consumed {dte_consumed}, "
            f"underlying ${closed.entry_underlying_price:.2f} at entry"
            + (f" → ${closed.exit_underlying_price:.2f} at exit" if closed.exit_underlying_price else "")
            + f". P&L is % of premium; planned stop -{closed.max_loss_pct*100:.0f}%, "
            f"target +{closed.profit_target_pct*100:.0f}%."
        )
        entry_inputs = closed.entry_inputs or {}
        vol_ctx = entry_inputs.get("volatility_context", "")
        if vol_ctx:
            option_context += f"\nVolatility at entry: {vol_ctx}"
        run_erl(
            track=track,
            trade=trade_dict,
            technicals_str=closed.technical_snapshot + option_context,
            regime_str=closed.regime,
            news_str=entry_inputs.get("news_summary", ""),
            macro_str=entry_inputs.get("macro_context", ""),
        )
    except Exception as exc:
        logger.error("ERL trigger error for %s option trade %s: %s", track, closed.trade_id, exc)


def _mark_and_persist(decisions_log: list[dict]) -> None:
    scan_loop._recent_decisions[MARKET_LABEL] = {
        "timestamp": datetime.utcnow().isoformat(),
        "decisions": decisions_log,
    }
    _persist_decisions(MARKET_LABEL, decisions_log)
