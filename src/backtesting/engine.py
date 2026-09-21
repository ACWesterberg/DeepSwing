from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import settings
from src.agent.risk import compute_return_correlations, validate_trade
from src.analysis.regime import classify_regime
from src.analysis.screener import screen_candidates
from src.analysis.technical import TechnicalSignals, compute_signals
from src.portfolio.daily import BacktestTrade, DailyPortfolio as _SimPortfolio

logger = logging.getLogger(__name__)

# Warmup period: minimum rows needed for all indicators (SMA200 + buffer)
_WARMUP_DAYS = 220


@dataclass
class WindowResult:
    window_index: int
    start: date
    end: date
    trades: list[BacktestTrade] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "window": self.window_index,
            "start": str(self.start),
            "end": str(self.end),
            "metrics": self.metrics,
            "trades": [t.to_dict() for t in self.trades],
        }


@dataclass
class BacktestResult:
    market: str
    start: date
    end: date
    n_windows: int
    initial_equity: float
    windows: list[WindowResult]
    overall_metrics: dict

    def to_dict(self) -> dict:
        return {
            "market": self.market,
            "start": str(self.start),
            "end": str(self.end),
            "n_windows": self.n_windows,
            "initial_equity": self.initial_equity,
            "overall_metrics": self.overall_metrics,
            "windows": [w.to_dict() for w in self.windows],
        }


class BacktestEngine:
    """
    Walk-forward backtesting engine.

    Splits [start, end] into n_windows equal chunks and runs the strategy
    on each independently. Uses real technical analysis, screening, and risk
    rules. Decision layer is replaced by ATR-based stops/targets (no AI calls).

    Mirrors live execution: slippage + commissions from settings, ATR-scaled
    trailing stop, intraday High/Low stop/target checks (stop-first when both
    trade in one bar, gaps fill at the open), mark-to-market equity.

    No look-ahead bias: on each simulated day d, only data up to and including
    d is used for indicator computation, and the trailing stop is raised from
    d's close only after d's exit checks.
    """

    def __init__(
        self,
        market: str,
        tickers: list[str],
        start: date,
        end: date,
        initial_equity: float = 100_000.0,
        n_windows: int = 1,
    ):
        self.market = market
        self.tickers = tickers
        self.start = start
        self.end = end
        self.initial_equity = initial_equity
        self.n_windows = max(1, n_windows)

    def run(self) -> BacktestResult:
        logger.info(
            "Backtest starting: market=%s, %s → %s, %d window(s), %d tickers",
            self.market, self.start, self.end, self.n_windows, len(self.tickers),
        )

        # Load data with warmup buffer so indicators have enough history on day 1
        load_start = self.start - timedelta(days=_WARMUP_DAYS * 1.5)
        ohlcv_map = self._load_data(load_start, self.end)
        if not ohlcv_map:
            logger.warning("No OHLCV data loaded for backtest")
            return BacktestResult(
                market=self.market, start=self.start, end=self.end,
                n_windows=0, initial_equity=self.initial_equity,
                windows=[], overall_metrics=_empty_metrics(),
            )

        windows = self._split_windows()
        window_results = []
        for i, (w_start, w_end) in enumerate(windows):
            logger.info("Running window %d/%d: %s → %s", i + 1, self.n_windows, w_start, w_end)
            result = self._simulate_window(ohlcv_map, i, w_start, w_end)
            window_results.append(result)

        all_trades = [t for w in window_results for t in w.trades]
        overall = _compute_metrics(all_trades, self.initial_equity * self.n_windows)

        return BacktestResult(
            market=self.market,
            start=self.start,
            end=self.end,
            n_windows=self.n_windows,
            initial_equity=self.initial_equity,
            windows=window_results,
            overall_metrics=overall,
        )

    # ------------------------------------------------------------------

    def _load_data(self, load_start: date, load_end: date) -> dict[str, pd.DataFrame]:
        yf_tickers = [
            t.replace(".STO", ".ST") if self.market == "nordic" else t
            for t in self.tickers
        ]
        ticker_map = dict(zip(yf_tickers, self.tickers))
        chunk_size = max(1, settings.ohlcv_batch_chunk_size)
        result: dict[str, pd.DataFrame] = {}

        for i in range(0, len(yf_tickers), chunk_size):
            chunk = yf_tickers[i : i + chunk_size]
            try:
                raw = yf.download(
                    chunk,
                    start=load_start.isoformat(),
                    end=(load_end + timedelta(days=1)).isoformat(),
                    interval="1d",
                    auto_adjust=True,
                    group_by="ticker",
                    progress=False,
                    threads=True,
                )
            except Exception as exc:
                logger.error("yfinance batch download error: %s", exc)
                continue

            if len(chunk) == 1:
                df = _standardize(raw)
                if not df.empty:
                    result[ticker_map[chunk[0]]] = df
                continue

            for yf_ticker in chunk:
                try:
                    df = raw[yf_ticker].dropna(how="all")
                    df = _standardize(df)
                    if not df.empty:
                        result[ticker_map[yf_ticker]] = df
                except Exception:
                    pass

        logger.info("Loaded %d/%d tickers for backtest", len(result), len(self.tickers))
        return result

    def _split_windows(self) -> list[tuple[date, date]]:
        total = (self.end - self.start).days
        chunk = total // self.n_windows
        windows = []
        for i in range(self.n_windows):
            w_start = self.start + timedelta(days=i * chunk)
            w_end = self.end if i == self.n_windows - 1 else self.start + timedelta(days=(i + 1) * chunk)
            windows.append((w_start, w_end))
        return windows

    def _simulate_window(
        self, ohlcv_map: dict[str, pd.DataFrame], window_idx: int, w_start: date, w_end: date
    ) -> WindowResult:
        commission_rate = settings.commission_pct + (
            settings.fx_commission_pct if self.market in ("us", "eu") else 0.0
        )
        portfolio = _SimPortfolio(
            self.initial_equity,
            commission_rate=commission_rate,
            slippage=settings.simulated_slippage,
        )

        # Build a sorted list of trading days within this window
        sample_df = next(iter(ohlcv_map.values()))
        trading_days = sorted(
            d.date()
            for d in sample_df.index
            if w_start <= d.date() <= w_end
        )

        if not trading_days:
            return WindowResult(window_index=window_idx, start=w_start, end=w_end,
                                metrics=_empty_metrics())

        for day in trading_days:
            bars = _get_bars_for_day(ohlcv_map, portfolio.open_tickers, day)
            portfolio.update(bars, day)
            # Build analysis_map using only data up to this day (no look-ahead)
            analysis_map: dict = {}
            slices: dict[str, pd.DataFrame] = {}
            for ticker, df in ohlcv_map.items():
                df_slice = df[df.index.date <= day]
                if len(df_slice) < _WARMUP_DAYS:
                    continue
                slices[ticker] = df_slice
                signals = compute_signals(ticker, df_slice)
                if signals is None:
                    continue
                regime = classify_regime(df_slice)
                analysis_map[ticker] = (signals, regime)

            candidates = screen_candidates(analysis_map, self.market)

            for candidate in candidates:
                if portfolio.has_ticker(candidate.ticker):
                    continue

                entry = candidate.signals.current_price
                stop = entry - settings.atr_stop_multiplier * candidate.signals.atr_14
                target = entry + settings.min_rrr * (entry - stop)

                open_pos_info = [{"ticker": t, "sector": ""} for t in portfolio.open_tickers]
                correlations = compute_return_correlations(
                    slices.get(candidate.ticker), portfolio.open_tickers, slices,
                )
                risk = validate_trade(
                    action="BUY",
                    entry_price=entry,
                    stop_loss=stop,
                    target=target,
                    portfolio_equity=portfolio.equity,
                    open_positions=open_pos_info,
                    signals=candidate.signals,
                    is_drawdown_mode=portfolio.is_drawdown_mode,
                    available_cash=portfolio.cash,
                    position_correlations=correlations,
                    market=self.market,
                )

                if risk.approved:
                    portfolio.open_position(
                        candidate.ticker, entry, stop, target, risk.quantity, day,
                        trail_distance=settings.trailing_stop_atr_multiplier * candidate.signals.atr_14,
                    )

        # Close any positions still open at window end
        last_prices = _get_prices_for_day(ohlcv_map, portfolio.open_tickers, trading_days[-1])
        for ticker in list(portfolio.open_tickers):
            price = last_prices.get(ticker, 0.0)
            if price > 0:
                portfolio.close_position(ticker, price, trading_days[-1], "end_of_window")

        metrics = _compute_metrics(portfolio.closed_trades, self.initial_equity)
        return WindowResult(
            window_index=window_idx,
            start=w_start,
            end=w_end,
            trades=portfolio.closed_trades,
            metrics=metrics,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: c.capitalize() for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")}
    df = df.rename(columns=rename)
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    return df[cols].dropna(how="all").copy()


def _get_prices_for_day(ohlcv_map: dict[str, pd.DataFrame], tickers: list[str], day: date) -> dict[str, float]:
    prices: dict[str, float] = {}
    for ticker in tickers:
        df = ohlcv_map.get(ticker)
        if df is None:
            continue
        rows = df[df.index.date == day]
        if not rows.empty:
            prices[ticker] = float(rows["Close"].iloc[-1])
    return prices


def _get_bars_for_day(ohlcv_map: dict[str, pd.DataFrame], tickers: list[str], day: date) -> dict[str, dict]:
    """Full OHLC bar per ticker for the day (Close backfills missing columns)."""
    bars: dict[str, dict] = {}
    for ticker in tickers:
        df = ohlcv_map.get(ticker)
        if df is None:
            continue
        rows = df[df.index.date == day]
        if rows.empty:
            continue
        row = rows.iloc[-1]
        close = float(row["Close"])
        bars[ticker] = {
            "open": float(row.get("Open", close)),
            "high": float(row.get("High", close)),
            "low": float(row.get("Low", close)),
            "close": close,
        }
    return bars


def _compute_metrics(trades: list[BacktestTrade], initial_equity: float) -> dict:
    closed = [t for t in trades if t.exit_price is not None]
    if not closed:
        return _empty_metrics()

    pnl_pcts = [t.pnl_pct for t in closed]
    wins = [t for t in closed if t.net_pnl > 0]
    win_rate = len(wins) / len(closed)
    avg_rrr = float(np.mean([t.rrr_achieved for t in closed]))
    total_commission = sum(t.commission for t in trades)
    total_pnl = sum(t.net_pnl for t in trades)
    total_return = total_pnl / initial_equity if initial_equity > 0 else 0.0

    # Sharpe from per-trade returns, annualized by actual holding period —
    # ×√252 on multi-day trades would overstate it several-fold
    if len(pnl_pcts) >= 2:
        durations = [
            max((t.exit_date - t.entry_date).days, 1)
            for t in closed
            if t.exit_date is not None
        ]
        avg_duration = float(np.mean(durations)) if durations else 1.0
        periods_per_year = 252.0 / max(avg_duration, 1.0)
        arr = np.array(pnl_pcts)
        std = np.std(arr, ddof=1)
        sharpe = float(np.mean(arr) / std * (periods_per_year ** 0.5)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown from cumulative equity curve
    equity_curve = [initial_equity]
    for t in sorted(trades, key=lambda x: x.exit_date or date.max):
        equity_curve.append(equity_curve[-1] + t.net_pnl)
    peak = equity_curve[0]
    max_dd = 0.0
    for e in equity_curve:
        if e > peak:
            peak = e
        dd = (peak - e) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    return {
        "total_trades": len(closed),
        "win_rate": round(win_rate, 4),
        "avg_rrr": round(avg_rrr, 2),
        "total_return_pct": round(total_return * 100, 2),
        "total_pnl": round(total_pnl, 2),
        "total_commission": round(total_commission, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "optimization_metric": round(avg_rrr, 4),
    }


def _empty_metrics() -> dict:
    return {
        "total_trades": 0, "win_rate": 0.0, "avg_rrr": 0.0,
        "total_return_pct": 0.0, "total_pnl": 0.0, "total_commission": 0.0,
        "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0,
        "optimization_metric": 0.0,
    }
