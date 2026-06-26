from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import settings
from src.agent.risk import validate_trade
from src.analysis.regime import classify_regime
from src.analysis.screener import screen_candidates
from src.analysis.technical import TechnicalSignals, compute_signals

logger = logging.getLogger(__name__)

# Warmup period: minimum rows needed for all indicators (SMA200 + buffer)
_WARMUP_DAYS = 220


@dataclass
class BacktestTrade:
    ticker: str
    entry_date: date
    entry_price: float
    exit_date: Optional[date]
    exit_price: Optional[float]
    exit_reason: str    # "stop_loss" | "take_profit" | "end_of_window" | "open"
    stop_loss: float
    target: float
    quantity: float

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.quantity

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0 or self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price

    @property
    def rrr_achieved(self) -> float:
        risk = self.entry_price - self.stop_loss
        if risk <= 0 or self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) / risk

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "entry_date": str(self.entry_date),
            "entry_price": round(self.entry_price, 4),
            "exit_date": str(self.exit_date) if self.exit_date else None,
            "exit_price": round(self.exit_price, 4) if self.exit_price else None,
            "exit_reason": self.exit_reason,
            "stop_loss": round(self.stop_loss, 4),
            "target": round(self.target, 4),
            "quantity": round(self.quantity, 4),
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct * 100, 2),
            "rrr_achieved": round(self.rrr_achieved, 2),
        }


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


class _SimPortfolio:
    """Lightweight in-memory portfolio for backtesting. Not connected to live DB."""

    def __init__(self, initial_equity: float):
        self.initial_equity = initial_equity
        self.cash = initial_equity
        self.peak_equity = initial_equity
        self._positions: dict[str, BacktestTrade] = {}  # ticker → open trade
        self.closed_trades: list[BacktestTrade] = []

    @property
    def open_equity(self) -> float:
        return sum(
            (p.entry_price * p.quantity) for p in self._positions.values()
        )

    @property
    def equity(self) -> float:
        return self.cash + self.open_equity

    @property
    def is_drawdown_mode(self) -> bool:
        if self.peak_equity == 0:
            return False
        return (self.peak_equity - self.equity) / self.peak_equity >= settings.drawdown_pause_threshold

    @property
    def open_tickers(self) -> list[str]:
        return list(self._positions.keys())

    def has_ticker(self, ticker: str) -> bool:
        return ticker in self._positions

    def open_position(
        self,
        ticker: str,
        entry_price: float,
        stop_loss: float,
        target: float,
        quantity: float,
        entry_date: date,
    ) -> None:
        cost = entry_price * quantity
        if cost > self.cash:
            return
        self.cash -= cost
        self._positions[ticker] = BacktestTrade(
            ticker=ticker,
            entry_date=entry_date,
            entry_price=entry_price,
            exit_date=None,
            exit_price=None,
            exit_reason="open",
            stop_loss=stop_loss,
            target=target,
            quantity=quantity,
        )

    def update(self, prices: dict[str, float], today: date) -> None:
        for ticker in list(self._positions):
            price = prices.get(ticker)
            if price is None:
                continue
            pos = self._positions[ticker]
            if price <= pos.stop_loss:
                self._close(ticker, price, today, "stop_loss")
            elif price >= pos.target:
                self._close(ticker, price, today, "take_profit")

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def close_position(self, ticker: str, price: float, today: date, reason: str) -> None:
        if ticker in self._positions:
            self._close(ticker, price, today, reason)

    def _close(self, ticker: str, price: float, today: date, reason: str) -> None:
        pos = self._positions.pop(ticker)
        pos.exit_date = today
        pos.exit_price = price
        pos.exit_reason = reason
        self.cash += price * pos.quantity
        self.closed_trades.append(pos)


class BacktestEngine:
    """
    Walk-forward backtesting engine.

    Splits [start, end] into n_windows equal chunks and runs the strategy
    on each independently. Uses real technical analysis, screening, and risk
    rules. Decision layer is replaced by ATR-based stops/targets (no AI calls).

    No look-ahead bias: on each simulated day d, only data up to and including
    d is used for indicator computation.
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

        try:
            raw = yf.download(
                yf_tickers,
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
            return {}

        result: dict[str, pd.DataFrame] = {}
        if len(yf_tickers) == 1:
            df = _standardize(raw)
            if not df.empty:
                result[self.tickers[0]] = df
        else:
            for yf_ticker, orig_ticker in ticker_map.items():
                try:
                    df = raw[yf_ticker].dropna(how="all")
                    df = _standardize(df)
                    if not df.empty:
                        result[orig_ticker] = df
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
        portfolio = _SimPortfolio(self.initial_equity)

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
            # Build analysis_map using only data up to this day (no look-ahead)
            analysis_map: dict = {}
            for ticker, df in ohlcv_map.items():
                df_slice = df[df.index.date <= day]
                if len(df_slice) < _WARMUP_DAYS:
                    continue
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
                # Target: RRR of 2.5 (above the 2.0 minimum)
                target = entry + 2.5 * (entry - stop)

                open_pos_info = [{"ticker": t, "sector": ""} for t in portfolio.open_tickers]
                risk = validate_trade(
                    action="BUY",
                    entry_price=entry,
                    stop_loss=stop,
                    target=target,
                    portfolio_equity=portfolio.equity,
                    open_positions=open_pos_info,
                    signals=candidate.signals,
                    is_drawdown_mode=portfolio.is_drawdown_mode,
                )

                if risk.approved:
                    portfolio.open_position(candidate.ticker, entry, stop, target, risk.quantity, day)

            # Update prices at today's close
            current_prices = _get_prices_for_day(ohlcv_map, portfolio.open_tickers, day)
            portfolio.update(current_prices, day)

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


def _compute_metrics(trades: list[BacktestTrade], initial_equity: float) -> dict:
    closed = [t for t in trades if t.exit_price is not None and t.exit_reason != "end_of_window"]
    if not closed:
        return _empty_metrics()

    pnl_pcts = [t.pnl_pct for t in closed]
    wins = [t for t in closed if t.pnl > 0]
    win_rate = len(wins) / len(closed)
    avg_rrr = float(np.mean([t.rrr_achieved for t in closed]))
    total_pnl = sum(t.pnl for t in trades)
    total_return = total_pnl / initial_equity if initial_equity > 0 else 0.0

    # Approximate Sharpe from daily returns
    if len(pnl_pcts) >= 2:
        arr = np.array(pnl_pcts)
        sharpe = float(np.mean(arr) / np.std(arr, ddof=1)) * (252 ** 0.5) if np.std(arr) > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown from cumulative equity curve
    equity_curve = [initial_equity]
    for t in sorted(trades, key=lambda x: x.exit_date or date.max):
        equity_curve.append(equity_curve[-1] + t.pnl)
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
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "optimization_metric": round(win_rate * avg_rrr, 4),
    }


def _empty_metrics() -> dict:
    return {
        "total_trades": 0, "win_rate": 0.0, "avg_rrr": 0.0,
        "total_return_pct": 0.0, "total_pnl": 0.0,
        "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0,
        "optimization_metric": 0.0,
    }
