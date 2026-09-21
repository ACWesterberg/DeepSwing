from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Optional

from config.settings import settings
from src.portfolio.simulator import breakeven_from_costs, resolve_exit


@dataclass
class BacktestTrade:
    ticker: str
    entry_date: date
    entry_price: float
    exit_date: Optional[date]
    exit_price: Optional[float]
    exit_reason: str    # stop_loss | breakeven_stop | trailing_stop | take_profit | end_of_window | open
    stop_loss: float
    target: float
    quantity: float
    trail_distance: float = 0.0
    trailing_stop: Optional[float] = None
    current_price: float = 0.0
    commission: float = 0.0   # entry + exit, accumulated at fill time
    breakeven_armed: bool = False

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.quantity

    @property
    def net_pnl(self) -> float:
        return self.pnl - self.commission

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0 or self.exit_price is None:
            return 0.0
        return self.net_pnl / (self.entry_price * self.quantity) if self.quantity > 0 else 0.0

    @property
    def rrr_achieved(self) -> float:
        risk = self.entry_price - self.stop_loss
        if risk <= 0 or self.exit_price is None:
            return 0.0
        return self.net_pnl / (risk * self.quantity) if self.quantity > 0 else 0.0

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
            "pnl": round(self.net_pnl, 2),
            "commission": round(self.commission, 2),
            "pnl_pct": round(self.pnl_pct * 100, 2),
            "rrr_achieved": round(self.rrr_achieved, 2),
        }


class DailyPortfolio:
    """Shared daily-bar execution for backtests and counterfactual outcomes."""

    def __init__(self, initial_equity: float, commission_rate: float = 0.0, slippage: float = 0.0,
                 *, trailing_atr_multiplier: float | None = None,
                 breakeven_arm_atr_multiplier: float | None = None):
        self.initial_equity = initial_equity
        self.cash = initial_equity
        self.peak_equity = initial_equity
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.trailing_atr_multiplier = (settings.trailing_stop_atr_multiplier
                                        if trailing_atr_multiplier is None else trailing_atr_multiplier)
        self.breakeven_arm_atr_multiplier = (settings.breakeven_arm_atr_multiplier
                                           if breakeven_arm_atr_multiplier is None else breakeven_arm_atr_multiplier)
        self.total_commission = 0.0
        self._positions: dict[str, BacktestTrade] = {}  # ticker → open trade
        self.closed_trades: list[BacktestTrade] = []

    @property
    def open_equity(self) -> float:
        # Mark-to-market — valuing at entry price hides open P&L from
        # equity/drawdown and made drawdown mode fire on the wrong days
        return sum(
            (p.current_price or p.entry_price) * p.quantity
            for p in self._positions.values()
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

    def _breakeven_floor(self, pos: "BacktestTrade") -> float:
        if not pos.breakeven_armed:
            return 0.0
        return breakeven_from_costs(pos.entry_price, self.commission_rate, self.slippage)

    def open_position(
        self,
        ticker: str,
        entry_price: float,
        stop_loss: float,
        target: float,
        quantity: float,
        entry_date: date,
        trail_distance: float = 0.0,
    ) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (entry_price, stop_loss, target, quantity)):
            raise ValueError("Invalid daily execution values")
        if not stop_loss < entry_price < target:
            raise ValueError("Invalid daily trade levels")
        fill = entry_price * (1 + self.slippage)
        cost = fill * quantity
        commission = cost * self.commission_rate
        if cost + commission > self.cash:
            return
        self.cash -= cost + commission
        self.total_commission += commission
        self._positions[ticker] = BacktestTrade(
            ticker=ticker,
            entry_date=entry_date,
            entry_price=fill,
            exit_date=None,
            exit_price=None,
            exit_reason="open",
            stop_loss=stop_loss,
            target=target,
            quantity=quantity,
            trail_distance=trail_distance,
            current_price=fill,
            commission=commission,
        )

    def update(self, bars: dict, today: date) -> None:
        """Advance one day. Values may be a plain close price (float) or a full
        {open, high, low, close} bar — exits check the intraday High/Low so a
        stop that traded through mid-day actually fires."""
        for ticker in list(self._positions):
            bar = bars.get(ticker)
            if bar is None:
                continue
            if isinstance(bar, dict):
                o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
            else:
                o = h = l = c = float(bar)

            if not all(math.isfinite(v) and v > 0 for v in (o, h, l, c)):
                raise ValueError(f"Invalid OHLC bar for {ticker}")
            pos = self._positions[ticker]
            pos.current_price = c

            breakeven_floor = self._breakeven_floor(pos)
            effective_stop = max(pos.stop_loss, breakeven_floor, pos.trailing_stop or 0.0)
            # The open has known precedence; high/low order after it is unknown.
            if o >= pos.target:
                self._close(ticker, o, today, "take_profit")
                continue
            reason = resolve_exit(
                l, pos.entry_price, pos.stop_loss, pos.trailing_stop, breakeven_floor
            )
            if reason is not None:
                fill = min(o, effective_stop)  # gap below the stop fills at the open
                self._close(ticker, fill, today, reason)
                continue
            if h >= pos.target:
                fill = max(o, pos.target)      # gap above the target fills at the open
                self._close(ticker, fill, today, "take_profit")
                continue

            # Trail and arm from the close AFTER exit checks — the intraday
            # ordering of high vs low is unknown, so today's high must not be
            # allowed to both raise the stop and trigger it (look-ahead).
            if pos.trail_distance > 0 and c > pos.entry_price:
                candidate_stop = c - pos.trail_distance
                if candidate_stop > (pos.trailing_stop or pos.stop_loss):
                    pos.trailing_stop = candidate_stop
            atr = (
                pos.trail_distance / self.trailing_atr_multiplier
                if self.trailing_atr_multiplier > 0
                else 0.0
            )
            arm_at = self.breakeven_arm_atr_multiplier
            if arm_at > 0 and atr > 0 and not pos.breakeven_armed:
                if c >= pos.entry_price + arm_at * atr:
                    pos.breakeven_armed = True

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def close_position(self, ticker: str, price: float, today: date, reason: str) -> None:
        if ticker in self._positions:
            self._close(ticker, price, today, reason)

    def _close(self, ticker: str, price: float, today: date, reason: str) -> None:
        pos = self._positions.pop(ticker)
        fill = price * (1 - self.slippage)
        proceeds = fill * pos.quantity
        commission = proceeds * self.commission_rate
        pos.exit_date = today
        pos.exit_price = fill
        pos.exit_reason = reason
        pos.current_price = fill
        pos.commission += commission
        self.cash += proceeds - commission
        self.total_commission += commission
        self.closed_trades.append(pos)
