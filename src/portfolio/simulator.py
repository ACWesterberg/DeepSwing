from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

TrackType = Literal["claude", "gpt"]


@dataclass
class OpenPosition:
    trade_id: int
    ticker: str
    market: str
    quantity: float
    entry_price: float
    stop_loss: float
    target: float
    entry_time: datetime
    trailing_stop: Optional[float] = None
    current_price: float = 0.0
    regime: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    technical_snapshot: str = ""
    sector: str = ""
    entry_inputs: dict = field(default_factory=dict)
    last_news_price: float = 0.0  # price (SEK) at the last news check; drives jump detection

    @property
    def unrealised_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealised_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def market_value(self) -> float:
        return self.current_price * self.quantity

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "market": self.market,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "trailing_stop": self.trailing_stop,
            "entry_time": self.entry_time.isoformat(),
            "unrealised_pnl": round(self.unrealised_pnl, 2),
            "unrealised_pnl_pct": round(self.unrealised_pnl_pct * 100, 2),
            "market_value": round(self.market_value, 2),
        }


@dataclass
class ClosedTrade:
    trade_id: int
    ticker: str
    market: str
    quantity: float
    entry_price: float
    exit_price: float
    stop_loss: float
    target: float
    entry_time: datetime
    exit_time: datetime
    regime: str
    reasoning: str
    confidence: float
    exit_reason: str  # "stop_loss" | "take_profit" | "trailing_stop" | "manual"
    technical_snapshot: str = ""
    entry_inputs: dict = field(default_factory=dict)

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.quantity

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price

    @property
    def rrr_achieved(self) -> float:
        risk = self.entry_price - self.stop_loss
        reward = self.exit_price - self.entry_price
        return reward / risk if risk > 0 else 0.0

    @property
    def duration_days(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 86400

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "market": self.market,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct * 100, 2),
            "rrr_achieved": round(self.rrr_achieved, 2),
            "exit_reason": self.exit_reason,
            "duration_days": round(self.duration_days, 2),
            "regime": self.regime,
            "confidence": self.confidence,
        }


class Portfolio:
    """
    In-memory paper trading portfolio for one simulation track.
    Persists to SQLite via the db layer (called externally).
    """

    def __init__(self, track: TrackType):
        self.track = track
        self.cash = settings.starting_capital_sek
        self.starting_equity = settings.starting_capital_sek
        self.peak_equity = settings.starting_capital_sek
        self.open_positions: list[OpenPosition] = []
        self.closed_trades: list[ClosedTrade] = []
        self.total_commission: float = 0.0
        self._next_trade_id = 1

    @property
    def equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.open_positions)

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak equity (0–1)."""
        if self.peak_equity == 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    @property
    def is_drawdown_mode(self) -> bool:
        return self.drawdown >= settings.drawdown_pause_threshold

    @property
    def can_open_new_position(self) -> bool:
        """False once free cash is too small to fund a new position — treated as fully allocated."""
        return self.cash >= settings.min_cash_for_new_position_pct * self.equity

    def open_trade(
        self,
        ticker: str,
        market: str,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        target: float,
        regime: str,
        reasoning: str,
        confidence: float,
        technical_snapshot: str = "",
        sector: str = "",
        entry_inputs: Optional[dict] = None,
    ) -> Optional[OpenPosition]:
        # Apply simulated slippage (adverse, so price moves against us)
        filled_price = entry_price * (1 + settings.simulated_slippage)
        cost = filled_price * quantity
        commission = cost * settings.commission_pct
        if market == "us":
            commission += cost * settings.fx_commission_pct

        if cost + commission > self.cash:
            logger.warning(
                "[%s] Insufficient cash for %s: need %.2f, have %.2f",
                self.track, ticker, cost + commission, self.cash,
            )
            return None

        self.cash -= cost + commission
        self.total_commission += commission
        position = OpenPosition(
            trade_id=self._next_trade_id,
            ticker=ticker,
            market=market,
            quantity=quantity,
            entry_price=filled_price,
            stop_loss=stop_loss,
            target=target,
            entry_time=datetime.utcnow(),
            current_price=filled_price,
            regime=regime,
            reasoning=reasoning,
            confidence=confidence,
            technical_snapshot=technical_snapshot,
            sector=sector,
            entry_inputs=entry_inputs or {},
            last_news_price=filled_price,
        )
        self.open_positions.append(position)
        self._next_trade_id += 1

        logger.info(
            "[%s] OPENED %s @ %.4f (qty=%.2f, stop=%.4f, target=%.4f)",
            self.track, ticker, filled_price, quantity, stop_loss, target,
        )
        return position

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str,
        regime: str = "",
        reasoning: str = "",
        confidence: float = 0.0,
    ) -> Optional[ClosedTrade]:
        position = next((p for p in self.open_positions if p.trade_id == trade_id), None)
        if position is None:
            logger.warning("[%s] No open position with trade_id=%s", self.track, trade_id)
            return None

        # Apply slippage (adverse for exit)
        filled_price = exit_price * (1 - settings.simulated_slippage)
        proceeds = filled_price * position.quantity
        commission = proceeds * settings.commission_pct
        if position.market == "us":
            commission += proceeds * settings.fx_commission_pct
        self.cash += proceeds - commission
        self.total_commission += commission
        self.open_positions.remove(position)

        closed = ClosedTrade(
            trade_id=trade_id,
            ticker=position.ticker,
            market=position.market,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=filled_price,
            stop_loss=position.stop_loss,
            target=position.target,
            entry_time=position.entry_time,
            exit_time=datetime.utcnow(),
            regime=regime or position.regime,
            reasoning=reasoning or position.reasoning,
            confidence=confidence or position.confidence,
            exit_reason=exit_reason,
            technical_snapshot=position.technical_snapshot,
            entry_inputs=position.entry_inputs,
        )
        self.closed_trades.append(closed)

        # Update peak equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        logger.info(
            "[%s] CLOSED %s @ %.4f (P&L=%.2f%%, reason=%s)",
            self.track, position.ticker, filled_price,
            closed.pnl_pct * 100, exit_reason,
        )
        return closed

    def update_prices(self, prices: dict[str, float]) -> list[ClosedTrade]:
        """
        Update current prices for all open positions.
        Checks stop-loss and take-profit, auto-closes if hit.
        Returns list of any trades closed by this update.
        """
        closed_this_update: list[ClosedTrade] = []

        for position in list(self.open_positions):
            price = prices.get(position.ticker)
            if price is None:
                continue

            position.current_price = price

            # Update trailing stop (Parabolic SAR approximation: step up by 2% of price)
            if position.trailing_stop is None:
                position.trailing_stop = position.stop_loss
            if price > position.entry_price and price * 0.98 > position.trailing_stop:
                position.trailing_stop = price * 0.98

            effective_stop = max(position.stop_loss, position.trailing_stop or 0)

            if price <= effective_stop:
                closed = self.close_trade(position.trade_id, price, "stop_loss")
                if closed:
                    closed_this_update.append(closed)
            elif price >= position.target:
                closed = self.close_trade(position.trade_id, price, "take_profit")
                if closed:
                    closed_this_update.append(closed)

        return closed_this_update

    def snapshot(self) -> dict:
        return {
            "track": self.track,
            "equity": round(self.equity, 2),
            "cash": round(self.cash, 2),
            "open_positions_value": round(sum(p.market_value for p in self.open_positions), 2),
            "open_positions_count": len(self.open_positions),
            "total_trades": len(self.closed_trades),
            "total_commission": round(self.total_commission, 2),
            "drawdown_pct": round(self.drawdown * 100, 2),
            "is_drawdown_mode": self.is_drawdown_mode,
        }

    def get_open_tickers(self) -> list[str]:
        return [p.ticker for p in self.open_positions]


_portfolios: dict[str, Portfolio] = {}


def get_portfolio(track: TrackType) -> Portfolio:
    if track not in _portfolios:
        _portfolios[track] = Portfolio(track)
    return _portfolios[track]


def reset_portfolios() -> None:
    """Clear all portfolio state. Used in tests."""
    _portfolios.clear()
