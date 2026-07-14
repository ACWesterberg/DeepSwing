from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Literal, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

TrackType = Literal["claude", "gpt"]


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


def _parse_dt(value) -> Optional[datetime]:
    if value is None:
        return None
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


# Optional persistence hook — set by main.py to mirror state to the DB on every
# open/close. Defaults to off, so simulator has no DB dependency in tests.
_persist_handler: Optional[Callable[["Portfolio"], None]] = None


def set_persistence_handler(fn: Optional[Callable[["Portfolio"], None]]) -> None:
    global _persist_handler
    _persist_handler = fn


def persist_portfolio(portfolio: "Portfolio") -> None:
    """Best-effort persist of a portfolio's state; never raises into a scan."""
    if _persist_handler is None:
        return
    try:
        _persist_handler(portfolio)
    except Exception as exc:
        logger.warning("Portfolio persistence error [%s]: %s", portfolio.track, exc)


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
    trail_distance: float = 0.0   # trailing-stop distance in SEK (ATR-scaled at entry)
    entry_commission: float = 0.0  # SEK paid at open; folded into net P&L at close
    entry_fx_rate: float = 0.0     # native→SEK rate at entry; lets ERL see FX-driven P&L

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
        from src.data.universe import get_exchange_for_ticker

        return {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "market": self.market,
            "exchange": get_exchange_for_ticker(self.ticker, self.market),
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

    def to_state(self) -> dict:
        """Full, lossless serialization for persistence."""
        return {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "market": self.market,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "entry_time": _iso(self.entry_time),
            "trailing_stop": self.trailing_stop,
            "current_price": self.current_price,
            "regime": self.regime,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "technical_snapshot": self.technical_snapshot,
            "sector": self.sector,
            "entry_inputs": self.entry_inputs,
            "last_news_price": self.last_news_price,
            "trail_distance": self.trail_distance,
            "entry_commission": self.entry_commission,
            "entry_fx_rate": self.entry_fx_rate,
        }

    @classmethod
    def from_state(cls, d: dict) -> "OpenPosition":
        return cls(
            trade_id=d["trade_id"],
            ticker=d["ticker"],
            market=d["market"],
            quantity=d["quantity"],
            entry_price=d["entry_price"],
            stop_loss=d["stop_loss"],
            target=d["target"],
            entry_time=_parse_dt(d.get("entry_time")) or datetime.utcnow(),
            trailing_stop=d.get("trailing_stop"),
            current_price=d.get("current_price", 0.0),
            regime=d.get("regime", ""),
            reasoning=d.get("reasoning", ""),
            confidence=d.get("confidence", 0.0),
            technical_snapshot=d.get("technical_snapshot", ""),
            sector=d.get("sector", ""),
            entry_inputs=d.get("entry_inputs") or {},
            last_news_price=d.get("last_news_price", 0.0),
            trail_distance=d.get("trail_distance", 0.0),
            entry_commission=d.get("entry_commission", 0.0),
            entry_fx_rate=d.get("entry_fx_rate", 0.0),
        )


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
    commission: float = 0.0     # entry + exit commission (SEK); 0 for pre-upgrade trades
    entry_fx_rate: float = 0.0  # native→SEK rate at entry (0 when unknown)

    @property
    def pnl(self) -> float:
        """Net P&L in SEK — slippage lives in the fill prices, commissions here."""
        return (self.exit_price - self.entry_price) * self.quantity - self.commission

    @property
    def pnl_pct(self) -> float:
        cost_basis = self.entry_price * self.quantity
        if cost_basis == 0:
            return 0.0
        return self.pnl / cost_basis

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
            "commission": round(self.commission, 2),
            "rrr_achieved": round(self.rrr_achieved, 2),
            "exit_reason": self.exit_reason,
            "duration_days": round(self.duration_days, 2),
            "regime": self.regime,
            "confidence": self.confidence,
        }

    def to_state(self) -> dict:
        """Full, lossless serialization for persistence."""
        return {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "market": self.market,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "entry_time": _iso(self.entry_time),
            "exit_time": _iso(self.exit_time),
            "regime": self.regime,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "exit_reason": self.exit_reason,
            "technical_snapshot": self.technical_snapshot,
            "entry_inputs": self.entry_inputs,
            "commission": self.commission,
            "entry_fx_rate": self.entry_fx_rate,
        }

    @classmethod
    def from_state(cls, d: dict) -> "ClosedTrade":
        return cls(
            trade_id=d["trade_id"],
            ticker=d["ticker"],
            market=d["market"],
            quantity=d["quantity"],
            entry_price=d["entry_price"],
            exit_price=d["exit_price"],
            stop_loss=d["stop_loss"],
            target=d["target"],
            entry_time=_parse_dt(d.get("entry_time")) or datetime.utcnow(),
            exit_time=_parse_dt(d.get("exit_time")) or datetime.utcnow(),
            regime=d.get("regime", ""),
            reasoning=d.get("reasoning", ""),
            confidence=d.get("confidence", 0.0),
            exit_reason=d.get("exit_reason", ""),
            technical_snapshot=d.get("technical_snapshot", ""),
            entry_inputs=d.get("entry_inputs") or {},
            commission=d.get("commission", 0.0),
            entry_fx_rate=d.get("entry_fx_rate", 0.0),
        )


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

    def market_exposure(self, market: str) -> float:
        """Total current value of open positions in a given market (SEK)."""
        return sum(p.market_value for p in self.open_positions if p.market == market)

    def market_budget_remaining(self, market: str) -> float:
        """SEK still investable in `market` under its allocation cap (never negative).
        A market with no configured cap (or a cap >= 1.0) is limited only by cash."""
        cap = settings.market_allocation.get(market)
        if cap is None or cap >= 1.0:
            return self.cash
        allowed = cap * self.equity
        return max(0.0, min(self.cash, allowed - self.market_exposure(market)))

    def can_open_in_market(self, market: str) -> bool:
        """True when this track has enough investable headroom in `market` — both
        free cash and remaining market-allocation budget — to fund a new position."""
        return self.market_budget_remaining(market) >= settings.min_cash_for_new_position_pct * self.equity

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
        trail_distance: float = 0.0,
        entry_fx_rate: float = 0.0,
    ) -> Optional[OpenPosition]:
        # Apply simulated slippage (adverse, so price moves against us)
        filled_price = entry_price * (1 + settings.simulated_slippage)
        cost = filled_price * quantity
        commission = cost * settings.commission_pct
        if market in ("us", "eu"):
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
            trail_distance=trail_distance,
            entry_commission=commission,
            entry_fx_rate=entry_fx_rate,
        )
        self.open_positions.append(position)
        self._next_trade_id += 1

        logger.info(
            "[%s] OPENED %s @ %.4f (qty=%.2f, stop=%.4f, target=%.4f)",
            self.track, ticker, filled_price, quantity, stop_loss, target,
        )
        persist_portfolio(self)
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
        if position.market in ("us", "eu"):
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
            commission=position.entry_commission + commission,
            entry_fx_rate=position.entry_fx_rate,
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
        persist_portfolio(self)
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

            # Trail by the position's ATR-scaled distance once in profit; the 2%
            # fallback covers positions persisted before trail_distance existed.
            trail_dist = position.trail_distance or price * 0.02
            if position.trailing_stop is None:
                position.trailing_stop = position.stop_loss
            if price > position.entry_price and price - trail_dist > position.trailing_stop:
                position.trailing_stop = price - trail_dist

            effective_stop = max(position.stop_loss, position.trailing_stop or 0)

            if price <= effective_stop:
                # Label correctly: a trailed stop above the original stop is a
                # trailing_stop exit — ERL treats these very differently.
                reason = (
                    "trailing_stop"
                    if (position.trailing_stop or 0) > position.stop_loss
                    else "stop_loss"
                )
                closed = self.close_trade(position.trade_id, price, reason)
                if closed:
                    closed_this_update.append(closed)
            elif price >= position.target:
                closed = self.close_trade(position.trade_id, price, "take_profit")
                if closed:
                    closed_this_update.append(closed)

        # Ratchet peak equity on mark-to-market too — not only on closes — so
        # drawdown mode reflects peaks reached while positions were open.
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

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

    def export_state(self) -> dict:
        """Full live state for persistence."""
        return {
            "cash": self.cash,
            "starting_equity": self.starting_equity,
            "peak_equity": self.peak_equity,
            "total_commission": self.total_commission,
            "next_trade_id": self._next_trade_id,
            "open_positions": [p.to_state() for p in self.open_positions],
            "closed_trades": [c.to_state() for c in self.closed_trades],
        }

    def import_state(self, state: dict) -> None:
        """Rehydrate live state from a persisted export_state() dict."""
        self.cash = state.get("cash", self.cash)
        self.starting_equity = state.get("starting_equity", self.starting_equity)
        self.peak_equity = state.get("peak_equity", self.peak_equity)
        self.total_commission = state.get("total_commission", 0.0)
        self._next_trade_id = state.get("next_trade_id", 1)
        self.open_positions = [OpenPosition.from_state(d) for d in state.get("open_positions", [])]
        self.closed_trades = [ClosedTrade.from_state(d) for d in state.get("closed_trades", [])]


_portfolios: dict[str, Portfolio] = {}


def get_portfolio(track: TrackType) -> Portfolio:
    if track not in _portfolios:
        _portfolios[track] = Portfolio(track)
    return _portfolios[track]


def reset_portfolios(tracks: Optional[list[str]] = None) -> None:
    """Drop in-memory portfolio state. Clears the given tracks, or all if None.
    Next get_portfolio() rebuilds a cleared track fresh at starting capital."""
    if tracks is None:
        _portfolios.clear()
    else:
        for track in tracks:
            _portfolios.pop(track, None)
