from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal, Optional

from config.settings import settings
from src.data.options_chain import CONTRACT_MULTIPLIER
from src.portfolio.simulator import _iso, _parse_dt, persist_portfolio

logger = logging.getLogger(__name__)

OptionsTrackType = Literal["claude-opt", "gpt-opt"]


def _parse_date(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


@dataclass
class OptionPosition:
    trade_id: int
    contract_symbol: str
    underlying: str
    right: str
    strike: float              # USD — contract identity
    expiry: date
    contracts: int
    entry_premium: float       # SEK per underlying share
    profit_target_pct: float   # close at entry * (1 + this)
    max_loss_pct: float        # close at entry * (1 - this)
    time_stop_dte: int
    entry_time: datetime
    market: str = "us"
    current_premium: float = 0.0
    entry_underlying_price: float = 0.0  # USD, for ERL attribution
    iv_at_entry: float = 0.0
    delta_at_entry: float = 0.0
    regime: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    technical_snapshot: str = ""
    sector: str = ""
    entry_inputs: dict = field(default_factory=dict)

    @property
    def dte(self) -> int:
        return (self.expiry - date.today()).days

    @property
    def market_value(self) -> float:
        return self.current_premium * CONTRACT_MULTIPLIER * self.contracts

    @property
    def unrealised_pnl_pct(self) -> float:
        if self.entry_premium == 0:
            return 0.0
        return (self.current_premium - self.entry_premium) / self.entry_premium

    @property
    def premium_stop_level(self) -> float:
        return self.entry_premium * (1 - self.max_loss_pct)

    @property
    def profit_target_level(self) -> float:
        return self.entry_premium * (1 + self.profit_target_pct)

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "ticker": f"{self.underlying} {self.right[0].upper()}${self.strike:g} {self.expiry.isoformat()}",
            "market": self.market,
            "quantity": self.contracts,
            "entry_price": self.entry_premium,
            "current_price": self.current_premium,
            "stop_loss": self.premium_stop_level,
            "target": self.profit_target_level,
            "trailing_stop": None,
            "entry_time": self.entry_time.isoformat(),
            "unrealised_pnl": round((self.current_premium - self.entry_premium) * CONTRACT_MULTIPLIER * self.contracts, 2),
            "unrealised_pnl_pct": round(self.unrealised_pnl_pct * 100, 2),
            "market_value": round(self.market_value, 2),
            "dte": self.dte,
        }

    def to_state(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "contract_symbol": self.contract_symbol,
            "underlying": self.underlying,
            "right": self.right,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "contracts": self.contracts,
            "entry_premium": self.entry_premium,
            "profit_target_pct": self.profit_target_pct,
            "max_loss_pct": self.max_loss_pct,
            "time_stop_dte": self.time_stop_dte,
            "entry_time": _iso(self.entry_time),
            "market": self.market,
            "current_premium": self.current_premium,
            "entry_underlying_price": self.entry_underlying_price,
            "iv_at_entry": self.iv_at_entry,
            "delta_at_entry": self.delta_at_entry,
            "regime": self.regime,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "technical_snapshot": self.technical_snapshot,
            "sector": self.sector,
            "entry_inputs": self.entry_inputs,
        }

    @classmethod
    def from_state(cls, d: dict) -> "OptionPosition":
        return cls(
            trade_id=d["trade_id"],
            contract_symbol=d["contract_symbol"],
            underlying=d["underlying"],
            right=d["right"],
            strike=d["strike"],
            expiry=_parse_date(d["expiry"]),
            contracts=d["contracts"],
            entry_premium=d["entry_premium"],
            profit_target_pct=d["profit_target_pct"],
            max_loss_pct=d["max_loss_pct"],
            time_stop_dte=d["time_stop_dte"],
            entry_time=_parse_dt(d.get("entry_time")) or datetime.utcnow(),
            market=d.get("market", "us"),
            current_premium=d.get("current_premium", 0.0),
            entry_underlying_price=d.get("entry_underlying_price", 0.0),
            iv_at_entry=d.get("iv_at_entry", 0.0),
            delta_at_entry=d.get("delta_at_entry", 0.0),
            regime=d.get("regime", ""),
            reasoning=d.get("reasoning", ""),
            confidence=d.get("confidence", 0.0),
            technical_snapshot=d.get("technical_snapshot", ""),
            sector=d.get("sector", ""),
            entry_inputs=d.get("entry_inputs") or {},
        )


@dataclass
class ClosedOptionTrade:
    trade_id: int
    contract_symbol: str
    underlying: str
    right: str
    strike: float
    expiry: date
    contracts: int
    entry_premium: float
    exit_premium: float
    profit_target_pct: float
    max_loss_pct: float
    entry_time: datetime
    exit_time: datetime
    regime: str
    reasoning: str
    confidence: float
    exit_reason: str  # "profit_target" | "premium_stop" | "time_stop" | "expired_itm" | "expired_worthless"
    market: str = "us"
    technical_snapshot: str = ""
    entry_inputs: dict = field(default_factory=dict)
    iv_at_entry: float = 0.0
    iv_at_exit: float = 0.0
    entry_underlying_price: float = 0.0
    exit_underlying_price: float = 0.0

    @property
    def ticker(self) -> str:
        return f"{self.underlying} {self.right[0].upper()}${self.strike:g} {self.expiry.isoformat()}"

    @property
    def pnl(self) -> float:
        return (self.exit_premium - self.entry_premium) * CONTRACT_MULTIPLIER * self.contracts

    @property
    def pnl_pct(self) -> float:
        if self.entry_premium == 0:
            return 0.0
        return (self.exit_premium - self.entry_premium) / self.entry_premium

    @property
    def rrr_achieved(self) -> float:
        # Reward realized relative to the planned max loss (the premium stop)
        if self.max_loss_pct <= 0:
            return 0.0
        return self.pnl_pct / self.max_loss_pct

    @property
    def duration_days(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 86400

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "market": self.market,
            "quantity": self.contracts,
            "entry_price": self.entry_premium,
            "exit_price": self.exit_premium,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct * 100, 2),
            "rrr_achieved": round(self.rrr_achieved, 2),
            "exit_reason": self.exit_reason,
            "duration_days": round(self.duration_days, 2),
            "regime": self.regime,
            "confidence": self.confidence,
        }

    def to_state(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "contract_symbol": self.contract_symbol,
            "underlying": self.underlying,
            "right": self.right,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "contracts": self.contracts,
            "entry_premium": self.entry_premium,
            "exit_premium": self.exit_premium,
            "profit_target_pct": self.profit_target_pct,
            "max_loss_pct": self.max_loss_pct,
            "entry_time": _iso(self.entry_time),
            "exit_time": _iso(self.exit_time),
            "regime": self.regime,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "exit_reason": self.exit_reason,
            "market": self.market,
            "technical_snapshot": self.technical_snapshot,
            "entry_inputs": self.entry_inputs,
            "iv_at_entry": self.iv_at_entry,
            "iv_at_exit": self.iv_at_exit,
            "entry_underlying_price": self.entry_underlying_price,
            "exit_underlying_price": self.exit_underlying_price,
        }

    @classmethod
    def from_state(cls, d: dict) -> "ClosedOptionTrade":
        return cls(
            trade_id=d["trade_id"],
            contract_symbol=d["contract_symbol"],
            underlying=d["underlying"],
            right=d["right"],
            strike=d["strike"],
            expiry=_parse_date(d["expiry"]),
            contracts=d["contracts"],
            entry_premium=d["entry_premium"],
            exit_premium=d["exit_premium"],
            profit_target_pct=d["profit_target_pct"],
            max_loss_pct=d["max_loss_pct"],
            entry_time=_parse_dt(d.get("entry_time")) or datetime.utcnow(),
            exit_time=_parse_dt(d.get("exit_time")) or datetime.utcnow(),
            regime=d.get("regime", ""),
            reasoning=d.get("reasoning", ""),
            confidence=d.get("confidence", 0.0),
            exit_reason=d.get("exit_reason", ""),
            market=d.get("market", "us"),
            technical_snapshot=d.get("technical_snapshot", ""),
            entry_inputs=d.get("entry_inputs") or {},
            iv_at_entry=d.get("iv_at_entry", 0.0),
            iv_at_exit=d.get("iv_at_exit", 0.0),
            entry_underlying_price=d.get("entry_underlying_price", 0.0),
            exit_underlying_price=d.get("exit_underlying_price", 0.0),
        )


class OptionsPortfolio:
    """Paper portfolio for one options track. Long single-leg contracts only, so
    max loss per position is the premium paid — no margin, no assignment."""

    def __init__(self, track: OptionsTrackType):
        self.track = track
        self.cash = settings.options_starting_capital_sek
        self.starting_equity = settings.options_starting_capital_sek
        self.peak_equity = settings.options_starting_capital_sek
        self.open_positions: list[OptionPosition] = []
        self.closed_trades: list[ClosedOptionTrade] = []
        self.total_commission: float = 0.0
        self._next_trade_id = 1

    @property
    def equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.open_positions)

    @property
    def drawdown(self) -> float:
        if self.peak_equity == 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    @property
    def is_drawdown_mode(self) -> bool:
        return self.drawdown >= settings.drawdown_pause_threshold

    @property
    def can_open_new_position(self) -> bool:
        return self.cash >= settings.min_cash_for_new_position_pct * self.equity

    def open_option(
        self,
        contract_symbol: str,
        underlying: str,
        right: str,
        strike: float,
        expiry: date,
        contracts: int,
        entry_premium: float,
        profit_target_pct: float,
        max_loss_pct: float,
        time_stop_dte: int,
        regime: str,
        reasoning: str,
        confidence: float,
        entry_underlying_price: float = 0.0,
        iv_at_entry: float = 0.0,
        delta_at_entry: float = 0.0,
        technical_snapshot: str = "",
        sector: str = "",
        entry_inputs: Optional[dict] = None,
    ) -> Optional[OptionPosition]:
        cost = entry_premium * CONTRACT_MULTIPLIER * contracts
        commission = settings.options_commission_per_contract_sek * contracts
        if cost + commission > self.cash:
            logger.warning(
                "[%s] Insufficient cash for %s: need %.2f, have %.2f",
                self.track, contract_symbol, cost + commission, self.cash,
            )
            return None

        self.cash -= cost + commission
        self.total_commission += commission
        position = OptionPosition(
            trade_id=self._next_trade_id,
            contract_symbol=contract_symbol,
            underlying=underlying,
            right=right,
            strike=strike,
            expiry=expiry,
            contracts=contracts,
            entry_premium=entry_premium,
            profit_target_pct=profit_target_pct,
            max_loss_pct=max_loss_pct,
            time_stop_dte=time_stop_dte,
            entry_time=datetime.utcnow(),
            current_premium=entry_premium,
            entry_underlying_price=entry_underlying_price,
            iv_at_entry=iv_at_entry,
            delta_at_entry=delta_at_entry,
            regime=regime,
            reasoning=reasoning,
            confidence=confidence,
            technical_snapshot=technical_snapshot,
            sector=sector,
            entry_inputs=entry_inputs or {},
        )
        self.open_positions.append(position)
        self._next_trade_id += 1

        logger.info(
            "[%s] OPENED %s x%d @ %.2f SEK/share (target +%.0f%%, stop -%.0f%%, time stop DTE %d)",
            self.track, position.to_dict()["ticker"], contracts, entry_premium,
            profit_target_pct * 100, max_loss_pct * 100, time_stop_dte,
        )
        persist_portfolio(self)
        return position

    def close_option(
        self,
        trade_id: int,
        exit_premium: float,
        exit_reason: str,
        iv_at_exit: float = 0.0,
        exit_underlying_price: float = 0.0,
    ) -> Optional[ClosedOptionTrade]:
        position = next((p for p in self.open_positions if p.trade_id == trade_id), None)
        if position is None:
            logger.warning("[%s] No open option position with trade_id=%s", self.track, trade_id)
            return None

        proceeds = exit_premium * CONTRACT_MULTIPLIER * position.contracts
        commission = settings.options_commission_per_contract_sek * position.contracts
        self.cash += proceeds - commission
        self.total_commission += commission
        self.open_positions.remove(position)

        closed = ClosedOptionTrade(
            trade_id=trade_id,
            contract_symbol=position.contract_symbol,
            underlying=position.underlying,
            right=position.right,
            strike=position.strike,
            expiry=position.expiry,
            contracts=position.contracts,
            entry_premium=position.entry_premium,
            exit_premium=exit_premium,
            profit_target_pct=position.profit_target_pct,
            max_loss_pct=position.max_loss_pct,
            entry_time=position.entry_time,
            exit_time=datetime.utcnow(),
            regime=position.regime,
            reasoning=position.reasoning,
            confidence=position.confidence,
            exit_reason=exit_reason,
            market=position.market,
            technical_snapshot=position.technical_snapshot,
            entry_inputs=position.entry_inputs,
            iv_at_entry=position.iv_at_entry,
            iv_at_exit=iv_at_exit,
            entry_underlying_price=position.entry_underlying_price,
            exit_underlying_price=exit_underlying_price,
        )
        self.closed_trades.append(closed)

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        logger.info(
            "[%s] CLOSED %s @ %.2f (P&L=%.1f%%, reason=%s)",
            self.track, closed.ticker, exit_premium, closed.pnl_pct * 100, exit_reason,
        )
        persist_portfolio(self)
        return closed

    def update_premiums(self, quotes: dict[str, float]) -> list[ClosedOptionTrade]:
        """Mark open positions to fresh SEK premiums; close any that hit their
        profit target, premium stop, or time stop. Expiry is handled separately."""
        closed_this_update: list[ClosedOptionTrade] = []

        for position in list(self.open_positions):
            premium = quotes.get(position.contract_symbol)
            if premium is not None:
                position.current_premium = premium

            mark = position.current_premium
            if mark <= 0:
                continue

            if mark <= position.premium_stop_level:
                closed = self.close_option(position.trade_id, mark, "premium_stop")
            elif mark >= position.profit_target_level:
                closed = self.close_option(position.trade_id, mark, "profit_target")
            elif position.dte <= position.time_stop_dte:
                closed = self.close_option(position.trade_id, mark, "time_stop")
            else:
                continue
            if closed:
                closed_this_update.append(closed)

        return closed_this_update

    def expired_positions(self, as_of: Optional[date] = None) -> list[OptionPosition]:
        today = as_of or date.today()
        return [p for p in self.open_positions if p.expiry <= today]

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

    def get_open_underlyings(self) -> list[str]:
        return [p.underlying for p in self.open_positions]

    def export_state(self) -> dict:
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
        self.cash = state.get("cash", self.cash)
        self.starting_equity = state.get("starting_equity", self.starting_equity)
        self.peak_equity = state.get("peak_equity", self.peak_equity)
        self.total_commission = state.get("total_commission", 0.0)
        self._next_trade_id = state.get("next_trade_id", 1)
        self.open_positions = [OptionPosition.from_state(d) for d in state.get("open_positions", [])]
        self.closed_trades = [ClosedOptionTrade.from_state(d) for d in state.get("closed_trades", [])]


_options_portfolios: dict[str, OptionsPortfolio] = {}


def get_options_portfolio(track: OptionsTrackType) -> OptionsPortfolio:
    if track not in _options_portfolios:
        _options_portfolios[track] = OptionsPortfolio(track)
    return _options_portfolios[track]


def reset_options_portfolios(tracks: Optional[list[str]] = None) -> None:
    if tracks is None:
        _options_portfolios.clear()
    else:
        for track in tracks:
            _options_portfolios.pop(track, None)
