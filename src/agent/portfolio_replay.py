"""Chronological, constant-FX portfolio research over frozen decision paths."""
from __future__ import annotations

import math
import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from config.settings import settings
from src.agent.plan_replay import PlanPolicy, check_plan, execution_code_hash, fixed_prediction, validate_path
from src.agent.replay import DECISION_INPUTS
from src.portfolio.daily import DailyPortfolio


@dataclass(frozen=True)
class PortfolioPolicy:
    initial_equity: float
    risk_fraction: float
    hard_risk_fraction: float
    position_cap: float
    market_caps: dict[str, float]
    min_cash_fraction: float
    drawdown_threshold: float
    max_correlation: float
    max_per_sector: int
    correlation_window: int = 60
    min_correlation_returns: int = 20

    @classmethod
    def current(cls, initial_equity=100000.0):
        return cls(initial_equity, settings.max_risk_per_trade, settings.hard_cap_risk_per_trade,
                   settings.max_position_pct, dict(settings.market_allocation),
                   settings.min_cash_for_new_position_pct, settings.drawdown_pause_threshold,
                   settings.max_sector_correlation, settings.max_positions_per_sector)

    def validate(self):
        if not math.isfinite(self.initial_equity) or self.initial_equity <= 0:
            raise ValueError('Initial equity must be finite and positive')
        for value in (self.risk_fraction, self.hard_risk_fraction, self.position_cap,
                      self.min_cash_fraction, self.drawdown_threshold, *self.market_caps.values()):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Portfolio fractions must be between zero and one')
        if not -1 <= self.max_correlation <= 1 or self.max_per_sector < 1:
            raise ValueError('Invalid portfolio concentration limits')
        if self.min_correlation_returns < 2 or self.correlation_window < self.min_correlation_returns:
            raise ValueError('Invalid correlation history requirements')


def _history(examples):
    """Merge stored observations; conflicting adjusted histories must not silently win."""
    prices = {}
    for e in examples:
        key = (e.market, e.ticker)
        series = prices.setdefault(key, {})
        decision_day = datetime.fromisoformat(e.plan_path['decision_time']).date().isoformat()
        history = e.plan_path.get('history', [])
        for bar in history:
            if date.fromisoformat(bar['date']).isoformat() >= decision_day:
                raise ValueError('Correlation history must precede its decision day')
        for bar in history + e.plan_path['bars']:
            day, close = date.fromisoformat(bar['date']).isoformat(), bar['close']
            if not math.isfinite(close) or close <= 0:
                raise ValueError('Invalid correlation price history')
            if day in series and not math.isclose(series[day], close, rel_tol=1e-8):
                raise ValueError(f'Conflicting price histories for {key} on {day}')
            series[day] = close
    return prices


def _correlation(a, b, day, policy):
    # Compare returns over identical intervals, using completed prior days only.
    dates = sorted(d for d in a.keys() & b.keys() if d < day)[-(policy.correlation_window + 1):]
    if len(dates) <= policy.min_correlation_returns:
        return None
    x, y = np.array([a[d] for d in dates]), np.array([b[d] for d in dates])
    x, y = x[1:] / x[:-1] - 1, y[1:] / y[:-1] - 1
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def replay_portfolio(examples: list, predictions: list, policy: PortfolioPolicy) -> dict:
    """Replay precomputed predictions; no provider calls or hidden portfolio state."""
    policy.validate()
    if not examples or len(examples) != len(predictions):
        raise ValueError('Every portfolio opportunity needs exactly one prediction')
    if len({e.track for e in examples}) != 1:
        raise ValueError('Portfolio replay requires a single track')
    for e in examples:
        validate_path(e.plan_path)
        if datetime.fromisoformat(e.timestamp) != datetime.fromisoformat(e.plan_path['decision_time']):
            raise ValueError('Decision time does not match frozen path')
    histories = _history(examples)
    opportunities = {}
    dates = set()
    outcomes = []
    for i, (e, prediction) in enumerate(zip(examples, predictions)):
        # Entry eligibility consults static risk rules, never forward payoff.
        checked = check_plan(e.plan_path, prediction)
        day = datetime.fromisoformat(e.timestamp).date().isoformat()
        opportunities.setdefault(day, []).append((i, e, prediction, checked))
        dates.add(day)
        dates.update(b['date'] for b in e.plan_path['bars'])
        outcomes.append(dict(index=i, ticker=e.ticker, market=e.market, timestamp=e.timestamp,
                             action=checked['action'], confidence=checked['confidence'],
                             stop_loss=checked['stop_loss'], target=checked['target'],
                             executed=False, reason=checked['reason']))
    cash = policy.initial_equity
    positions = {}
    closed = []
    peak = cash
    nav = [dict(date=min(dates), phase='initial', equity=cash, cash=cash, exposure=0.0,
                open_positions=0, stale_positions=0)]
    correlation_checks = 0
    unknown_sectors = 0

    def equity():
        return cash + sum(p['engine'].open_equity for p in positions.values())

    for day in sorted(dates):
        # All entries use only prior-session marks. Same-day full bars have no
        # trustworthy intraday timestamp, so their exits cannot finance entries.
        ordered = sorted(opportunities.get(day, []), key=lambda row: (row[1].timestamp, row[1].market, row[1].ticker, row[0]))
        for i, e, prediction, checked in ordered:
            event = outcomes[i]
            if checked['reason'] != 'eligible':
                continue
            key = (e.market, e.ticker)
            sector = e.entry_inputs.get('replay_sector', '')
            if not sector or sector == 'Unknown':
                unknown_sectors += 1
            reason = None
            if key in positions:
                reason = 'already_open'
            elif sector and sector != 'Unknown' and sum(p['sector'] == sector for p in positions.values()) >= policy.max_per_sector:
                reason = 'sector_cap'
            if reason is None:
                for other, position in positions.items():
                    if other[0] != e.market:
                        continue  # matches live same-market correlation scope
                    correlation_checks += 1
                    corr = _correlation(histories[key], histories[other], day, policy)
                    if corr is None:
                        reason = 'correlation_history_unavailable'
                        break
                    if corr > policy.max_correlation:
                        reason = 'correlation_cap'
                        break
            current_equity = equity()
            exposure = sum(p['engine'].open_equity for p in positions.values() if p['market'] == e.market)
            budget = max(0.0, min(cash, policy.market_caps.get(e.market, 1.0) * current_equity - exposure))
            if reason is None and (budget <= 0 or budget < policy.min_cash_fraction * current_equity):
                reason = 'cash_or_market_budget'
            if reason:
                event['reason'] = reason
                continue
            execution = PlanPolicy(**e.plan_path['policy'])
            scale = 100.0 / e.plan_path['price']
            stop, target = checked['stop_loss'] * scale, checked['target'] * scale
            quantity = min(current_equity * min(policy.risk_fraction, policy.hard_risk_fraction) / (100 - stop),
                           current_equity * policy.position_cap / 100,
                           budget / (100 * (1 + execution.slippage) * (1 + execution.commission)))
            drawdown_mode = (peak - current_equity) / peak >= policy.drawdown_threshold
            if drawdown_mode:
                quantity *= 0.5
            # Fractional normalized units: no invented historical native FX rates.
            quantity = math.floor(quantity * 10000) / 10000
            fill_cost = 100 * (1 + execution.slippage) * quantity
            cost = fill_cost + fill_cost * execution.commission
            if quantity <= 0 or cost > cash:
                event['reason'] = 'insufficient_cash'
                continue
            engine = DailyPortfolio(cost, execution.commission, execution.slippage,
                                    trailing_atr_multiplier=execution.trailing_atr,
                                    breakeven_arm_atr_multiplier=execution.breakeven_atr)
            engine.open_position('position', 100, stop, target, quantity, date.fromisoformat(day),
                                 trail_distance=execution.trailing_atr * e.plan_path['atr'] * scale)
            if not engine.has_ticker('position'):
                raise ValueError('Portfolio ledger failed to fund an approved entry')
            # Entry-day NAV marks at the quote, exposing entry fees and slippage.
            engine._positions['position'].current_price = 100
            cash -= cost
            positions[key] = dict(engine=engine, market=e.market, sector=sector, scale=scale, index=i,
                                  bars={b['date']: b for b in e.plan_path['bars']},
                                  end=e.plan_path['bars'][-1]['date'], last_mark=day)
            event.update(executed=True, reason='opened', normalized_units=quantity,
                         entry_cost=cost, drawdown_mode=drawdown_mode)
        for key, position in list(positions.items()):
            bar = position['bars'].get(day)
            if bar is None:
                continue
            engine = position['engine']
            scaled = {k: bar[k] * position['scale'] for k in ('open', 'high', 'low', 'close')}
            engine.update({'position': scaled}, date.fromisoformat(day))
            position['last_mark'] = day
            if day == position['end'] and engine.has_ticker('position'):
                engine.close_position('position', scaled['close'], date.fromisoformat(day), 'horizon')
            if engine.closed_trades:
                trade = engine.closed_trades[0]
                cash += engine.cash
                closed.append(dict(index=position['index'], ticker=key[1], market=key[0],
                                   entry_date=str(trade.entry_date), exit_date=str(trade.exit_date),
                                   net_pnl=trade.net_pnl, commission=trade.commission, reason=trade.exit_reason))
                del positions[key]
        value = equity()
        peak = max(peak, value)
        nav.append(dict(date=day, phase='close', equity=value, cash=cash, exposure=value-cash,
                        open_positions=len(positions),
                        stale_positions=sum(p['last_mark'] < day for p in positions.values())))
    if positions or not math.isclose(cash, policy.initial_equity + sum(t['net_pnl'] for t in closed), abs_tol=1e-6):
        raise ValueError('Portfolio cash and trade ledger do not reconcile')
    peak, max_drawdown = policy.initial_equity, 0.0
    for point in nav:
        peak = max(peak, point['equity'])
        max_drawdown = max(max_drawdown, (peak-point['equity'])/peak)
    return dict(scope='sampled opportunities; constant FX; daily entry-before-exit ordering; no discretionary exits',
                execution_code_hash=hashlib.sha256(Path(__file__).read_bytes() + execution_code_hash().encode()).hexdigest(),
                policy=asdict(policy), initial_equity=policy.initial_equity, final_equity=cash,
                total_return_pct=(cash/policy.initial_equity-1)*100, max_drawdown_pct=max_drawdown*100,
                trades=len(closed), total_commission=sum(t['commission'] for t in closed),
                blocked=dict(Counter(o['reason'] for o in outcomes if o['action']=='BUY' and not o['executed'])),
                correlation_checks=correlation_checks, proposed_buys_without_sector=unknown_sectors,
                nav=nav, decisions=outcomes, closed_trades=closed)


def score_portfolio(examples: list, program=None, *, reference=None, policy=None):
    if reference not in (None, 'always_buy', 'always_pass'):
        raise ValueError('Invalid portfolio reference')
    # Validate the entire corpus before spending on model inference.
    if policy is None:
        policies = [e.plan_path.get('portfolio_policy') if e.plan_path else None for e in examples]
        if not policies or not all(policies) or any(p != policies[0] for p in policies):
            raise ValueError('Portfolio replay requires one frozen portfolio policy; rebuild with --plans')
        policy = PortfolioPolicy(**policies[0])
    replay_portfolio(examples, [SimpleNamespace(action='PASS', confidence=1, stop_loss=0, target=0)
                                for _ in examples], policy)
    predictions = [(fixed_prediction(e.plan_path, 'BUY' if reference=='always_buy' else 'PASS') if reference
                    else program(**{k: e.entry_inputs[k] for k in DECISION_INPUTS})) for e in examples]
    return replay_portfolio(examples, predictions, policy)
