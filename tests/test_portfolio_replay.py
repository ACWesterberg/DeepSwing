from dataclasses import asdict, replace
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.agent.portfolio_replay import PortfolioPolicy, replay_portfolio, score_portfolio
from tests.test_plan_replay import example, path, prediction


def policy(**kwargs):
    base = PortfolioPolicy(10000, .01, .02, .1, {'us': .4, 'eu': .2}, .05, .1, .7, 2)
    return replace(base, **kwargs)


def opportunity(ticker='A', day='2025-01-01', bars=None, market='us', sector='Tech'):
    e = example(path(bars or [(100, 101, 99, 100)] * 3))
    e.ticker, e.timestamp, e.market = ticker, day + 'T10:00:00', market
    e.entry_inputs['replay_sector'] = sector
    e.plan_path['decision_time'] = e.timestamp
    start = datetime.fromisoformat(day)
    for i, bar in enumerate(e.plan_path['bars']):
        bar['date'] = (start + timedelta(days=i+1)).date().isoformat()
    e.plan_path['portfolio_policy'] = asdict(policy())
    # Enough nonconstant prior observations for correlation checks.
    close = 80
    history = []
    for i in range(61):
        close *= 1 + (.003 if i % 2 else -.001)
        history.append(dict(date=(start - timedelta(days=61-i)).date().isoformat(), close=close))
    e.plan_path['history'] = history
    return e


def replay(rows, **kwargs):
    return replay_portfolio(rows, [prediction(target=120) for _ in rows], policy(**kwargs))


def test_cash_ledger_sizing_and_fee_reconciliation():
    e = opportunity(bars=[(100, 111, 99, 108)])
    e.plan_path['policy'].update(commission=.002, slippage=.001)
    report = replay_portfolio([e], [prediction()], policy())
    assert report['trades'] == 1
    assert report['decisions'][0]['normalized_units'] == 10
    assert report['final_equity'] == pytest.approx(10000 + report['closed_trades'][0]['net_pnl'])
    assert report['nav'][1]['equity'] < 10000  # entry-day fee and slippage
    assert report['total_commission'] > 0
    assert report['nav'][-1]['cash'] == report['final_equity']


def test_duplicate_open_ticker_is_blocked():
    a, b = opportunity(), opportunity()
    # Exact observations must agree when merging repeated opportunities.
    report = replay([a, b])
    assert report['trades'] == 1
    assert report['blocked'] == {'already_open': 1}


def test_sector_concentration_blocks_before_correlation():
    report = replay([opportunity('A'), opportunity('B')], max_per_sector=1)
    assert report['blocked'] == {'sector_cap': 1}


def test_correlated_positions_are_blocked_using_only_prior_observations():
    report = replay([opportunity('A'), opportunity('B', sector='Other')])
    assert report['blocked'] == {'correlation_cap': 1}
    assert report['correlation_checks'] == 1


def test_missing_correlation_evidence_is_visible_not_assumed_safe():
    a, b = opportunity('A'), opportunity('B', sector='Other')
    b.plan_path['history'] = []
    report = replay([a, b])
    assert report['blocked'] == {'correlation_history_unavailable': 1}


def test_future_bars_cannot_supply_correlation_history():
    a, b = opportunity('A'), opportunity('B', sector='Other')
    a.plan_path['history'] = b.plan_path['history'] = []
    report = replay([a, b])
    assert report['blocked'] == {'correlation_history_unavailable': 1}


def test_market_budget_includes_fees_and_slippage():
    e = opportunity()
    e.plan_path['policy'].update(commission=.002, slippage=.001)
    report = replay([e], position_cap=1, market_caps={'us': .01}, min_cash_fraction=0)
    assert report['decisions'][0]['entry_cost'] <= 100
    assert min(p['cash'] for p in report['nav']) >= 0


def test_no_future_exit_proceeds_fund_earlier_same_day_entry():
    a = opportunity('A', bars=[(100, 130, 99, 125)])
    b = opportunity('B', day='2025-01-02', market='eu', sector='Other')
    report = replay([a, b], risk_fraction=1, hard_risk_fraction=1, position_cap=1,
                    market_caps={'us': 1, 'eu': 1})
    assert report['decisions'][1]['reason'] == 'cash_or_market_budget'
    assert report['closed_trades'][0]['exit_date'] == '2025-01-02'


def test_proceeds_become_available_the_following_day():
    a = opportunity('A', bars=[(100, 130, 99, 125)])
    b = opportunity('B', day='2025-01-03', market='eu', sector='Other')
    report = replay([a, b], risk_fraction=1, hard_risk_fraction=1, position_cap=1,
                    market_caps={'us': 1, 'eu': 1})
    assert report['trades'] == 2


def test_nav_captures_unrealized_drawdown_even_when_trade_recovers():
    e = opportunity(bars=[(100, 101, 98, 98), (98, 101, 98, 100)])
    report = replay([e])
    assert report['final_equity'] == 10000
    assert report['max_drawdown_pct'] == pytest.approx(.2)


def test_drawdown_halves_size_after_caps():
    a = opportunity('A', bars=[(90, 95, 89, 94)])
    b = opportunity('B', day='2025-01-03', market='eu', sector='Other')
    report = replay([a, b], drawdown_threshold=.005)
    event = report['decisions'][1]
    assert event['drawdown_mode']
    assert event['normalized_units'] == pytest.approx(4.95)


def test_deterministic_tie_order_does_not_depend_on_input_list():
    a, b = opportunity('A'), opportunity('B')
    first, second = replay([a, b], max_per_sector=1), replay([b, a], max_per_sector=1)
    assert first['closed_trades'][0]['ticker'] == second['closed_trades'][0]['ticker'] == 'A'
    assert first['final_equity'] == second['final_equity']


def test_mixed_tracks_and_conflicting_prices_are_rejected_before_inference():
    a, b = opportunity('A'), opportunity('A')
    b.plan_path['bars'][0]['close'] = 100.5
    model = Mock()
    with pytest.raises(ValueError, match='Conflicting'):
        score_portfolio([a, b], model)
    model.assert_not_called()
    b.track = 'claude'
    with pytest.raises(ValueError, match='single track'):
        score_portfolio([a, b], model)


def test_policy_is_frozen_and_model_receives_no_history_or_sector_metadata():
    from src.agent.replay import DECISION_INPUTS
    model = Mock(return_value=prediction())
    result = score_portfolio([opportunity()], model)
    assert set(model.call_args.kwargs) == set(DECISION_INPUTS)
    assert result['policy'] == asdict(policy())


def test_pass_reference_retains_cash_and_zero_drawdown():
    result = score_portfolio([opportunity()], reference='always_pass')
    assert result['final_equity'] == 10000
    assert result['max_drawdown_pct'] == 0
    assert result['trades'] == 0


def test_portfolio_cli_requires_track_and_writes_nav(tmp_path):
    from src.agent.replay import save_corpus
    from scripts.replay_decisions import _score
    import json
    cache, out = tmp_path / 'corpus.json', tmp_path / 'report.json'
    save_corpus([opportunity()], cache)
    args = SimpleNamespace(portfolio=True, plans=False, track=None, program=[], reference=True,
                           corpus=str(cache), out=str(out))
    assert _score(args) == 2
    args.track = 'gpt'
    assert _score(args) == 0
    report = json.loads(out.read_text())
    assert report['results']['always_buy']['nav']


def test_native_price_scale_does_not_invent_currency_exposure():
    e = opportunity(bars=[(100, 111, 99, 108)])
    before = replay_portfolio([e], [prediction()], policy())
    e.plan_path['price'] *= 10
    e.plan_path['atr'] *= 10
    for bar in e.plan_path['bars']:
        for key in ('open', 'high', 'low', 'close'):
            bar[key] *= 10
    for bar in e.plan_path['history']:
        bar['close'] *= 10
    after = replay_portfolio([e], [prediction(stop=970, target=1100)], policy())
    assert before['final_equity'] == after['final_equity']
    assert before['nav'] == after['nav']


def test_missing_sector_coverage_is_reported():
    result = replay([opportunity(sector='')])
    assert result['proposed_buys_without_sector'] == 1


def test_missing_portfolio_policy_is_not_replaced_by_current_settings():
    e = opportunity()
    del e.plan_path['portfolio_policy']
    model = Mock()
    with pytest.raises(ValueError, match='frozen portfolio policy'):
        score_portfolio([e], model)
    model.assert_not_called()


def test_stale_marks_remain_visible_in_nav():
    a = opportunity('A', bars=[(100, 101, 99, 100), (100, 101, 99, 100)])
    a.plan_path['bars'][1]['date'] = '2025-01-05'
    b = opportunity('B', day='2025-01-03', market='eu', sector='Other')
    result = replay([a, b])
    assert any(point['stale_positions'] > 0 for point in result['nav'])
