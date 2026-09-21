from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from config.settings import settings
from src.agent.plan_replay import (
    evaluate_plan, fixed_prediction, freeze_path, plan_metric, score_plans, validate_path,
)
from src.agent.replay import DECISION_INPUTS, ReplayExample, load_corpus, save_corpus


def path(bars=None):
    bars = bars or [(100, 111, 99, 108), (108, 109, 104, 105), (105, 106, 102, 103)]
    frame = pd.DataFrame(bars, columns=['Open', 'High', 'Low', 'Close'],
                         index=pd.date_range('2025-01-02', periods=len(bars)))
    result = freeze_path(frame, 100.0, 2.0, 'us', datetime(2025, 1, 1))
    result['policy'].update(commission=0.0, slippage=0.0, trailing_atr=0.0, breakeven_atr=0.0,
                            min_stop_cost=0.0, min_stop_atr=0.5, min_rrr=2.5, atr_stop=1.5)
    return result


def prediction(stop=97, target=110, action='BUY'):
    return SimpleNamespace(action=action, confidence=0.8, stop_loss=stop, target=target)


def example(frozen=None):
    return ReplayExample('gpt', 'TEST', 'us', '2025-01-01',
                         dict.fromkeys(DECISION_INPUTS, 'test'), 'BUY', 2.5,
                         plan_path=frozen or path(), source_action='BUY')


def test_identical_buy_decisions_with_different_targets_have_different_payoffs():
    frozen = path()
    assert evaluate_plan(frozen, prediction(target=108))['net_r'] == pytest.approx(8 / 3)
    assert evaluate_plan(frozen, prediction(target=120))['net_r'] == pytest.approx(1)


def test_tighter_stop_does_not_inflate_r_for_same_proceeds():
    frozen = path([(100, 111, 99.5, 108)])
    wide = evaluate_plan(frozen, prediction(stop=97))
    tight = evaluate_plan(frozen, prediction(stop=98))
    assert wide['net_r'] == tight['net_r'] == pytest.approx(10 / 3)


def test_stops_change_actual_outcome():
    frozen = path([(100, 111, 97.5, 108)])
    assert evaluate_plan(frozen, prediction(stop=97))['net_r'] > 0
    assert evaluate_plan(frozen, prediction(stop=98))['net_r'] == pytest.approx(-2 / 3)


@pytest.mark.parametrize('stop,target,reason', [(101, 110, 'invalid_levels'), (97, 99, 'invalid_levels'),
    (99.9, 110, 'stop_inside_floor'), (90, 130, 'stop_beyond_ceiling'), (97, 103, 'rrr_below_minimum')])
def test_live_static_risk_constraints_block_plans(stop, target, reason):
    outcome = evaluate_plan(path(), prediction(stop, target))
    assert not outcome['executed'] and outcome['reason'] == reason
    assert outcome['net_r'] == 0 and outcome['score'] == 0.5


def test_pass_is_zero_profit_not_hindsight_credit():
    outcome = evaluate_plan(path(), prediction(action='PASS'))
    assert outcome['net_r'] == 0 and outcome['score'] == 0.5


def test_gaps_and_ambiguous_bars_are_conservative():
    assert evaluate_plan(path([(90, 95, 89, 94)]), prediction())['net_r'] == pytest.approx(-10 / 3)
    assert evaluate_plan(path([(100, 112, 95, 101)]), prediction())['net_r'] == pytest.approx(-1)
    assert evaluate_plan(path([(112, 115, 90, 100)]), prediction())['net_r'] == pytest.approx(4)


def test_costs_and_policy_remain_frozen_when_settings_change(monkeypatch):
    frozen = path()
    frozen['policy'].update(commission=0.002, slippage=0.001)
    before = evaluate_plan(frozen, prediction())
    assert before['net_r'] < 10 / 3
    monkeypatch.setattr(settings, 'commission_pct', 0.1)
    monkeypatch.setattr(settings, 'simulated_slippage', 0.1)
    monkeypatch.setattr(settings, 'breakeven_arm_atr_multiplier', 99)
    monkeypatch.setattr(settings, 'trailing_stop_atr_multiplier', 99)
    assert evaluate_plan(frozen, prediction()) == before


@pytest.mark.parametrize('change', ['nan', 'geometry', 'same_day', 'duplicate', 'empty'])
def test_invalid_paths_abort_even_for_pass(change):
    frozen = path()
    if change == 'nan': frozen['bars'][-1]['close'] = float('nan')
    elif change == 'geometry': frozen['bars'][0]['high'] = 50
    elif change == 'same_day': frozen['bars'][0]['date'] = '2025-01-01'
    elif change == 'duplicate': frozen['bars'][1]['date'] = frozen['bars'][0]['date']
    else: frozen['bars'] = []
    with pytest.raises(ValueError):
        evaluate_plan(frozen, prediction(action='PASS'))


def test_all_paths_validated_before_any_inference():
    rows = [example(), example()]
    rows[-1].plan_path = None
    model = Mock()
    with pytest.raises(ValueError, match='rebuild'):
        score_plans(rows, model)
    model.assert_not_called()


def test_predictions_receive_only_decision_time_inputs_and_report_plans():
    model = Mock(return_value=prediction())
    report = score_plans([example()], model)
    assert set(model.call_args.kwargs) == set(DECISION_INPUTS)
    assert report['buys'] == 1 and report['outcomes'][0]['target'] == 110


def test_provider_failure_is_not_scored_as_pass():
    with pytest.raises(RuntimeError, match='provider'):
        score_plans([example()], Mock(side_effect=RuntimeError('provider')))


def test_invalid_prediction_aborts_test_but_gets_zero_during_search():
    e = example()
    dated = {**e.entry_inputs, 'plan_path': e.plan_path}
    metric = plan_metric([dated])
    bad = prediction(target=float('nan'))
    assert metric(dated, bad) == 0
    with pytest.raises(ValueError):
        score_plans([e], Mock(return_value=bad))


def test_metric_scores_predicted_plan_not_stored_label():
    e = example()
    dated = {**e.entry_inputs, 'plan_path': e.plan_path, 'r_multiple': -999}
    metric = plan_metric([dated])
    assert metric(dated, prediction(target=108)) > metric(dated, prediction(target=120))


def test_corpus_round_trip_preserves_path_and_plan_score(tmp_path):
    corpus = [example()]
    dest = tmp_path / 'corpus.json'
    save_corpus(corpus, dest)
    loaded = load_corpus(dest)
    assert loaded == corpus
    assert score_plans(loaded, reference='always_buy') == score_plans(corpus, reference='always_buy')


def test_no_false_oracle_claim_for_alternative_plans():
    with pytest.raises(ValueError, match='No plan oracle'):
        score_plans([example()], reference='oracle')


def test_plan_cli_writes_a_reproducible_report(tmp_path):
    from scripts.replay_decisions import _score
    import json
    corpus = tmp_path / 'corpus.json'
    out = tmp_path / 'report.json'
    save_corpus([example()], corpus)
    args = SimpleNamespace(program=[], track='gpt', corpus=str(corpus), plans=True, reference=True, out=str(out))
    assert _score(args) == 0
    report = json.loads(out.read_text())
    assert report['corpus_hash'] and report['results']['always_pass']['total_r'] == 0


@pytest.mark.parametrize('stop,target', [(97, 110), (98, 110), (99.9, 110), (90, 130), (97, 103), (101, 110)])
def test_static_risk_decisions_match_live_validator(stop, target):
    from dataclasses import asdict
    from src.agent.plan_replay import PlanPolicy
    from src.agent.risk import validate_trade
    frozen = path()
    frozen['policy'] = asdict(PlanPolicy.current('us'))
    live = validate_trade('BUY', 100, stop, target, 100000, [],
                          SimpleNamespace(ticker='TEST', current_price=100, atr_14=2), market='us')
    assert evaluate_plan(frozen, prediction(stop, target))['executed'] == live.approved


def test_changed_executor_cannot_silently_rescore_an_old_corpus():
    frozen = path()
    frozen['execution_code_hash'] = 'old-executor'
    with pytest.raises(ValueError, match='execution code changed'):
        evaluate_plan(frozen, prediction())
