"""Reproduce review findings using selected source definitions and synthetic inputs.

Run: python3 reviews/reproduce_review_findings.py
Requires numpy. Does not import the application, access services, or edit state.
Assertions intentionally confirm the defects present at review time; once fixed,
these checks should fail and be replaced by regression tests of correct behavior.
"""
import ast
import dataclasses
import datetime
import json
import logging
import math
from pathlib import Path
import re
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ns = dict(__name__='__main__', dataclass=dataclasses.dataclass, field=dataclasses.field,
          date=datetime.date, datetime=datetime.datetime, math=math, np=np, re=re,
          logger=logging.getLogger('review'), R_METRIC_SCALE=0.35)
settings_values = {}
for node in ast.parse((ROOT/'config/settings.py').read_text()).body:
    if isinstance(node, ast.ClassDef) and node.name == 'Settings':
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                try:
                    settings_values[item.target.id] = ast.literal_eval(item.value)
                except (ValueError, TypeError):
                    pass
ns['settings'] = SimpleNamespace(**settings_values)

def load(path, names):
    tree = ast.parse((ROOT/path).read_text())
    body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    body += [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(ROOT/path), 'exec'), ns)

load('src/portfolio/metrics.py', {'decision_metric','compute_metrics','PerformanceMetrics','_build_equity_curve','_max_drawdown','_compute_sharpe'})
load('src/agent/replay.py', {'select_decision_rows'})
load('src/scheduler/optimizer.py', {'_forward_split','_label_forward_path'})
load('src/portfolio/simulator.py', {'breakeven_from_costs','round_trip_cost_frac','resolve_exit'})
load('src/backtesting/engine.py', {'BacktestTrade','_SimPortfolio','_compute_metrics','_empty_metrics'})
load('src/agent/risk.py', {'RiskValidation','validate_trade'})
load('src/analysis/regime.py', {'_hurst_rs_levels'})
load('src/agent/memory.py', {'canonical_tokens','similarity','heuristic_similarity'})
mem_tree = ast.parse((ROOT/'src/agent/memory.py').read_text())
for node in mem_tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('_STOPWORDS','_NUMBER_RE') for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<memory constants>', 'exec'), ns)

out = {}
metric = ns['decision_metric']
ex = SimpleNamespace(r_multiple=2.5)
out['metric_ignores_execution'] = [metric(ex, SimpleNamespace(action='BUY', stop_loss=s, target=t)) for s,t in [(97,107.5),(101,90),(float('nan'),float('inf'))]]
rows = [dict(ticker=t, timestamp=d) for t,d in [('A',10),('B',9),('A',8),('B',7),('A',6),('B',5),('A',4),('B',3),('A',2),('B',1)]]
selected = ns['select_decision_rows'](rows, 10, 5, 150)
train,val = ns['_forward_split'](list(reversed(selected)))
out['temporal_split'] = dict(train=[r['timestamp'] for r in train], validation=[r['timestamp'] for r in val])

class Window:
    columns = ['Open','High','Low','Close']
    def __init__(self, rows): self.rows=rows
    def iterrows(self): return enumerate(self.rows)
label = ns['_label_forward_path'](Window([dict(Open=100,High=103,Low=99,Close=102),dict(Open=102,High=102,Low=100,Close=101)]),100,2)
out['breakeven_label'] = dict(label=label, buy_metric=metric(SimpleNamespace(r_multiple=label[1]),SimpleNamespace(action='BUY')), oracle_pass_metric=metric(SimpleNamespace(r_multiple=label[1]),SimpleNamespace(action=label[0])))
out['gap_label'] = ns['_label_forward_path'](Window([dict(Open=90,High=92,Low=89,Close=91)]),100,2)

p=ns['_SimPortfolio'](100000)
day=datetime.date(2026,9,1)
p.open_position('TEST',100,97,107.5,10,day,trail_distance=4)
p.update({'TEST':dict(open=99,high=101,low=96,close=100)},day)
out['same_day_exit'] = [(str(t.entry_date),str(t.exit_date),t.exit_price,t.exit_reason) for t in p.closed_trades]

signals=SimpleNamespace(ticker='TEST',current_price=100,atr_14=2)
args=dict(action='BUY',entry_price=100,stop_loss=97,target=107.5,portfolio_equity=100000,open_positions=[],signals=signals,available_cash=100000,market='us')
normal=ns['validate_trade'](**args)
drawdown=ns['validate_trade'](**args,is_drawdown_mode=True)
out['drawdown_sizing'] = dict(normal=normal.quantity,drawdown=drawdown.quantity)
out['nonfinite_target_approved'] = {str(v):ns['validate_trade'](**{**args,'target':v}).approved for v in [float('nan'),float('inf')]}

first=100+np.random.default_rng(42).normal(size=100)
changed=first.copy(); changed[45:]=np.linspace(145,1,55)
out['hurst_ignores_recent_55_bars']=[ns['_hurst_rs_levels'](a) for a in [first,changed]]
out['contradictory_rule_similarity']=ns['heuristic_similarity'](dict(trigger='RSI above 70',action='Do buy the breakout'),dict(trigger='RSI below 70',action='Do not buy the breakout'))
out['heuristic_update_magnitude_one_rule']=dict(real_minus_3pct=max(-1,min(1,-0.03*10)),counterfactual_minus_1R=max(-1,min(1,-1*10)))

portfolio=SimpleNamespace(track='gpt',closed_trades=[],equity=90000,starting_equity=100000)
out['no_closed_trades_return_reported']=ns['compute_metrics'](portfolio).total_return_pct
p=ns['_SimPortfolio'](100000,commission_rate=0.002)
p.open_position('TEST',100,97,107.5,10,day)
p.close_position('TEST',100.1,day,'news_exit')
t=p.closed_trades[0]
out['backtest_cost_loser_as_winner']=dict(net_pnl=t.net_pnl,gross_return=t.pnl_pct,metrics=ns['_compute_metrics']([t],100000))
assert len(set(out['metric_ignores_execution'])) == 1
assert max(out['temporal_split']['train']) > min(out['temporal_split']['validation'])
assert label[0] == 'PASS' and label[1] > 0
assert out['gap_label'][1] == -1
assert out['same_day_exit'][0][0] == out['same_day_exit'][0][1]
assert normal.quantity == drawdown.quantity
assert all(out['nonfinite_target_approved'].values())
assert out['hurst_ignores_recent_55_bars'][0] == out['hurst_ignores_recent_55_bars'][1]
assert out['contradictory_rule_similarity'] == 1
assert out['no_closed_trades_return_reported'] == 0
assert t.net_pnl < 0 and out['backtest_cost_loser_as_winner']['metrics']['win_rate'] == 1
print(json.dumps(out,indent=2))
print('11 assertions confirmed the reproduced behaviors.')
