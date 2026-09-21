from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from config.settings import settings
from src.agent.evaluation import evaluate_program, promotion_decision, summarize_actions, temporal_split
from src.agent.replay import DECISION_INPUTS


def _examples(n=100):
    start = datetime(2025, 1, 1)
    return [dict(
        example_id=f"sample:{i}", ticker=f"T{i % 10}", market="us", source="real" if i % 3 == 0 else "counterfactual",
        decision_time=(start + timedelta(days=i)).isoformat(),
        label_available_at=(start + timedelta(days=i + 2)).isoformat(),
        technicals="positive" if i % 2 == 0 else "negative", regime="trending",
        news_summary="n", macro_context="m", heuristics="h",
        action="BUY" if i % 2 == 0 else "PASS", r_multiple=2.0 if i % 2 == 0 else -1.0,
    ) for i in range(n)]


NOW = datetime(2026, 1, 1)


class TestGlobalTemporalSplit:
    def test_mixed_sources_and_ticker_groups_use_one_time_boundary(self):
        examples = _examples()
        shuffled = sorted(examples, key=lambda e: (e["ticker"], e["source"]))
        split = temporal_split(shuffled, now=NOW)
        assert max(e["decision_time"] for e in split.train) < min(e["decision_time"] for e in split.validation)
        assert max(e["label_available_at"] for e in split.train) < split.validation_start.isoformat()
        assert max(e["label_available_at"] for e in split.validation) < split.test_start.isoformat()
        assert (len(split.train), len(split.validation), len(split.test), split.purged) == (58, 18, 20, 4)

    def test_long_held_training_trade_is_purged(self):
        examples = _examples()
        examples[0]["label_available_at"] = examples[-1]["decision_time"]
        split = temporal_split(examples, now=NOW)
        assert examples[0] not in split.train

    def test_equal_timestamps_never_straddle_a_boundary(self):
        examples = _examples()
        for i in range(55, 65):
            examples[i]["decision_time"] = examples[60]["decision_time"]
            examples[i]["label_available_at"] = examples[70]["decision_time"]
        split = temporal_split(examples, now=NOW)
        train_times = {e["decision_time"] for e in split.train}
        assert not train_times & {e["decision_time"] for e in split.validation + split.test}

    def test_exposed_test_period_is_not_reused(self):
        examples = _examples()
        split = temporal_split(examples, now=NOW, last_test_time=examples[90]["decision_time"])
        assert [e["example_id"] for e in split.test] == [f"sample:{i}" for i in range(91, 100)]

    def test_immature_test_outcomes_are_not_used(self):
        examples = _examples()
        split = temporal_split(examples, now=datetime.fromisoformat(examples[90]["decision_time"]))
        assert len(split.test) == 9

    def test_missing_timing_does_not_fall_back_to_random_split(self):
        examples = _examples()
        del examples[0]["label_available_at"]
        with pytest.raises(KeyError):
            temporal_split(examples, now=NOW)

    def test_small_sets_are_never_duplicated_between_train_and_test(self):
        with pytest.raises(ValueError):
            temporal_split(_examples(1), now=NOW)


class TestPromotionComparison:
    def _results(self, rows, candidate_actions):
        return {
            "candidate": summarize_actions(rows, candidate_actions),
            "incumbent": summarize_actions(rows, ["BUY"] * len(rows)),
            "baseline": summarize_actions(rows, ["BUY"] * len(rows)),
            "always_buy": summarize_actions(rows, ["BUY"] * len(rows)),
            "always_pass": summarize_actions(rows, ["PASS"] * len(rows)),
        }

    def test_paired_improvement_over_all_references_can_pass(self):
        rows = _examples(40)
        result = promotion_decision(rows, self._results(rows, [e["action"] for e in rows]), min_gain=0.01, min_buys=5)
        assert result["promote"]
        assert all(c["lower_bound"] > 0 for c in result["comparisons"].values())

    @pytest.mark.parametrize("action", ["BUY", "PASS"])
    def test_trivial_strategy_never_promotes_itself(self, action):
        rows = _examples(40)
        result = promotion_decision(rows, self._results(rows, [action] * len(rows)), min_gain=0.01, min_buys=5)
        assert not result["promote"]

    def test_candidate_must_beat_an_already_good_incumbent(self):
        rows = _examples(40)
        results = self._results(rows, [e["action"] for e in rows])
        results["incumbent"] = results["candidate"]
        assert not promotion_decision(rows, results, min_gain=0.01, min_buys=5)["promote"]

    def test_too_few_buys_cannot_establish_improvement(self):
        rows = _examples(40)
        results = self._results(rows, [e["action"] for e in rows])
        assert not promotion_decision(rows, results, min_gain=0.01, min_buys=21)["promote"]

    def test_bad_numeric_predictions_abort(self):
        program = Mock(return_value=SimpleNamespace(action="BUY", confidence=0.8, stop_loss=97, target=float("nan")))
        with pytest.raises(ValueError, match="Invalid structured"):
            evaluate_program(_examples(1), program)

    def test_missing_reference_cannot_pass_vacuously(self):
        rows = _examples(40)
        results = self._results(rows, [e["action"] for e in rows])
        del results["incumbent"]
        with pytest.raises(ValueError, match="all reference"):
            promotion_decision(rows, results, min_gain=0.01, min_buys=5)

    def test_incomplete_scores_cannot_broadcast_into_a_comparison(self):
        rows = _examples(40)
        results = self._results(rows, [e["action"] for e in rows])
        results["baseline"]["scores"] = [0.1]
        with pytest.raises(ValueError, match="paired scores"):
            promotion_decision(rows, results, min_gain=0.01, min_buys=5)


class FakeProgram:
    def __init__(self, mode="baseline"):
        self.mode = mode

    def set_lm(self, lm):
        self.lm = lm

    def save(self, path):
        Path(path).write_text(json.dumps({"mode": self.mode}))

    def load(self, path):
        self.mode = json.loads(Path(path).read_text())["mode"]

    def __call__(self, **inputs):
        if self.mode == "error":
            raise RuntimeError("provider failure")
        assert set(inputs) == set(DECISION_INPUTS)
        action = "PASS" if self.mode == "candidate" and inputs["technicals"] == "negative" else "BUY"
        return SimpleNamespace(action=action, confidence=0.8, stop_loss=97, target=110)


@pytest.fixture
def optimizer_env(tmp_path, monkeypatch):
    import src.scheduler.optimizer as opt
    from src.agent.decision import DecisionEngine
    from src.scheduler import backup

    monkeypatch.setattr(type(settings), "compiled_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(settings, "openai_api_key", "test-only")
    monkeypatch.setattr(opt.dspy, "Predict", lambda signature: FakeProgram())
    monkeypatch.setattr(opt.dspy, "context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(opt, "build_lm", lambda *args, **kwargs: object())
    monkeypatch.setattr(opt, "_make_example", lambda inputs, action, r: {
        **{key: inputs[key] for key in DECISION_INPUTS}, "action": action, "r_multiple": r,
    })
    compiler = Mock(return_value=FakeProgram("candidate"))
    monkeypatch.setattr(opt, "MIPROv2", lambda **kwargs: SimpleNamespace(compile=compiler))
    engine = Mock()
    monkeypatch.setattr(DecisionEngine, "for_track", lambda track: engine)
    monkeypatch.setattr(backup, "backup_compiled_program", Mock())
    active = tmp_path / "gpt_trade_decision.json"
    FakeProgram().save(active)
    return SimpleNamespace(opt=opt, compiler=compiler, engine=engine, active=active, root=tmp_path)


def _reports(env):
    return [json.loads(p.read_text()) for p in env.root.glob("evaluations/gpt/*/report.json")]


class TestPromotionLifecycle:
    def test_only_independently_better_candidate_replaces_the_incumbent(self, optimizer_env):
        env = optimizer_env
        assert env.opt._compile_and_evaluate("gpt", _examples())
        assert json.loads(env.active.read_text())["mode"] == "candidate"
        env.engine.reload.assert_called_once()
        report, = _reports(env)
        assert report["status"] == "promoted"
        assert report["gate"]["promote"]
        kwargs = env.compiler.call_args.kwargs
        assert len(kwargs["trainset"]) == 58 and len(kwargs["valset"]) == 18
        assert all("decision_time" not in row for row in kwargs["trainset"])
        corpus_path, = env.root.glob("evaluations/gpt/*/corpus.json")
        corpus = json.loads(corpus_path.read_text())
        assert len(corpus["datasets"]["test"]) == 20
        assert report["corpus_hash"] == corpus["hash"]

    def test_failed_test_is_consumed_and_incumbent_is_preserved(self, optimizer_env):
        env = optimizer_env
        before = env.active.read_bytes()
        env.compiler.return_value = FakeProgram("error")
        assert not env.opt._compile_and_evaluate("gpt", _examples())
        assert env.active.read_bytes() == before
        env.engine.reload.assert_not_called()
        assert not env.opt._compile_and_evaluate("gpt", _examples())
        assert env.compiler.call_count == 1
        assert {r["status"] for r in _reports(env)} == {"error", "skipped"}

    def test_non_improvement_is_recorded_without_promotion(self, optimizer_env):
        env = optimizer_env
        before = env.active.read_bytes()
        env.compiler.return_value = FakeProgram("baseline")
        assert not env.opt._compile_and_evaluate("gpt", _examples())
        assert env.active.read_bytes() == before
        assert _reports(env)[0]["status"] == "rejected"

    def test_insufficient_diversity_skips_before_paid_work(self, optimizer_env):
        env = optimizer_env
        rows = _examples()
        for row in rows:
            row["ticker"] = "SAME"
        assert not env.opt._compile_and_evaluate("gpt", rows)
        env.compiler.assert_not_called()

    @pytest.mark.parametrize("contents", ["bad json", "{}", "[]"])
    def test_corrupt_watermark_does_not_reset_test_history(self, optimizer_env, contents):
        env = optimizer_env
        state = env.root / "evaluations/gpt/state.json"
        state.parent.mkdir(parents=True)
        state.write_text(contents)
        assert not env.opt._compile_and_evaluate("gpt", _examples())
        env.compiler.assert_not_called()

    def test_reload_failure_is_distinguished_from_failed_promotion(self, optimizer_env):
        env = optimizer_env
        env.engine.reload.side_effect = RuntimeError("reload failed")
        assert env.opt._compile_and_evaluate("gpt", _examples())
        assert json.loads(env.active.read_text())["mode"] == "candidate"
        report, = _reports(env)
        assert report["status"] == "promoted"
        assert report["reload_error"] == "reload failed"

    def test_compile_failure_does_not_consume_unseen_test(self, optimizer_env):
        env = optimizer_env
        env.compiler.side_effect = RuntimeError("compile failed")
        assert not env.opt._compile_and_evaluate("gpt", _examples())
        assert not (env.root / "evaluations/gpt/state.json").exists()

    def test_plan_mode_uses_paths_for_search_and_promotion(self, optimizer_env, monkeypatch):
        from tests.test_plan_replay import path, prediction
        env = optimizer_env
        rows = _examples()
        for i, row in enumerate(rows):
            row['plan_path'] = path([(100, 111, 99, 108)] if i % 2 == 0 else [(100, 101, 95, 96)])
            # Labels may be wrong: held-out scoring must use the path, not this value.
            row['r_multiple'] = -99
        captured = {}
        def optimizer(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(compile=env.compiler)
        monkeypatch.setattr(env.opt, 'MIPROv2', optimizer)
        assert env.opt._compile_and_evaluate('gpt', rows)
        report, = _reports(env)
        assert report['scope'].startswith('isolated daily trade plans')
        assert report['results']['candidate']['total_r'] > 0
        assert report['results']['candidate']['outcomes'][0]['target'] == 110
        assert all('plan_path' not in row for row in env.compiler.call_args.kwargs['trainset'])
        assert captured['metric'](rows[0], prediction(target=110)) > captured['metric'](rows[0], prediction(target=120))
