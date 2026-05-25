"""Tests for ragdx.optim.objectives.CompositeObjective."""

from __future__ import annotations

import pytest

from ragdx.optim.objectives import CompositeObjective, default_objective


# ---------------------------------------------------------------- score
def test_weighted_sum_basic():
    obj = CompositeObjective(metrics={"a": 1.0, "b": 0.5})
    assert obj.score({"a": 0.8, "b": 0.4}) == pytest.approx(0.8 + 0.5 * 0.4)


def test_weighted_mean_normalises_by_contributing_weights():
    obj = CompositeObjective(metrics={"a": 1.0, "b": 1.0}, mode="weighted_mean")
    assert obj.score({"a": 0.6, "b": 0.4}) == pytest.approx(0.5)
    # b missing: mean should still divide by just the contributing weight (a=1.0)
    assert obj.score({"a": 0.6}) == pytest.approx(0.6)


def test_nan_values_ignored():
    obj = CompositeObjective(metrics={"a": 1.0, "b": 1.0})
    assert obj.score({"a": 0.8, "b": float("nan")}) == pytest.approx(0.8)


def test_missing_metrics_contribute_zero_in_sum():
    obj = CompositeObjective(metrics={"a": 1.0, "b": 1.0, "missing": 2.0})
    # 'missing' is absent from values; only a + b contribute.
    assert obj.score({"a": 0.5, "b": 0.5}) == pytest.approx(1.0)


def test_empty_metric_dict_yields_zero():
    assert CompositeObjective(metrics={}).score({"a": 1.0}) == 0.0


# ------------------------------------------------------------ constraints
def test_min_constraint_violation():
    obj = CompositeObjective(metrics={"a": 1.0}, constraints={"a": ("min", 0.5)})
    ok, violations = obj.satisfies_constraints({"a": 0.4})
    assert not ok
    assert any("0.4" in v for v in violations)


def test_max_constraint_violation():
    obj = CompositeObjective(
        metrics={"a": 1.0},
        constraints={"hallucination": ("max", 0.2)},
    )
    ok, violations = obj.satisfies_constraints({"a": 0.9, "hallucination": 0.35})
    assert not ok
    assert any("0.35" in v for v in violations)


def test_missing_metric_in_constraint():
    obj = CompositeObjective(metrics={"a": 1.0}, constraints={"missing": ("min", 0.1)})
    ok, violations = obj.satisfies_constraints({"a": 0.9})
    assert not ok
    assert "missing" in violations[0]


def test_satisfies_constraints_returns_true_when_no_constraints():
    obj = CompositeObjective(metrics={"a": 1.0})
    assert obj.satisfies_constraints({"a": 0.5}) == (True, [])


# ----------------------------------------------------------- best_among
def test_best_among_prefers_feasible_even_with_lower_score():
    obj = CompositeObjective(
        metrics={"a": 1.0},
        constraints={"halluc": ("max", 0.2)},
    )
    candidates = [
        {"name": "infeasible_but_high", "scores": {"a": 0.95, "halluc": 0.5}},
        {"name": "feasible_but_lower", "scores": {"a": 0.70, "halluc": 0.1}},
    ]
    best = obj.best_among(candidates)
    assert best["name"] == "feasible_but_lower"
    assert best["_feasible"] is True


def test_best_among_falls_back_to_highest_when_all_infeasible():
    obj = CompositeObjective(metrics={"a": 1.0}, constraints={"a": ("min", 1.0)})
    candidates = [
        {"name": "low", "scores": {"a": 0.3}},
        {"name": "high", "scores": {"a": 0.7}},
    ]
    best = obj.best_among(candidates)
    assert best["name"] == "high"
    assert best["_feasible"] is False


def test_best_among_empty_returns_none():
    obj = CompositeObjective(metrics={"a": 1.0})
    assert obj.best_among([]) is None


# ----------------------------------------------------------- evaluate()
def test_evaluate_packages_score_and_feasibility():
    obj = CompositeObjective(
        metrics={"a": 1.0},
        constraints={"halluc": ("max", 0.2)},
    )
    out = obj.evaluate({"a": 0.8, "halluc": 0.1})
    assert out["score"] == pytest.approx(0.8)
    assert out["feasible"] is True
    assert out["violations"] == []


# --------------------------------------------------------- overrides + io
def test_with_overrides_layers_on_top_of_defaults():
    base = default_objective("with_gt")
    bumped = base.with_overrides(metrics={"faithfulness": 2.0})
    assert bumped.metrics["faithfulness"] == 2.0
    # other metrics preserved
    assert "context_recall" in bumped.metrics


def test_to_dict_and_from_dict_roundtrip():
    obj = CompositeObjective(
        metrics={"a": 1.0, "b": 0.5},
        constraints={"c": ("max", 0.3)},
        mode="weighted_mean",
    )
    d = obj.to_dict()
    obj2 = CompositeObjective.from_dict(d)
    assert obj2 == obj


# --------------------------------------------------------- default factory
def test_default_objective_with_gt_contains_recall():
    obj = default_objective("with_gt")
    assert "context_recall" in obj.metrics
    assert obj.metrics["context_recall"] == 1.0


def test_default_objective_no_gt_excludes_recall():
    obj = default_objective("no_gt")
    assert "context_recall" not in obj.metrics
    assert obj.metrics["faithfulness"] == 1.0


def test_default_objective_no_constraints_by_default():
    assert default_objective("with_gt").constraints == {}
    assert default_objective("no_gt").constraints == {}
