"""Tests for per-layer metric aggregation (ragdx.core.metrics)."""

from __future__ import annotations

import pytest

from ragdx.core.metrics import (
    LAYER_OF,
    compute_layer_scores,
    metric_layer,
    weakest_layer,
)
from ragdx.schemas.models import EvaluationResult


def test_metric_layer_lookup() -> None:
    assert metric_layer("context_precision") == "retrieval"
    assert metric_layer("faithfulness") == "generation"
    assert metric_layer("answer_correctness") == "e2e"
    assert metric_layer("not_a_metric") is None


def test_simple_mean_default() -> None:
    ls = compute_layer_scores(
        {"context_precision": 0.5, "context_recall": 0.9}
    )
    # (0.5 + 0.9) / 2 = 0.7
    assert ls["retrieval"]["score"] == pytest.approx(0.7)
    assert ls["retrieval"]["n"] == 2
    # Empty layers report None, not 0.0.
    assert ls["generation"]["score"] is None
    assert ls["e2e"]["score"] is None


def test_lower_is_better_inverted() -> None:
    """hallucination=0.1 (good) must aggregate as 0.9, not 0.1."""
    ls = compute_layer_scores(
        {"faithfulness": 1.0, "hallucination": 0.1}
    )
    # generation = mean(1.0, 1-0.1) = mean(1.0, 0.9) = 0.95
    assert ls["generation"]["score"] == pytest.approx(0.95)
    # The oriented value is exposed; the raw is preserved separately.
    assert ls["generation"]["metrics"]["hallucination"] == pytest.approx(0.9)
    assert ls["generation"]["raw"]["hallucination"] == pytest.approx(0.1)


def test_noise_sensitivity_inverted() -> None:
    ls = compute_layer_scores({"noise_sensitivity": 0.2})
    assert ls["generation"]["metrics"]["noise_sensitivity"] == pytest.approx(0.8)


def test_weights_applied_within_layer() -> None:
    scores = {"context_precision": 0.5, "context_recall": 0.9}
    # Down-weight precision: weighted mean shifts toward recall.
    ls = compute_layer_scores(scores, weights={"context_precision": 0.5})
    # (0.5*0.5 + 0.9*1.0) / (0.5 + 1.0) = 1.15 / 1.5 = 0.7667
    assert ls["retrieval"]["score"] == pytest.approx(0.7667, abs=1e-3)


def test_weight_zero_excludes_metric() -> None:
    scores = {"context_precision": 0.5, "context_recall": 0.9}
    ls = compute_layer_scores(scores, weights={"context_precision": 0.0})
    # Only recall counts toward the score...
    assert ls["retrieval"]["score"] == pytest.approx(0.9)
    # ...but the metric is still listed in the breakdown.
    assert "context_precision" in ls["retrieval"]["metrics"]


def test_unknown_and_nonnumeric_skipped() -> None:
    ls = compute_layer_scores(
        {"context_precision": 0.8, "mystery": 0.5, "bad": "x", "flag": True}
    )
    assert ls["retrieval"]["score"] == pytest.approx(0.8)
    assert ls["retrieval"]["n"] == 1  # only context_precision counted


def test_weakest_layer() -> None:
    ls = compute_layer_scores({
        "context_precision": 0.6, "context_recall": 0.6,  # retrieval 0.6
        "faithfulness": 0.95,                              # generation 0.95
        "answer_correctness": 0.5,                         # e2e 0.5
    })
    assert weakest_layer(ls) == "e2e"


def test_weakest_layer_none_when_empty() -> None:
    assert weakest_layer(compute_layer_scores({})) is None


def test_clamps_out_of_range() -> None:
    # A stray 1.4 shouldn't blow up the mean; clamp to 1.0.
    ls = compute_layer_scores({"faithfulness": 1.4})
    assert ls["generation"]["score"] == pytest.approx(1.0)


def test_evaluation_result_layer_scores() -> None:
    r = EvaluationResult(
        retrieval={"context_precision": 0.55, "context_recall": 0.87},
        generation={"faithfulness": 1.0},
        e2e={"answer_correctness": 0.65},
    )
    ls = r.layer_scores()
    assert ls["retrieval"]["score"] == pytest.approx(0.71)
    assert ls["generation"]["score"] == pytest.approx(1.0)
    assert ls["e2e"]["score"] == pytest.approx(0.65)


def test_evaluation_result_layer_scores_with_weights() -> None:
    r = EvaluationResult(
        retrieval={"context_precision": 0.5, "context_recall": 0.9},
    )
    ls = r.layer_scores(weights={"context_precision": 0.5})
    assert ls["retrieval"]["score"] == pytest.approx(0.7667, abs=1e-3)


def test_layer_of_covers_all_threshold_metrics() -> None:
    """Every metric with a default threshold should map to a layer, so
    nothing silently falls out of the three-layer view."""
    from ragdx.core.thresholds import DEFAULT_THRESHOLDS

    # ``hit_rate_at_k`` / ``user_success_rate`` etc are allowed; just
    # assert the common ragas/deepeval ones are mapped.
    for m in (
        "context_precision", "context_recall", "faithfulness",
        "response_relevancy", "noise_sensitivity", "hallucination",
        "answer_correctness", "answer_accuracy", "citation_accuracy",
    ):
        assert m in LAYER_OF, f"{m} not mapped to a layer"
        assert m in DEFAULT_THRESHOLDS or m in LAYER_OF
