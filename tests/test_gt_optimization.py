"""Tests for the GT-aware optimization flow (helpers + AutoRAG + DSPy adapter).

The DSPy in-process ``optimize()`` path itself is not exercised here because
it requires the optional ``dspy`` dependency and a live LM. We do verify the
metric builder for the with-GT path (pure Python, no dspy required) and that
the no-GT path raises the documented errors when dspy/judge are missing.
"""

from __future__ import annotations

import pytest

from ragdx.optim._gt_helpers import (
    REFERENCE_FREE_METRICS,
    REFERENCE_REQUIRED_METRICS,
    gt_mode,
    has_ground_truth,
    select_metrics,
    validate_metrics_for_mode,
)
from ragdx.optim.autorag_adapter import AutoRAGAdapter
from ragdx.optim.dspy_adapter import DSPyAdapter, _token_f1
from ragdx.schemas.models import DatasetRecord, OptimizationExperiment


def _record(question: str, *, gt: str | None = None, answer: str | None = None) -> DatasetRecord:
    return DatasetRecord(
        question=question,
        ground_truth=gt,
        answer=answer,
        contexts=["c1", "c2"],
    )


# --------------------------------------------------------------- gt helpers
def test_has_ground_truth_majority_threshold():
    rs = [_record("q1", gt="A"), _record("q2", gt="B"), _record("q3")]
    assert has_ground_truth(rs) is True  # 2/3 >= 0.5


def test_has_ground_truth_strict_threshold_requires_all():
    rs = [_record("q1", gt="A"), _record("q2", gt="B"), _record("q3")]
    assert has_ground_truth(rs, threshold=1.0) is False
    rs_all = [_record("q1", gt="A"), _record("q2", gt="B"), _record("q3", gt="C")]
    assert has_ground_truth(rs_all, threshold=1.0) is True


def test_has_ground_truth_empty_string_is_no_gt():
    rs = [_record("q1", gt=""), _record("q2", gt="   ")]
    assert has_ground_truth(rs) is False


def test_gt_mode_returns_with_gt_or_no_gt():
    assert gt_mode([_record("q1", gt="A")]) == "with_gt"
    assert gt_mode([_record("q1")]) == "no_gt"
    assert gt_mode([]) == "no_gt"


def test_select_metrics_with_gt_includes_both_families():
    metrics = select_metrics("with_gt")
    assert REFERENCE_REQUIRED_METRICS.issubset(set(metrics))
    assert REFERENCE_FREE_METRICS.issubset(set(metrics))


def test_select_metrics_no_gt_excludes_reference_required():
    metrics = select_metrics("no_gt")
    assert set(metrics).isdisjoint(REFERENCE_REQUIRED_METRICS)
    assert REFERENCE_FREE_METRICS.issubset(set(metrics))


def test_select_metrics_extra_and_drop():
    metrics = select_metrics("no_gt", extra=["custom_score"], drop=["hallucination"])
    assert "custom_score" in metrics
    assert "hallucination" not in metrics


# ----------------------------------------- declarative validate_metrics_for_mode
def test_validate_metrics_for_mode_accepts_matching_set():
    assert validate_metrics_for_mode(
        ["faithfulness", "context_precision", "hallucination"], "no_gt"
    ) == {}
    assert validate_metrics_for_mode(
        ["faithfulness", "context_recall", "answer_correctness"], "with_gt"
    ) == {}


def test_validate_metrics_for_mode_rejects_gt_required_in_no_gt():
    errors = validate_metrics_for_mode(
        ["faithfulness", "context_recall", "answer_correctness"], "no_gt"
    )
    assert set(errors) == {"context_recall", "answer_correctness"}
    for reason in errors.values():
        assert "ground_truth" in reason or "no_gt" in reason


def test_validate_metrics_for_mode_rejects_unknown_metric():
    errors = validate_metrics_for_mode(["faithfulness", "not_a_real_metric"], "with_gt")
    assert "not_a_real_metric" in errors


# ------------------------------------------------------------ autorag adapter
def _exp(name: str = "exp", **kw) -> OptimizationExperiment:
    base = dict(
        name=name,
        tool="autorag",
        target_component="e2e",
        description="test",
        parameters={},
        objectives={},
        search_space={},
        max_trials=3,
    )
    base.update(kw)
    return OptimizationExperiment(**base)


def test_autorag_spec_with_gt_includes_reference_metrics():
    adapter = AutoRAGAdapter()
    spec = adapter.build_search_spec(_exp(), {}, records=[_record("q", gt="A")])
    assert spec["gt_mode"] == "with_gt"
    assert "answer_correctness" in spec["metrics"]
    assert "context_recall" in spec["metrics"]
    assert spec["objective_metric"] == "answer_correctness"
    assert spec["yaml_template"]["evaluation"]["requires_ground_truth"] is True


def test_autorag_spec_no_gt_uses_reference_free_metrics():
    adapter = AutoRAGAdapter()
    spec = adapter.build_search_spec(_exp(), {}, records=[_record("q")])
    assert spec["gt_mode"] == "no_gt"
    assert "answer_correctness" not in spec["metrics"]
    assert "context_recall" not in spec["metrics"]
    assert "faithfulness" in spec["metrics"]
    assert spec["objective_metric"] == "faithfulness"
    assert spec["yaml_template"]["evaluation"]["requires_ground_truth"] is False


def test_autorag_spec_explicit_mode_overrides_records():
    adapter = AutoRAGAdapter()
    # records say "with GT", but explicit mode forces no_gt
    exp = _exp(parameters={"gt_mode": "no_gt"})
    spec = adapter.build_search_spec(exp, {}, records=[_record("q", gt="A")])
    assert spec["gt_mode"] == "no_gt"


def test_autorag_run_note_varies_by_mode():
    adapter = AutoRAGAdapter()
    with_gt = adapter.run(_exp(), {}, records=[_record("q", gt="A")])
    no_gt = adapter.run(_exp(), {}, records=[_record("q")])
    assert "with-GT" in with_gt.note
    assert "no-GT" in no_gt.note
    assert with_gt.payload["metrics"] != no_gt.payload["metrics"]


# --------------------------------------------------------------- dspy adapter
def test_dspy_spec_records_gt_mode():
    adapter = DSPyAdapter()
    spec = adapter.build_optimizer_spec(_exp(), {}, records=[_record("q", gt="A")])
    assert spec["gt_mode"] == "with_gt"
    assert spec["objective_metric"] == "token_f1"
    assert spec["compile_hints"]["metric_kind"] == "reference_based"


def test_dspy_spec_no_gt_uses_judge():
    adapter = DSPyAdapter()
    spec = adapter.build_optimizer_spec(_exp(), {}, records=[_record("q")])
    assert spec["gt_mode"] == "no_gt"
    assert spec["objective_metric"] == "llm_judge_faithfulness"
    assert spec["compile_hints"]["metric_kind"] == "reference_free_llm_judge"


def test_dspy_with_gt_metric_runs_without_dspy_installed():
    """The with-GT metric must work even when dspy isn't installed —
    it's pure Python token-F1."""
    adapter = DSPyAdapter()
    metric = adapter.build_metric_function("with_gt")

    class _Stub:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    example = _Stub(answer="the cat sat on the mat")
    pred_perfect = _Stub(answer="the cat sat on the mat")
    pred_partial = _Stub(answer="the cat ran away")
    pred_empty = _Stub(answer="")

    assert metric(example, pred_perfect) == pytest.approx(1.0)
    assert 0.0 < metric(example, pred_partial) < 1.0
    assert metric(example, pred_empty) == 0.0


def test_token_f1_symmetric_and_zero_safe():
    assert _token_f1("a b c", "a b c") == pytest.approx(1.0)
    assert _token_f1("", "anything") == 0.0
    assert _token_f1("foo", "") == 0.0
    assert _token_f1("x y", "y x") == pytest.approx(1.0)  # order-independent


def test_dspy_no_gt_metric_requires_judge_or_dspy():
    """In no-GT mode the metric needs an LLM-as-judge; without one we either
    raise ImportError (no dspy) or ValueError (dspy present, judge missing)."""
    adapter = DSPyAdapter()
    try:
        import dspy  # noqa: F401
    except Exception:
        with pytest.raises(ImportError, match="DSPy is required"):
            adapter.build_metric_function("no_gt")
    else:
        with pytest.raises(ValueError, match="LLM-as-judge"):
            adapter.build_metric_function("no_gt")


def test_dspy_custom_metric_pass_through():
    adapter = DSPyAdapter()

    def my_metric(example, pred, trace=None):
        return 0.42

    out = adapter.build_metric_function("no_gt", custom_metric=my_metric)
    assert out is my_metric


def test_dspy_build_trainset_requires_dspy():
    """The trainset builder explicitly imports dspy; when missing the error
    message points at the right install hint."""
    adapter = DSPyAdapter()
    try:
        import dspy  # noqa: F401
    except Exception:
        with pytest.raises(ImportError, match="DSPy is required"):
            adapter.build_trainset([_record("q", gt="A")])
    else:
        # When dspy IS installed, this should succeed.
        examples = adapter.build_trainset([_record("q", gt="A")])
        assert len(examples) == 1


def test_dspy_optimize_unknown_optimizer():
    """Unknown optimizer name should be rejected before we even try to import."""
    adapter = DSPyAdapter()
    try:
        import dspy  # noqa: F401
        from dspy.teleprompt import MIPROv2  # noqa: F401
    except Exception:
        pytest.skip("dspy not installed; optimize() raises ImportError before validation")
    with pytest.raises(ValueError, match="Unknown optimizer"):
        adapter.optimize([_record("q", gt="A")], optimizer="NotAReal")
