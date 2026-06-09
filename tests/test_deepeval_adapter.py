"""Tests for the DeepEval adapter.

We test the adapter shape + normalization without requiring deepeval
itself to be installed (CI's ``.[dev]`` extra doesn't pull deepeval).
The "run_deepeval=True" path is only exercised when the deepeval
import succeeds; we skip otherwise.
"""

from __future__ import annotations

import pytest

from ragdx.core.normalization import DEEPEVAL_MAP
from ragdx.engines.deepeval_adapter import DeepEvalAdapter
from ragdx.schemas.models import DatasetRecord


def _sample_records() -> list[DatasetRecord]:
    return [
        DatasetRecord(
            question="What is X?",
            ground_truth="X is the first letter.",
            contexts=["X is the 24th letter of the English alphabet."],
            answer="X is the 24th letter.",
        ),
    ]


def test_normalize_scores_buckets_into_evaluation_result() -> None:
    adapter = DeepEvalAdapter()
    raw = {
        "Faithfulness": 0.9,
        "Answer Relevancy": 0.75,
        "Contextual Precision": 0.6,
        "MysteryMetric": 0.5,  # unknown -> goes only to raw_tool_outputs
    }
    result = adapter.normalize_scores(raw)

    assert result.generation.get("faithfulness") == pytest.approx(0.9)
    assert result.generation.get("answer_relevancy") == pytest.approx(0.75)
    assert result.retrieval.get("context_precision") == pytest.approx(0.6)
    # Unknown metric should NOT pollute the typed buckets.
    assert "mysterymetric" not in {k.lower() for k in result.retrieval}
    assert "mysterymetric" not in {k.lower() for k in result.generation}
    assert "mysterymetric" not in {k.lower() for k in result.e2e}
    # But it must survive in the raw payload for debugging.
    assert (
        result.raw_tool_outputs["deepeval"]["MysteryMetric"]
        == pytest.approx(0.5)
    )


def test_precomputed_path_attaches_metadata() -> None:
    adapter = DeepEvalAdapter()
    records = _sample_records()
    result = adapter.evaluate(
        records,
        raw_scores={"Faithfulness": 1.0},
    )
    assert result.metadata.get("mode") == "precomputed"
    assert result.metadata.get("record_count") == 1
    assert result.generation["faithfulness"] == pytest.approx(1.0)


def test_no_args_path_raises_when_deepeval_missing() -> None:
    """Without ``raw_scores`` and without ``run_deepeval=True``, the
    adapter still pre-flights an ``import deepeval`` so callers know
    whether the prepared-only path is viable."""
    adapter = DeepEvalAdapter()
    records = _sample_records()
    try:
        import deepeval  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="deepeval is not installed"):
            adapter.evaluate(records)
    else:
        # If deepeval IS installed, the prepared_only path returns
        # metadata only (no scores). Still useful to exercise.
        result = adapter.evaluate(records)
        assert result.metadata.get("mode") == "prepared_only"


def test_deepeval_map_covers_canonical_names() -> None:
    """The schema bucket assignment must match RAGAS_MAP for the
    overlapping metrics so downstream objectives / thresholds don't
    have to special-case the evaluator."""
    # context_precision must be a retrieval-bucket metric in both maps.
    assert DEEPEVAL_MAP["Contextual Precision"][0] == "retrieval"
    assert DEEPEVAL_MAP["Contextual Recall"][0] == "retrieval"
    assert DEEPEVAL_MAP["Faithfulness"][0] == "generation"
    assert DEEPEVAL_MAP["Answer Relevancy"][0] == "generation"


def test_to_test_cases_requires_deepeval() -> None:
    """``_to_test_cases`` uses ``deepeval.test_case.LLMTestCase``.
    Behaviour-document: if deepeval is missing, this raises ImportError
    (it's a private helper, but worth pinning the contract)."""
    adapter = DeepEvalAdapter()
    try:
        import deepeval  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            adapter._to_test_cases(_sample_records())
    else:
        cases = adapter._to_test_cases(_sample_records())
        assert len(cases) == 1
        c = cases[0]
        assert c.input == "What is X?"
        assert c.actual_output == "X is the 24th letter."
