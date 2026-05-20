"""Tests for the dependency-free EmbeddingEvaluator."""

from __future__ import annotations

from ragdx.engines.embedding_eval import EmbeddingEvaluator
from ragdx.schemas.models import DatasetRecord


def _record(question: str, answer: str, ground_truth: str, contexts: list[str], ref: list[str] | None = None) -> DatasetRecord:
    return DatasetRecord(
        question=question,
        answer=answer,
        ground_truth=ground_truth,
        contexts=contexts,
        reference_contexts=ref or [],
    )


def test_empty_dataset_returns_metadata_only():
    out = EmbeddingEvaluator().evaluate([])
    assert out.retrieval == {}
    assert out.generation == {}
    assert out.e2e == {}
    assert out.metadata["record_count"] == 0


def test_high_overlap_yields_high_scores():
    records = [
        _record(
            question="What is the capital of France?",
            answer="The capital of France is Paris.",
            ground_truth="Paris is the capital of France.",
            contexts=["Paris is the capital and largest city of France."],
        ),
    ]
    result = EmbeddingEvaluator().evaluate(records)
    # Every bucket should be populated
    assert result.retrieval.get("context_precision", 0) > 0.1
    assert result.retrieval.get("context_recall", 0) > 0.1
    assert result.generation.get("faithfulness", 0) > 0.1
    assert result.generation.get("response_relevancy", 0) > 0.1
    assert result.e2e.get("answer_correctness", 0) > 0.1


def test_low_overlap_yields_low_scores():
    records = [
        _record(
            question="What is the capital of France?",
            answer="Bananas grow in tropical climates.",
            ground_truth="Paris is the capital of France.",
            contexts=["The Eiffel Tower is in Paris."],
        ),
    ]
    result = EmbeddingEvaluator().evaluate(records)
    # End-to-end correctness should be very low for wildly off-topic answer
    assert result.e2e.get("answer_correctness", 1.0) < 0.3


def test_metrics_are_bounded_zero_to_one():
    records = [
        _record(
            question="What is two plus two?",
            answer="Four.",
            ground_truth="Four.",
            contexts=["Two plus two equals four."],
        ),
    ]
    result = EmbeddingEvaluator().evaluate(records)
    for bucket in (result.retrieval, result.generation, result.e2e):
        for v in bucket.values():
            assert 0.0 <= v <= 1.0


def test_uses_reference_contexts_when_ground_truth_missing():
    records = [
        DatasetRecord(
            question="What color is the sky?",
            answer="Blue.",
            contexts=["The sky appears blue due to Rayleigh scattering."],
            reference_contexts=["Blue sky color due to Rayleigh scattering of sunlight."],
        ),
    ]
    result = EmbeddingEvaluator().evaluate(records)
    assert "context_recall" in result.retrieval
