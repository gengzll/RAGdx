"""Tests for the RAGCheckerAdapter.

Focus on the record-conversion contract: the adapter must produce dicts that
``ragchecker.RAGResults.from_dict`` can deserialize. Historically a missing
``query_id`` and a flat ``retrieved_context: list[str]`` made the in-process
path fail with ``KeyError('query_id')``.
"""

from __future__ import annotations

import pytest

from ragdx.engines.ragchecker_adapter import RAGCheckerAdapter
from ragdx.schemas.models import DatasetRecord


def _record(q: str, a: str, gt: str, ctxs: list[str]) -> DatasetRecord:
    return DatasetRecord(
        question=q, answer=a, ground_truth=gt, contexts=ctxs, reference_contexts=[],
    )


def test_to_ragchecker_records_has_query_id_and_structured_context():
    records = [
        _record("Q1", "A1", "GT1", ["ctx1a", "ctx1b"]),
        _record("Q2", "A2", "GT2", ["ctx2a"]),
    ]
    rows = RAGCheckerAdapter()._to_ragchecker_records(records)

    assert len(rows) == 2
    for i, row in enumerate(rows):
        # query_id must be a non-empty string (RAGResult requires it).
        assert isinstance(row["query_id"], str) and row["query_id"]
        # query_id must be unique per record.
        assert row["query_id"] == f"q{i}"
        # retrieved_context must be list of dicts with doc_id + text.
        assert isinstance(row["retrieved_context"], list)
        for doc in row["retrieved_context"]:
            assert isinstance(doc, dict)
            assert "doc_id" in doc and "text" in doc
            assert isinstance(doc["text"], str)

    # Spot-check the actual text.
    assert rows[0]["retrieved_context"][0]["text"] == "ctx1a"
    assert rows[0]["retrieved_context"][1]["text"] == "ctx1b"
    assert rows[1]["retrieved_context"][0]["text"] == "ctx2a"


def test_to_ragchecker_records_query_ids_unique():
    records = [_record(f"Q{i}", "A", "GT", ["c"]) for i in range(5)]
    rows = RAGCheckerAdapter()._to_ragchecker_records(records)
    ids = [row["query_id"] for row in rows]
    assert len(ids) == len(set(ids)), "query_id values must be unique across records"


def test_to_ragchecker_records_handles_empty_contexts():
    rows = RAGCheckerAdapter()._to_ragchecker_records([_record("Q", "A", "GT", [])])
    assert rows[0]["retrieved_context"] == []


def test_records_roundtrip_through_ragresults_from_dict():
    """The dicts must be valid input to ``RAGResults.from_dict`` — this is the
    real bug we fixed (dataclasses_json raised KeyError without query_id)."""
    pytest.importorskip("ragchecker")
    from ragchecker import RAGResults  # type: ignore[import-not-found]

    records = [
        _record("Q1", "A1", "GT1", ["c1", "c2"]),
        _record("Q2", "A2", "GT2", []),
    ]
    rows = RAGCheckerAdapter()._to_ragchecker_records(records)
    results = RAGResults.from_dict({"results": rows})
    assert len(results.results) == 2
    assert results.results[0].query_id == "q0"
    assert results.results[1].query_id == "q1"
    # retrieved_context deserialized as RetrievedDoc objects with text.
    assert results.results[0].retrieved_context[0].text == "c1"
    assert results.results[0].retrieved_context[1].text == "c2"
    assert results.results[1].retrieved_context == []


def test_normalize_scores_maps_to_buckets():
    raw = {
        "precision": 0.5,
        "recall": 0.7,
        "faithfulness": 0.9,
        "hallucination": 0.1,
        "claim_recall": 0.6,
    }
    out = RAGCheckerAdapter().normalize_scores(raw)
    assert out.retrieval["context_precision"] == 0.5
    assert out.retrieval["context_recall"] == 0.7
    assert out.generation["faithfulness"] == 0.9
    assert out.generation["hallucination"] == 0.1
    assert out.e2e["answer_correctness"] == 0.6
    # raw_tool_outputs preserves the original keys verbatim.
    assert out.raw_tool_outputs["ragchecker"] == raw


def test_evaluate_with_raw_scores_skips_ragchecker_import():
    records = [_record("Q", "A", "GT", ["c"])]
    out = RAGCheckerAdapter().evaluate(records, raw_scores={"precision": 0.5})
    assert out.metadata["mode"] == "precomputed"
    assert out.metadata["record_count"] == 1
    assert out.retrieval["context_precision"] == 0.5
