"""Tests that the evaluator adapters refuse to emit false scores.

Three classes of "fake scores" used to leak through:

1. ragas got reference="" for records missing ground_truth, then computed
   reference-based metrics like context_recall/answer_correctness against
   that empty string (producing a stable 0.0 that looked real).
2. ragchecker got gt_answer="" / response="" with the same effect — every
   per-record claim_recall came back as 0.0 by construction.
3. RAGChecker can't compute anything without an answer field, but the old
   code silently ran and returned NaN-or-zero averages.

The fixes use ``ragdx.optim._gt_helpers.filter_metrics_by_data`` to drop
unsupported metrics before evaluation runs, recording the skip reason in
``EvaluationResult.metadata["skipped_metrics"]``. These tests pin that
behaviour without needing the heavy ragas/ragchecker imports — we test the
helpers and the adapter pre-flight branches that don't require the optional
deps.
"""

from __future__ import annotations

import pytest

from ragdx.engines.ragas_adapter import RagasAdapter
from ragdx.engines.ragchecker_adapter import RAGCheckerAdapter
from ragdx.optim._gt_helpers import (
    ANSWER_REQUIRED_METRICS,
    REFERENCE_REQUIRED_METRICS,
    filter_metrics_by_data,
    has_answers,
    has_contexts,
)
from ragdx.schemas.models import DatasetRecord


def _record(
    *,
    q: str = "Q",
    a: str | None = "A",
    gt: str | None = None,
    ctxs: list[str] | None = None,
) -> DatasetRecord:
    return DatasetRecord(
        question=q,
        answer=a,
        ground_truth=gt,
        contexts=ctxs if ctxs is not None else ["ctx1"],
    )


# ---------------------------------------------------------------- helpers
def test_has_answers_majority():
    rs = [_record(a="A1"), _record(a="A2"), _record(a=None)]
    assert has_answers(rs) is True
    rs = [_record(a=None), _record(a=None), _record(a="A")]
    assert has_answers(rs) is False


def test_has_contexts_treats_empty_strings_as_missing():
    rs = [_record(ctxs=["c"]), _record(ctxs=["c"]), _record(ctxs=["", "  "])]
    assert has_contexts(rs) is True
    rs = [_record(ctxs=[]), _record(ctxs=[""]), _record(ctxs=[])]
    assert has_contexts(rs) is False


def test_filter_metrics_by_data_drops_gt_metrics_when_no_gt():
    rs = [_record(a="A", ctxs=["c"]) for _ in range(3)]  # no GT
    kept, skipped = filter_metrics_by_data(
        ["faithfulness", "context_recall", "answer_correctness", "context_precision"], rs
    )
    assert "faithfulness" in kept
    assert "context_precision" in kept
    assert "context_recall" not in kept
    assert "answer_correctness" not in kept
    assert "no ground_truth" in skipped["context_recall"]
    assert "no ground_truth" in skipped["answer_correctness"]


def test_filter_metrics_by_data_drops_answer_metrics_when_no_answer():
    rs = [_record(a=None, ctxs=["c"], gt="GT") for _ in range(3)]
    kept, skipped = filter_metrics_by_data(
        ["faithfulness", "context_precision", "context_recall"], rs
    )
    # context_precision needs contexts (present) but no answer
    assert "context_precision" in kept
    assert "context_recall" in kept  # GT present, no answer dependency
    # faithfulness needs answer → dropped
    assert "faithfulness" not in kept
    assert "no answer" in skipped["faithfulness"]


def test_filter_metrics_by_data_drops_context_metrics_when_no_context():
    rs = [_record(a="A", ctxs=[], gt="GT") for _ in range(3)]
    kept, skipped = filter_metrics_by_data(
        ["faithfulness", "context_precision", "answer_correctness"], rs
    )
    # answer_correctness only needs answer+GT, no contexts
    assert "answer_correctness" in kept
    assert "faithfulness" not in kept
    assert "context_precision" not in kept
    assert "no contexts" in skipped["faithfulness"]
    assert "no contexts" in skipped["context_precision"]


def test_ragchecker_specific_names_in_reference_required():
    # RAGChecker's `recall` and `context_utilization` need GT.
    assert "recall" in REFERENCE_REQUIRED_METRICS
    assert "context_utilization" in REFERENCE_REQUIRED_METRICS
    # `precision` does NOT need GT (it's claims-in-answer vs context).
    assert "precision" not in REFERENCE_REQUIRED_METRICS


def test_metric_classifications_internally_consistent():
    """Every metric requiring GT should also require an answer (you can't
    compare to a reference without something to compare). RAGChecker's
    retrieval-only metrics are the exception (context_recall etc. don't
    technically need the system answer)."""
    for m in REFERENCE_REQUIRED_METRICS:
        if m in {"context_recall", "context_entity_recall"}:
            continue  # retrieval-only
        assert m in ANSWER_REQUIRED_METRICS, f"{m} needs GT but not flagged as needing answer"


# ----------------------------------------------------------- ragas adapter
def test_ragas_metric_name_extraction_uses_name_attribute():
    """``_metric_name`` should prefer the .name attribute (real ragas metrics
    expose it) but fall back to class name for objects without it."""
    class WithName:
        name = "context_recall"

    class WithoutName:
        pass

    assert RagasAdapter._metric_name(WithName()) == "context_recall"
    assert RagasAdapter._metric_name(WithoutName()) == "withoutname"


def test_ragas_filter_metrics_by_data_drops_unsupported():
    class _M:
        def __init__(self, name): self.name = name

    metrics = [_M("faithfulness"), _M("context_recall"), _M("answer_correctness")]
    no_gt = [_record(a="A", ctxs=["c"]) for _ in range(3)]
    kept, skipped = RagasAdapter()._filter_metrics_by_data(metrics, no_gt)
    kept_names = [m.name for m in kept]
    assert "faithfulness" in kept_names
    assert "context_recall" not in kept_names
    assert "answer_correctness" not in kept_names
    assert "context_recall" in skipped
    assert "answer_correctness" in skipped


# ------------------------------------------------------ ragchecker adapter
def test_ragchecker_evaluate_raises_when_no_answers():
    """RAGChecker needs an answer for every metric — no answers should fail
    loudly rather than silently produce 0s."""
    pytest.importorskip("ragchecker")
    rs = [_record(a=None, ctxs=["c"], gt="GT") for _ in range(3)]
    with pytest.raises(RuntimeError, match="requires `answer`"):
        RAGCheckerAdapter()._evaluate_in_process(rs, evaluator=object())


def test_ragchecker_evaluate_raises_when_no_contexts():
    pytest.importorskip("ragchecker")
    rs = [_record(a="A", ctxs=[], gt="GT") for _ in range(3)]
    with pytest.raises(RuntimeError, match="requires `contexts`"):
        RAGCheckerAdapter()._evaluate_in_process(rs, evaluator=object())


def test_ragchecker_normalize_scores_unaffected_by_filtering():
    """Score normalization itself doesn't gate on data; only in-process eval does."""
    raw = {"precision": 0.5, "recall": 0.7, "faithfulness": 0.9}
    out = RAGCheckerAdapter().normalize_scores(raw)
    assert out.retrieval["context_precision"] == 0.5
    assert out.retrieval["context_recall"] == 0.7
    assert out.generation["faithfulness"] == 0.9


def test_ragchecker_raw_scores_path_no_data_check():
    """When raw_scores are passed, we skip in-process eval and the data
    pre-flight — caller is responsible for sanity."""
    rs = [_record(a=None, ctxs=[], gt=None) for _ in range(3)]
    out = RAGCheckerAdapter().evaluate(rs, raw_scores={"faithfulness": 0.5})
    assert out.metadata["mode"] == "precomputed"
    assert out.generation["faithfulness"] == 0.5


# -------------------------------------------- embedding-proxy stays correct
def test_embedding_eval_skips_metrics_when_inputs_missing():
    """The embedding-proxy evaluator was already well-behaved — pin its
    contract so regressions are caught."""
    from ragdx.engines.embedding_eval import EmbeddingEvaluator

    # Records without GT → no answer_correctness, no context_recall
    rs = [_record(q="What is X?", a="X is a thing.", gt=None, ctxs=["X is described."])]
    out = EmbeddingEvaluator().evaluate(rs)
    assert "answer_correctness" not in out.e2e
    assert "context_recall" not in out.retrieval
    # But context_precision / faithfulness / relevancy should be computed.
    assert "context_precision" in out.retrieval
    assert "faithfulness" in out.generation
    assert "response_relevancy" in out.generation


def test_embedding_eval_with_gt_includes_correctness():
    from ragdx.engines.embedding_eval import EmbeddingEvaluator

    rs = [_record(q="What is X?", a="X is a thing.", gt="X is a thing.", ctxs=["X."])]
    out = EmbeddingEvaluator().evaluate(rs)
    assert "answer_correctness" in out.e2e
    assert "context_recall" in out.retrieval
