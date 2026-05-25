"""Ground-truth detection and reference-aware metric selection.

A small utility shared by the AutoRAG / DSPy adapters AND by the evaluator
adapters so all of them adapt their behaviour to the data they're handed:

* When the trainset has populated ``ground_truth`` we can use reference-based
  metrics like ``answer_correctness`` and ``context_recall``.
* When ``ground_truth`` is missing or empty we fall back to reference-free
  signals (``faithfulness``, ``answer_relevancy``, ``context_precision``,
  ``hallucination``) — typically computed by an LLM-as-judge.

Beyond GT, evaluator adapters also need to know whether the records have
populated ``answer`` and ``contexts``. Without ``answer`` you cannot compute
any generation/E2E metric; without ``contexts`` you cannot compute retrieval
metrics. The helpers here let adapters skip those metrics honestly rather
than silently emitting 0.0.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from ragdx.schemas.models import DatasetRecord

GTMode = Literal["with_gt", "no_gt"]

# Metrics that REQUIRE a ground-truth reference to be meaningful.
# We list both the ragas vocabulary and the ragchecker vocabulary because
# evaluators receive their metric names verbatim.
REFERENCE_REQUIRED_METRICS: frozenset[str] = frozenset(
    {
        # ragas
        "answer_correctness",
        "answer_accuracy",
        "context_recall",
        "context_entity_recall",
        # ragchecker (uses shorter names; semantics are the same)
        "recall",  # fraction of GT claims covered by retrieved context
        "claim_recall",  # alias depending on ragchecker version
        "context_utilization",  # GT claims used in the answer
    }
)

# Metrics that work WITHOUT a ground-truth reference (LLM-as-judge / NLI).
# Note: ``context_utilization`` is intentionally absent — RAGChecker computes
# it against GT claims, so it is classified as reference-required below.
REFERENCE_FREE_METRICS: frozenset[str] = frozenset(
    {
        "faithfulness",
        "answer_relevancy",
        "response_relevancy",
        "context_precision",
        "hallucination",
        "self_knowledge",
        "noise_sensitivity",
    }
)

# Metrics that require the system-generated ``answer`` to be present (any
# generation- or E2E-layer metric). Without an answer we cannot judge whether
# claims are supported, relevant, hallucinated, etc.
ANSWER_REQUIRED_METRICS: frozenset[str] = frozenset(
    {
        # ragas
        "faithfulness",
        "answer_relevancy",
        "response_relevancy",
        "answer_correctness",
        "answer_accuracy",
        "noise_sensitivity",
        # ragchecker
        "precision",  # claims-in-answer supported by context
        "recall",
        "claim_recall",
        "hallucination",
        "self_knowledge",
        "context_utilization",
    }
)

# Metrics that require ``contexts`` (the retrieved chunks) to be present —
# both the retrieval-layer ones and the generation-layer ones that compare
# the answer against retrieved context.
CONTEXT_REQUIRED_METRICS: frozenset[str] = frozenset(
    {
        # ragas
        "context_precision",
        "context_recall",
        "context_entity_recall",
        "faithfulness",
        "noise_sensitivity",
        # ragchecker (all of its metrics inspect retrieved chunks)
        "precision",
        "recall",
        "claim_recall",
        "context_utilization",
        "hallucination",
        "self_knowledge",
    }
)


def has_ground_truth(records: Iterable[DatasetRecord], *, threshold: float = 0.5) -> bool:
    """Return True when at least ``threshold`` fraction of records have a non-empty GT."""
    records = list(records)
    if not records:
        return False
    have = sum(1 for r in records if (r.ground_truth or "").strip())
    return (have / len(records)) >= threshold


def has_answers(records: Iterable[DatasetRecord], *, threshold: float = 0.5) -> bool:
    """Return True when at least ``threshold`` fraction of records have a non-empty answer."""
    records = list(records)
    if not records:
        return False
    have = sum(1 for r in records if (r.answer or "").strip())
    return (have / len(records)) >= threshold


def has_contexts(records: Iterable[DatasetRecord], *, threshold: float = 0.5) -> bool:
    """Return True when at least ``threshold`` fraction of records have non-empty contexts."""
    records = list(records)
    if not records:
        return False
    have = sum(1 for r in records if r.contexts and any((c or "").strip() for c in r.contexts))
    return (have / len(records)) >= threshold


def gt_mode(records: Iterable[DatasetRecord], *, threshold: float = 0.5) -> GTMode:
    return "with_gt" if has_ground_truth(records, threshold=threshold) else "no_gt"


def select_metrics(
    mode: GTMode,
    *,
    extra: Sequence[str] | None = None,
    drop: Sequence[str] | None = None,
) -> list[str]:
    """Return the canonical metric list for a given GT mode."""
    chosen: list[str] = sorted(REFERENCE_FREE_METRICS)
    if mode == "with_gt":
        chosen = sorted(REFERENCE_FREE_METRICS | REFERENCE_REQUIRED_METRICS)
    if extra:
        for m in extra:
            if m not in chosen:
                chosen.append(m)
    if drop:
        skip = set(drop)
        chosen = [m for m in chosen if m not in skip]
    return chosen


def validate_metrics_for_mode(
    requested: Sequence[str],
    mode: GTMode,
) -> dict[str, str]:
    """Static validation: every requested metric must be in
    ``select_metrics(mode)``. Returns a ``{metric: reason}`` mapping for any
    metric that is *not* supported in the declared mode. Empty dict means OK.

    This is the canonical, declarative pre-flight: ``gt_mode`` -> fixed metric
    set, no peeking at records. Use it when the caller has already
    committed to a mode and you want to fail loudly on a bad request.
    """
    allowed = set(select_metrics(mode))
    errors: dict[str, str] = {}
    for m in requested:
        if m not in allowed:
            if m in REFERENCE_REQUIRED_METRICS and mode == "no_gt":
                errors[m] = f"metric '{m}' requires ground_truth; gt_mode='no_gt' has no GT"
            else:
                errors[m] = f"metric '{m}' is not in the canonical set for gt_mode='{mode}'"
    return errors


def filter_metrics_by_data(
    requested: Sequence[str],
    records: Iterable[DatasetRecord],
    *,
    threshold: float = 0.5,
) -> tuple[list[str], dict[str, str]]:
    """Defensive data-driven filter (kept for adapters that need to inspect
    records before paying for an LLM-judge call). Prefer
    :func:`validate_metrics_for_mode` for the declarative path.
    """
    records = list(records)
    has_gt = has_ground_truth(records, threshold=threshold)
    has_ans = has_answers(records, threshold=threshold)
    has_ctx = has_contexts(records, threshold=threshold)

    kept: list[str] = []
    skipped: dict[str, str] = {}
    for m in requested:
        if m in REFERENCE_REQUIRED_METRICS and not has_gt:
            skipped[m] = "no ground_truth in dataset"
            continue
        if m in ANSWER_REQUIRED_METRICS and not has_ans:
            skipped[m] = "no answer in dataset"
            continue
        if m in CONTEXT_REQUIRED_METRICS and not has_ctx:
            skipped[m] = "no contexts in dataset"
            continue
        kept.append(m)
    return kept, skipped


__all__ = [
    "ANSWER_REQUIRED_METRICS",
    "CONTEXT_REQUIRED_METRICS",
    "REFERENCE_FREE_METRICS",
    "REFERENCE_REQUIRED_METRICS",
    "GTMode",
    "filter_metrics_by_data",
    "gt_mode",
    "has_answers",
    "has_contexts",
    "has_ground_truth",
    "select_metrics",
    "validate_metrics_for_mode",
]
