"""Ground-truth detection and reference-aware metric selection.

A small utility shared by the AutoRAG and DSPy adapters so both adapt their
behaviour to the data they're handed:

* When the trainset has populated ``ground_truth`` we can use reference-based
  metrics like ``answer_correctness`` and ``context_recall``.
* When ``ground_truth`` is missing or empty we fall back to reference-free
  signals (``faithfulness``, ``answer_relevancy``, ``context_precision``,
  ``hallucination``) — typically computed by an LLM-as-judge.

The motivation is that real "cold-start" RAG projects rarely ship with a
labelled test set; we want the optimisation surface to keep working anyway and
to be honest about what it's optimising against.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from ragdx.schemas.models import DatasetRecord

GTMode = Literal["with_gt", "no_gt"]

# Metrics that REQUIRE a ground-truth reference to be meaningful.
REFERENCE_REQUIRED_METRICS: frozenset[str] = frozenset(
    {
        "answer_correctness",
        "answer_accuracy",
        "context_recall",
        "context_entity_recall",
        "claim_recall",  # ragchecker — derived from reference claims
    }
)

# Metrics that work WITHOUT a ground-truth reference (LLM-as-judge / NLI).
REFERENCE_FREE_METRICS: frozenset[str] = frozenset(
    {
        "faithfulness",
        "answer_relevancy",
        "response_relevancy",
        "context_precision",
        "context_utilization",
        "hallucination",
        "self_knowledge",
        "noise_sensitivity",
    }
)


def has_ground_truth(records: Iterable[DatasetRecord], *, threshold: float = 0.5) -> bool:
    """Return True when at least ``threshold`` fraction of records have a non-empty GT.

    A single dataset can be "mostly labelled" but contain a few stragglers;
    we accept the dataset as GT-capable as long as the majority is labelled.
    Pass ``threshold=1.0`` to require every record to be labelled.
    """
    records = list(records)
    if not records:
        return False
    have = sum(1 for r in records if (r.ground_truth or "").strip())
    return (have / len(records)) >= threshold


def gt_mode(records: Iterable[DatasetRecord], *, threshold: float = 0.5) -> GTMode:
    """Return ``"with_gt"`` or ``"no_gt"`` for the trainset."""
    return "with_gt" if has_ground_truth(records, threshold=threshold) else "no_gt"


def select_metrics(
    mode: GTMode,
    *,
    extra: Sequence[str] | None = None,
    drop: Sequence[str] | None = None,
) -> list[str]:
    """Return the canonical metric list for a given GT mode.

    Reference-free metrics are always included (they're useful regardless of
    label availability). Reference-based metrics are added only when ``mode``
    is ``"with_gt"``.

    ``extra`` lets a caller append project-specific metrics; ``drop`` removes
    items from the default set (e.g. when an evaluator doesn't implement one).
    """
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


__all__ = [
    "REFERENCE_FREE_METRICS",
    "REFERENCE_REQUIRED_METRICS",
    "GTMode",
    "gt_mode",
    "has_ground_truth",
    "select_metrics",
]
