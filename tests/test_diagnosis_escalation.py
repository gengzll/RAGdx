"""Tests for history-aware diagnosis escalation.

When a defect persists *after* the optimization that targets it has
already run, the analyzer should escalate its recommendation instead
of repeating the same advice. These tests pin that behaviour.
"""

from __future__ import annotations

from ragdx.engines.root_cause import (
    _ESCALATION_LADDERS,
    RuleBasedRootCauseAnalyzer,
    _escalation_level,
)
from ragdx.schemas.models import EvaluationResult


def _precision_below_threshold() -> EvaluationResult:
    """A result where retrieval precision is the bottleneck."""
    return EvaluationResult(
        retrieval={"context_precision": 0.55, "context_recall": 0.85},
        generation={"faithfulness": 0.95},
        e2e={},
    )


# ============================================================ _escalation_level
def test_escalation_level_climbs_with_repeated_attempts() -> None:
    assert _escalation_level("retrieval_precision_defect", []) == 0
    assert _escalation_level(
        "retrieval_precision_defect", ["autorag_pipeline_search"]
    ) == 1
    assert _escalation_level(
        "retrieval_precision_defect",
        ["autorag_pipeline_search", "autorag_pipeline_search"],
    ) == 2


def test_escalation_level_clamps_at_ladder_top() -> None:
    # Three attempts but the ladder only has 3 rungs (0,1,2) -> clamps at 2.
    lvl = _escalation_level(
        "retrieval_precision_defect",
        ["autorag_pipeline_search"] * 5,
    )
    assert lvl == len(_ESCALATION_LADDERS["retrieval_precision_defect"]) - 1


def test_escalation_level_ignores_unrelated_candidates() -> None:
    # A prompt-tune doesn't count toward a retrieval defect's level.
    assert _escalation_level(
        "retrieval_precision_defect", ["dspy_prompt_optimization"]
    ) == 0


def test_unknown_defect_has_no_ladder() -> None:
    assert _escalation_level("some_unknown_defect", ["x", "y"]) == 0


# ============================================================ analyze()
def test_baseline_diagnosis_gives_base_level_advice() -> None:
    an = RuleBasedRootCauseAnalyzer()
    rep = an.analyze(_precision_below_threshold())
    retrieval_hyps = [h for h in rep.hypotheses if h.component == "retrieval"]
    assert retrieval_hyps, "expected a retrieval hypothesis"
    h = retrieval_hyps[0]
    assert h.root_cause == "retrieval noise or weak ranking quality"
    # Base advice mentions a reranker.
    assert any("reranker" in a.lower() for a in h.recommended_actions)


def test_rediagnosis_after_failed_tune_escalates() -> None:
    an = RuleBasedRootCauseAnalyzer()
    rep = an.analyze(
        _precision_below_threshold(),
        optimization_history=["autorag_pipeline_search"],
    )
    retrieval_hyps = [h for h in rep.hypotheses if h.component == "retrieval"]
    assert retrieval_hyps
    h = retrieval_hyps[0]
    # Root cause should signal escalation, not repeat the base phrasing.
    assert "escalate" in h.root_cause.lower()
    # Escalated advice introduces a new lever (hybrid retrieval).
    joined = " ".join(h.recommended_actions).lower()
    assert "hybrid" in joined or "cross-encoder" in joined
    # The candidate set grows to include corpus chunking.
    assert "corpus_chunking_search" in rep.optimization_candidates
    # Evidence records that the lever was already pulled.
    assert any("already applied" in e.lower() for e in h.evidence)


def test_escalation_level_two_targets_corpus() -> None:
    an = RuleBasedRootCauseAnalyzer()
    rep = an.analyze(
        _precision_below_threshold(),
        optimization_history=["autorag_pipeline_search"] * 2,
    )
    retrieval_hyps = [h for h in rep.hypotheses if h.component == "retrieval"]
    assert retrieval_hyps
    joined = " ".join(retrieval_hyps[0].recommended_actions).lower()
    # Level 2 moves to the corpus / chunking tier.
    assert "re-chunk" in joined or "chunk boundaries" in joined


def test_grounding_defect_escalates_independently() -> None:
    an = RuleBasedRootCauseAnalyzer()
    result = EvaluationResult(
        retrieval={"context_precision": 0.9, "context_recall": 0.85},
        generation={"faithfulness": 0.6},  # below threshold
        e2e={},
    )
    rep = an.analyze(result, optimization_history=["dspy_prompt_optimization"])
    gen_hyps = [h for h in rep.hypotheses if h.component == "generation"]
    assert gen_hyps
    assert "escalate" in gen_hyps[0].root_cause.lower()
    joined = " ".join(gen_hyps[0].recommended_actions).lower()
    assert "verification" in joined or "stronger" in joined
