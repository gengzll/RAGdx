"""Tests for the per-source diagnosis layers (Phase 2).

The rule-based analyzer populates ``DiagnosisReport.rule_based``; the
LLM explainer additionally populates ``llm_based`` and (in
``--use-both`` mode) ``synthesis``. The top-level fields still reflect
the active layer so legacy consumers keep working.
"""

from __future__ import annotations

import pytest

from ragdx.engines.llm_diagnosis import LLMDiagnosisExplainer
from ragdx.engines.root_cause import RuleBasedRootCauseAnalyzer
from ragdx.schemas.models import (
    DiagnosisLayer,
    DiagnosisReport,
    EvaluationResult,
)


def _eval_result() -> EvaluationResult:
    return EvaluationResult(
        retrieval={"context_precision": 0.55, "context_recall": 0.65},
        generation={"faithfulness": 0.72, "answer_relevancy": 0.7},
        e2e={"answer_correctness": 0.6, "citation_accuracy": 0.45},
        metadata={"test": True},
    )


# ============================================================ rule analyzer
def test_rule_analyzer_populates_rule_based_layer(tmp_path) -> None:
    """The rule-based analyzer must round-trip its own output into
    ``report.rule_based`` so downstream wrappers can preserve lineage."""
    analyzer = RuleBasedRootCauseAnalyzer(root=str(tmp_path))
    report = analyzer.analyze(_eval_result())

    assert report.rule_based is not None
    assert report.rule_based.source == "rule"
    assert report.active_source == "rule"
    # The layer carries the same headline content as the top-level fields.
    assert report.rule_based.summary == report.summary
    assert len(report.rule_based.hypotheses) == len(report.hypotheses)
    assert (
        len(report.rule_based.causal_signals) == len(report.causal_signals)
    )


def test_rule_analyzer_does_not_populate_llm_or_synthesis(tmp_path) -> None:
    """Pure rule run leaves ``llm_based`` / ``synthesis`` as None so
    the renderer knows not to show empty panels."""
    analyzer = RuleBasedRootCauseAnalyzer(root=str(tmp_path))
    report = analyzer.analyze(_eval_result())
    assert report.llm_based is None
    assert report.synthesis is None


# ============================================================ explain (LLM)
def _stub_llm_response() -> str:
    """Minimum valid LLM diagnosis response that passes Pydantic
    validation. Matches the schema described in DEFAULT_REFINE_PROMPT."""
    return (
        '{"summary":"LLM-rewritten summary",'
        '"hypotheses":[{"component":"retrieval","root_cause":"LLM hypothesis",'
        '"severity":"high","confidence":0.9,"evidence":[],"recommended_actions":[]}],'
        '"optimization_candidates":["llm_candidate"],'
        '"priority_actions":["LLM action"],'
        '"causal_signals":[],"expected_thresholds":{},"metric_gaps":{},'
        '"evaluator_agreement":{},"diagnosis_confidence":0.85,'
        '"disambiguation_actions":[]}'
    )


def test_explain_preserves_both_layers(tmp_path) -> None:
    """After ``LLMDiagnosisExplainer.explain``, both ``rule_based`` and
    ``llm_based`` are populated, ``active_source`` is ``llm``, and the
    top-level summary reflects the LLM output."""
    rule_report = RuleBasedRootCauseAnalyzer(root=str(tmp_path)).analyze(_eval_result())
    explainer = LLMDiagnosisExplainer(llm_callable=lambda _prompt: _stub_llm_response())

    refined = explainer.explain(_eval_result(), rule_report)

    assert refined.active_source == "llm"
    assert refined.rule_based is not None
    assert refined.llm_based is not None
    assert refined.synthesis is None
    # Top-level == LLM layer (active).
    assert refined.summary == "LLM-rewritten summary"
    assert refined.summary == refined.llm_based.summary
    # rule_based layer must preserve the original rule output unchanged.
    assert refined.rule_based.summary == rule_report.summary
    assert (
        len(refined.rule_based.hypotheses) == len(rule_report.hypotheses)
    )


# ============================================================ summarize_both
def test_summarize_both_populates_all_three_layers(tmp_path) -> None:
    """``summarize_both`` keeps rule, LLM, and synthesis side-by-side."""
    rule_report = RuleBasedRootCauseAnalyzer(root=str(tmp_path)).analyze(_eval_result())
    explainer = LLMDiagnosisExplainer(llm_callable=lambda _prompt: _stub_llm_response())
    llm_report = explainer.explain(_eval_result(), rule_report)

    # Use a different stub response for the synthesis step so we can
    # check that the synthesis layer carries that one (not the LLM one).
    def _synth_stub(_prompt: str) -> str:
        return (
            '{"summary":"Synthesised summary",'
            '"hypotheses":[],"optimization_candidates":[],'
            '"priority_actions":["combined action"],'
            '"causal_signals":[],"expected_thresholds":{},"metric_gaps":{},'
            '"evaluator_agreement":{},"diagnosis_confidence":0.92,'
            '"disambiguation_actions":[]}'
        )
    explainer.llm_callable = _synth_stub

    synth = explainer.summarize_both(_eval_result(), rule_report, llm_report)

    assert synth.active_source == "synthesis"
    assert synth.summary == "Synthesised summary"
    assert synth.rule_based is not None
    assert synth.llm_based is not None
    assert synth.synthesis is not None
    # Layers preserve each pipeline stage's view.
    assert synth.rule_based.summary == rule_report.summary
    assert synth.llm_based.summary == "LLM-rewritten summary"
    assert synth.synthesis.summary == "Synthesised summary"


# ============================================================ schema
def test_diagnosis_layer_round_trips() -> None:
    """``DiagnosisLayer`` survives a ``model_dump`` / ``model_validate``
    round trip without losing fields. Critical because we serialise
    these to JSON in the bundle."""
    layer = DiagnosisLayer(
        source="rule",
        summary="hi",
        priority_actions=["a"],
        diagnosis_confidence=0.5,
    )
    payload = layer.model_dump()
    rebuilt = DiagnosisLayer.model_validate(payload)
    assert rebuilt == layer


def test_diagnosis_report_layer_fields_default_to_none() -> None:
    """Backwards-compat: a freshly-constructed report has no layers
    populated unless the producer opts in."""
    rep = DiagnosisReport(summary="empty")
    assert rep.rule_based is None
    assert rep.llm_based is None
    assert rep.synthesis is None
    assert rep.active_source == "rule"


def test_diagnosis_report_with_layers_serialises_json() -> None:
    """Mixed-layer reports must survive JSON serialisation -- this is
    the path used by ``ragdx diagnose --use-llm > report.json``."""
    layer = DiagnosisLayer(source="rule", summary="rule summary")
    rep = DiagnosisReport(summary="rule summary", rule_based=layer)
    payload = rep.model_dump_json()
    assert "rule_based" in payload
    rebuilt = DiagnosisReport.model_validate_json(payload)
    assert rebuilt.rule_based == layer


# Skip the explain test if Pydantic version mismatch breaks the stub.
@pytest.mark.parametrize("source", ["rule", "llm", "synthesis"])
def test_layer_source_literal_accepts_all_three(source: str) -> None:
    """The Literal restriction on ``source`` must permit each known
    layer kind, and reject typos."""
    layer = DiagnosisLayer(source=source)  # type: ignore[arg-type]
    assert layer.source == source


def test_layer_source_literal_rejects_unknown() -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        DiagnosisLayer(source="bogus")  # type: ignore[arg-type]
