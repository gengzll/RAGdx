from ragdx.engines.llm_diagnosis import LLMDiagnosisExplainer
from ragdx.schemas.models import DiagnosisReport, EvaluationResult


def test_llm_diagnosis_normalizes_hypothesis_components():
    result = EvaluationResult(
        retrieval={"context_recall": 0.5},
        generation={"hallucination": 0.2},
        e2e={"answer_correctness": 0.7},
    )
    rule_report = DiagnosisReport(
        summary="rule report",
        expected_thresholds={},
        metric_gaps={},
        hypotheses=[],
        optimization_candidates=[],
        priority_actions=[],
    )
    llm_report = DiagnosisReport(
        summary="llm report",
        expected_thresholds={},
        metric_gaps={},
        hypotheses=[],
        optimization_candidates=[],
        priority_actions=[],
    )

    def fake_llm(prompt: str):
        return {
            "summary": "Combined diagnosis",
            "expected_thresholds": {},
            "metric_gaps": {},
            "hypotheses": [
                {
                    "component": "e2e / citation mapping",
                    "root_cause": "Citation handling is weak.",
                    "severity": "medium",
                    "confidence": 0.7,
                    "evidence": ["citation_accuracy is low."],
                    "recommended_actions": ["Improve citation mapping."],
                },
                {
                    "component": "pipeline / observability",
                    "root_cause": "Pipeline visibility is lacking.",
                    "severity": "high",
                    "confidence": 0.8,
                    "evidence": ["missing per-stage telemetry."],
                    "recommended_actions": ["Add stage-level logging."],
                },
            ],
            "optimization_candidates": [],
            "priority_actions": [],
        }

    explainer = LLMDiagnosisExplainer(fake_llm)
    combined = explainer.summarize_both(result, rule_report, llm_report)

    assert combined.hypotheses[0].component == "e2e"
    assert combined.hypotheses[1].component == "pipeline"
