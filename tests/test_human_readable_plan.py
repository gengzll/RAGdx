from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import EvaluationResult
from ragdx.utils.reporting import summarize_plan, summarize_target_spec


def _demo_result():
    return EvaluationResult(
        retrieval={"context_precision": 0.63, "context_recall": 0.57, "context_entities_recall": 0.54},
        generation={"faithfulness": 0.79, "response_relevancy": 0.82, "noise_sensitivity": 0.31, "context_utilization": 0.61},
        e2e={"answer_correctness": 0.68, "citation_accuracy": 0.71},
        metadata={"dataset": "demo", "tools": ["ragas", "ragchecker"]},
    )


def test_human_readable_plan_summary_contains_key_sections():
    result = _demo_result()
    report = RAGDiagnosisEngine().diagnose(result)
    plan = OptimizationPlanner().build_plan(report, result=result)
    text = summarize_plan(plan.model_dump())
    assert "Objective metric:" in text
    assert "Objective weights (trade-off coefficients, not metric targets):" in text
    assert "Constraint bounds:" in text


def test_target_spec_summary_mentions_baseline_and_target():
    spec = {
        "direction": "maximize",
        "mode": "improve",
        "baseline_value": 0.81,
        "target_value": 0.85,
        "delta_from_baseline": 0.04,
        "min_acceptable": 0.8,
    }
    text = summarize_target_spec("answer_correctness", spec)
    assert "baseline=0.8100" in text
    assert "target=0.8500" in text
    assert "min_acceptable=0.8000" in text
