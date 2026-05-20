from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import DiagnosisReport, EvaluationResult


def test_plan_includes_baseline_relative_fields_and_non_regressive_targets():
    planner = OptimizationPlanner()
    report = DiagnosisReport(
        summary="Chunking defect dominates retrieval quality.",
        expected_thresholds={"answer_correctness": 0.85, "context_precision": 0.8, "latency_ms": 50.0, "cost_usd": 0.11},
        metric_gaps={"context_precision": 0.03},
        optimization_candidates=["corpus_chunking_search"],
        priority_actions=["Fix chunking."],
        causal_signals=[],
    )
    result = EvaluationResult(
        retrieval={"context_recall": 0.86, "context_precision": 0.84},
        generation={"faithfulness": 0.9},
        e2e={"answer_correctness": 0.89, "citation_accuracy": 0.91, "latency_ms": 42.0, "cost_usd": 0.09},
        metadata={},
    )
    plan = planner.build_plan(report, result=result)
    exp = next(e for e in plan.experiments if e.name == "corpus-chunking-search")
    assert exp.baseline_score == 0.89
    assert exp.parameters["baseline_metrics"]["answer_correctness"] == 0.89
    assert exp.parameters["metric_directions"]["answer_correctness"] == "maximize"
    assert exp.parameters["metric_directions"]["latency_ms"] == "minimize"
    assert exp.parameters["target_thresholds"]["answer_correctness"] >= 0.89
    assert exp.parameters["target_thresholds"]["context_precision"] >= 0.84
    assert exp.parameters["target_thresholds"]["latency_ms"] <= 42.0
    assert exp.parameters["objective_weights"] == exp.objectives
    assert exp.parameters["target_specs"]["answer_correctness"]["mode"] in {"improve", "maintain"}
    assert "Objective weights are trade-off coefficients" in exp.notes
