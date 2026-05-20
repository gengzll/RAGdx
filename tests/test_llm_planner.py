from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import DiagnosisReport, EvaluationResult


def _make_report():
    return DiagnosisReport(
        summary="Retrieval is decent but generation lags.",
        expected_thresholds={"context_recall": 0.8, "faithfulness": 0.9, "hallucination": 0.08, "latency_ms": 1200.0},
        metric_gaps={"faithfulness": 0.08, "hallucination": 0.02},
        optimization_candidates=["autorag_pipeline_search", "dspy_prompt_optimization"],
        priority_actions=["Improve grounding."],
        causal_signals=[],
    )


def test_dynamic_constraints_respect_strong_baseline():
    planner = OptimizationPlanner()
    result = EvaluationResult(
        retrieval={"context_recall": 0.91, "context_precision": 0.88},
        generation={"faithfulness": 0.83, "hallucination": 0.04},
        e2e={"answer_correctness": 0.84, "citation_accuracy": 0.86},
        metadata={},
    )
    constraints = planner._constraints(result=result, report=_make_report())
    assert constraints["hallucination_max"] <= 0.05
    assert constraints["latency_ms_max"] <= 250.0 + 1200.0


def test_llm_planner_can_override_weights_and_targets():
    def fake_llm(_prompt: str):
        return {
            "global_rationale": ["Baseline recall is already strong, so keep retrieval lighter and push generation."],
            "component_guidance": {
                "generation": {
                    "objective_weights": {"faithfulness": 0.5, "answer_correctness": 0.3, "citation_accuracy": 0.2},
                    "target_thresholds": {"faithfulness": 0.92},
                    "max_trials": 7,
                    "notes": "Focus on grounding and citation discipline.",
                }
            },
        }

    planner = OptimizationPlanner(llm_callable=fake_llm)
    result = EvaluationResult(
        retrieval={"context_recall": 0.91, "context_precision": 0.88},
        generation={"faithfulness": 0.83, "hallucination": 0.04, "response_relevancy": 0.86},
        e2e={"answer_correctness": 0.84, "citation_accuracy": 0.86},
        metadata={},
    )
    plan = planner.build_plan(_make_report(), result=result)
    generation = next(exp for exp in plan.experiments if exp.stage == "generation")
    assert generation.objectives["faithfulness"] == 0.5
    assert generation.parameters["target_thresholds"]["faithfulness"] == 0.92
    assert generation.max_trials == 7
    assert "LLM planner notes" in generation.notes
    assert any("Baseline recall is already strong" in item for item in plan.rationale)
