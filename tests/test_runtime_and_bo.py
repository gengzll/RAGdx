import os

from ragdx.optim.executor import OptimizationExecutor
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import DiagnosisReport, EvaluationResult


def test_planner_adds_runtime_validation_experiment():
    result = EvaluationResult(
        retrieval={"context_precision": 0.6, "context_recall": 0.55},
        generation={"faithfulness": 0.75, "hallucination": 0.14},
        e2e={"answer_correctness": 0.7, "citation_accuracy": 0.68},
        metadata={"runtime_framework": "langchain", "dataset_path": "examples/demo_dataset.jsonl", "pipeline_module": "examples.langchain_pipeline:create_pipeline"},
    )
    report = DiagnosisReport(summary="mixed", optimization_candidates=["autorag_pipeline_search"], causal_signals=[])
    plan = OptimizationPlanner().build_plan(report, result=result, budget=6)
    assert any(e.tool == "langchain" for e in plan.experiments)


def test_runner_template_supports_langchain_and_llamaindex(monkeypatch):
    monkeypatch.setenv("RAGDX_LANGCHAIN_RUNNER_CMD", "python examples/run_langchain_trial.py --config {config} --output {output}")
    monkeypatch.setenv("RAGDX_LLAMAINDEX_RUNNER_CMD", "python examples/run_llamaindex_trial.py --config {config} --output {output}")
    ex = OptimizationExecutor()
    assert ex._runner_template("langchain")
    assert ex._runner_template("llamaindex")
