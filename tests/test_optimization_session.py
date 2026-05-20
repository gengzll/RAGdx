from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.optim.executor import OptimizationExecutor
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import EvaluationResult
from ragdx.storage.run_store import RunStore


def test_optimize_session_simulate(tmp_path):
    result = EvaluationResult(
        retrieval={"context_precision": 0.60, "context_recall": 0.55},
        generation={"faithfulness": 0.78, "response_relevancy": 0.80, "noise_sensitivity": 0.30},
        e2e={"answer_correctness": 0.65, "citation_accuracy": 0.68},
        metadata={"dataset": "test"},
    )
    report = RAGDiagnosisEngine().diagnose(result)
    plan = OptimizationPlanner().build_plan(report, result=result, strategy="pareto_evolutionary", budget=6)
    session = OptimizationExecutor(root=tmp_path / ".ragdx").execute_plan(
        plan, baseline=result, strategy="pareto_evolutionary", mode="simulate"
    )
    assert session.completed_trials == session.total_trials
    assert session.trials
    assert any(t.config_path for t in session.trials)
    assert session.pareto_front_ids

    store = RunStore(tmp_path / ".ragdx")
    store.save_session(session)
    loaded = store.load_session(session.session_id)
    assert loaded.session_id == session.session_id


def test_constraint_enforcement_and_feasible_best_trial(tmp_path):
    result = EvaluationResult(
        retrieval={"context_precision": 0.62, "context_recall": 0.58},
        generation={"faithfulness": 0.78, "response_relevancy": 0.8, "noise_sensitivity": 0.16, "hallucination": 0.10, "cost_usd": 0.02, "latency_ms": 0.3},
        e2e={"answer_correctness": 0.66, "citation_accuracy": 0.68},
        metadata={"dataset": "test"},
    )
    report = RAGDiagnosisEngine().diagnose(result)
    plan = OptimizationPlanner().build_plan(report, result=result, strategy="pareto_evolutionary", budget=6)
    executor = OptimizationExecutor(root=tmp_path / ".ragdx")
    session = executor.execute_plan(plan, baseline=result, strategy="pareto_evolutionary", mode="simulate")
    feasible, violations, penalty = executor._evaluate_constraints({"hallucination": 0.2, "cost_usd": 0.08}, {"hallucination_max": 0.12, "cost_usd_max": 0.04})
    assert feasible is False
    assert violations
    assert penalty > 0
    assert session.feasible_pareto_front_ids
    best = next(t for t in session.trials if t.trial_id == session.best_trial_id)
    assert best.feasible is True
