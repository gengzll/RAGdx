from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.optim.executor import OptimizationExecutor
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import EvaluationResult, FeedbackEvent
from ragdx.storage.run_store import RunStore


def test_persistent_causal_priors_learn_from_feedback(tmp_path):
    root = tmp_path / ".ragdx"
    result = EvaluationResult(
        retrieval={"context_precision": 0.86, "context_recall": 0.50},
        generation={"faithfulness": 0.70, "context_utilization": 0.55, "hallucination": 0.28},
        e2e={"answer_correctness": 0.62, "citation_accuracy": 0.58},
        feedback_events=[
            FeedbackEvent(feedback_id="f1", kind="hallucination"),
            FeedbackEvent(feedback_id="f2", kind="hallucination"),
            FeedbackEvent(feedback_id="f3", kind="escalation"),
        ],
    )
    engine = RAGDiagnosisEngine()
    engine.analyzer.store = RunStore(root)
    report1 = engine.diagnose(result)
    priors = RunStore(root).load_causal_priors(engine.analyzer.base_priors)
    assert priors["grounding_defect"] >= report1.causal_graph.nodes[0].prior or priors["grounding_defect"] > engine.analyzer.base_priors["grounding_defect"]


def test_bayesian_executor_tracks_feasible_hypervolume(tmp_path):
    result = EvaluationResult(
        retrieval={"context_precision": 0.62, "context_recall": 0.56},
        generation={"faithfulness": 0.77, "response_relevancy": 0.79, "noise_sensitivity": 0.18, "hallucination": 0.10, "cost_usd": 0.02, "latency_ms": 0.3},
        e2e={"answer_correctness": 0.66, "citation_accuracy": 0.69},
    )
    report = RAGDiagnosisEngine().diagnose(result)
    plan = OptimizationPlanner().build_plan(report, result=result, strategy="bayesian", budget=8)
    session = OptimizationExecutor(root=tmp_path / ".ragdx").execute_plan(plan, baseline=result, strategy="bayesian", mode="simulate")
    assert session.hypervolume >= session.feasible_hypervolume >= 0.0
    assert session.best_trial_id
    assert any(t.feasible is not None for t in session.trials)
