from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import EvaluationResult, EvaluatorCalibration, EvaluatorScore, FeedbackEvent, QueryTrace
from ragdx.storage.run_store import RunStore


def test_feedback_and_causal_signals(tmp_path):
    result = EvaluationResult(
        retrieval={"context_precision": 0.83, "context_recall": 0.58},
        generation={"faithfulness": 0.74, "hallucination": 0.21, "context_utilization": 0.60},
        e2e={"answer_correctness": 0.67, "citation_accuracy": 0.62},
        metadata={"dataset_shift": True},
        traces=[
            QueryTrace(trace_id="t1", question="q1", answer="a1", retrieved_chunks=[{"id": 1}], citations=[], latency_ms=1200),
            QueryTrace(trace_id="t2", question="q2", answer="a2", retrieved_chunks=[{"id": 2}], citations=[1], latency_ms=1400),
        ],
        feedback_events=[
            FeedbackEvent(feedback_id="f1", kind="hallucination", note="unsupported claim"),
            FeedbackEvent(feedback_id="f2", kind="escalation", note="manual review needed"),
        ],
        evaluator_scores=[
            EvaluatorScore(evaluator="ragas", metric="faithfulness", score=0.74),
            EvaluatorScore(evaluator="ragchecker", metric="faithfulness", score=0.64),
        ],
        calibrations=[EvaluatorCalibration(metric="faithfulness", agreement_score=0.72, audit_sample_size=10)],
    )
    report = RAGDiagnosisEngine().diagnose(result)
    assert report.causal_signals
    assert report.diagnosis_confidence > 0
    assert report.evaluator_agreement["faithfulness"] == 0.72

    plan = OptimizationPlanner().build_plan(report, result=result, strategy="bayesian", budget=8)
    assert any(exp.stage in {"corpus", "retrieval", "generation", "orchestration"} for exp in plan.experiments)

    store = RunStore(tmp_path / ".ragdx")
    saved = store.save_run(result, report, plan, name="with-feedback")
    store.attach_feedback_to_run(saved.run_id, [FeedbackEvent(feedback_id="f3", kind="thumbs_down", rating=0.0)])
    summary = store.feedback_summary()
    assert summary["total_feedback"] >= 1.0


def test_causal_graph_has_edges_and_feedback_adjusted_priors():
    result = EvaluationResult(
        retrieval={"context_precision": 0.88, "context_recall": 0.52},
        generation={"faithfulness": 0.72, "context_utilization": 0.58, "hallucination": 0.25},
        e2e={"answer_correctness": 0.63, "citation_accuracy": 0.60},
        metadata={"document_structure_preserved": False, "causal_prior_updates": {"distribution_shift": 0.2}},
        feedback_events=[
            FeedbackEvent(feedback_id="f1", kind="hallucination"),
            FeedbackEvent(feedback_id="f2", kind="hallucination"),
            FeedbackEvent(feedback_id="f3", kind="escalation"),
        ],
    )
    report = RAGDiagnosisEngine().diagnose(result)
    assert report.causal_graph.edges
    node_names = {n.node for n in report.causal_graph.nodes}
    assert "grounding_defect" in node_names
    dist = next(n for n in report.causal_graph.nodes if n.node == "distribution_shift")
    assert dist.prior >= 0.2
    assert any("Upstream propagation" in ev for n in report.causal_graph.nodes for ev in n.evidence)
