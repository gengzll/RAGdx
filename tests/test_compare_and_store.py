from ragdx.core.compare import compare_results
from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import EvaluationResult
from ragdx.storage.run_store import RunStore


def test_compare_and_store(tmp_path):
    base = EvaluationResult(retrieval={"context_recall": 0.5}, generation={"hallucination": 0.3}, e2e={"answer_correctness": 0.6})
    cur = EvaluationResult(retrieval={"context_recall": 0.7}, generation={"hallucination": 0.2}, e2e={"answer_correctness": 0.75})
    cmp = compare_results(cur, base)
    assert any(x.metric == "context_recall" and x.direction == "improved" for x in cmp)
    assert any(x.metric == "hallucination" and x.direction == "improved" for x in cmp)

    engine = RAGDiagnosisEngine()
    report = engine.diagnose(cur)
    plan = OptimizationPlanner().build_plan(report)
    store = RunStore(tmp_path / ".ragdx")
    saved = store.save_run(cur, report, plan, name="test")
    loaded = store.load_run(saved.run_id)
    assert loaded.name == "test"
