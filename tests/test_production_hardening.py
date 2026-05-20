"""Tests for the production-hardening changes.

Covers:
- Pydantic validator constraints
- RunStore atomic write & corrupt-file resilience
- Logging module idempotency
- Executor strict-execute behavior
- LangChain in-process runner end-to-end
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ragdx.optim.executor import OptimizationExecutor
from ragdx.optim.langchain_adapter import LangChainAdapter
from ragdx.schemas.models import (
    CausalSignal,
    DatasetRecord,
    DiagnosisHypothesis,
    EvaluationResult,
    OptimizationExperiment,
    OptimizationPlan,
)
from ragdx.storage.run_store import (
    RunNotFoundError,
    RunStore,
    SessionNotFoundError,
    atomic_write_text,
)
from ragdx.utils.logging import configure_logging, get_logger


# --------------------------------------------------------------------- #
# Pydantic validators
# --------------------------------------------------------------------- #
def test_evaluation_clips_out_of_range_bounded_metrics():
    # Bounded metric should be clamped to [0, 1], not raise.
    result = EvaluationResult(retrieval={"context_recall": 1.4})
    assert result.retrieval["context_recall"] == 1.0
    result2 = EvaluationResult(generation={"faithfulness": -0.3})
    assert result2.generation["faithfulness"] == 0.0


def test_evaluation_unbounded_metrics_pass_through():
    # Latency/cost are unbounded; preserve verbatim.
    result = EvaluationResult(generation={"latency_ms": 1234.5, "cost_usd": 0.0421})
    assert result.generation["latency_ms"] == 1234.5
    assert result.generation["cost_usd"] == 0.0421


def test_evaluation_rejects_non_numeric_metric():
    with pytest.raises(ValidationError):
        EvaluationResult(retrieval={"context_recall": "high"})


def test_causal_signal_probability_must_be_unit_interval():
    with pytest.raises(ValidationError):
        CausalSignal(node="x", component="retrieval", prior=1.5)
    with pytest.raises(ValidationError):
        CausalSignal(node="x", component="retrieval", posterior=-0.01)


def test_hypothesis_confidence_validated():
    with pytest.raises(ValidationError):
        DiagnosisHypothesis(component="retrieval", root_cause="x", confidence=1.2)


def test_experiment_rejects_negative_weight_and_self_dep():
    with pytest.raises(ValidationError):
        OptimizationExperiment(
            name="e1", tool="manual", target_component="retrieval",
            description="d", objectives={"answer_correctness": -0.1},
        )
    with pytest.raises(ValidationError):
        OptimizationExperiment(
            name="e1", tool="manual", target_component="retrieval",
            description="d", depends_on=["e1"],
        )


def test_plan_rejects_dangling_dependency():
    e1 = OptimizationExperiment(
        name="e1", tool="manual", target_component="retrieval",
        description="d", depends_on=["nope"],
    )
    with pytest.raises(ValidationError):
        OptimizationPlan(objective_metric="answer_correctness", experiments=[e1])


# --------------------------------------------------------------------- #
# atomic_write + locking + RunStore
# --------------------------------------------------------------------- #
def test_atomic_write_does_not_leave_tmp_files_on_success(tmp_path: Path):
    target = tmp_path / "out.json"
    atomic_write_text(target, '{"k": 1}')
    assert target.read_text() == '{"k": 1}'
    leftover = [p.name for p in tmp_path.iterdir() if p.name != "out.json"]
    assert leftover == [], f"Found leftover files: {leftover}"


def test_run_store_load_missing_raises_typed_error(tmp_path: Path):
    store = RunStore(root=tmp_path)
    with pytest.raises(RunNotFoundError):
        store.load_run("does-not-exist")
    with pytest.raises(SessionNotFoundError):
        store.load_session("nope")


def test_run_store_skips_corrupt_files_and_logs(tmp_path: Path, caplog):
    store = RunStore(root=tmp_path)
    # Write a malformed run JSON next to nothing.
    bad = store.runs_dir / "broken.json"
    bad.write_text("{not valid json}", encoding="utf-8")
    ragdx_logger = logging.getLogger("ragdx")
    ragdx_logger.addHandler(caplog.handler)
    try:
        caplog.set_level(logging.WARNING)
        runs = store.list_runs()
    finally:
        ragdx_logger.removeHandler(caplog.handler)
    assert runs == []
    assert any("corrupt run" in r.message.lower() for r in caplog.records)


def test_run_store_concurrent_writes_do_not_corrupt(tmp_path: Path):
    """Smoke test for the cross-process file lock + per-process RLock."""
    eval_ = EvaluationResult(retrieval={"context_recall": 0.5})
    from ragdx.schemas.models import DiagnosisReport
    diagnosis = DiagnosisReport(summary="s")
    plan = OptimizationPlan(objective_metric="answer_correctness")

    store = RunStore(root=tmp_path)
    barrier = threading.Barrier(8)
    saved_ids: list[str] = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        run = store.save_run(eval_, diagnosis, plan, name="t")
        with lock:
            saved_ids.append(run.run_id)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(saved_ids) == 8
    assert len(set(saved_ids)) == 8
    # All files should load cleanly (no half-written corruption).
    loaded = [store.load_run(rid) for rid in saved_ids]
    assert all(r.name == "t" for r in loaded)


# --------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------- #
def test_get_logger_returns_namespaced_child():
    root = get_logger()
    child = get_logger("ragdx.optim.executor")
    assert child.name == "ragdx.optim.executor"
    assert child.parent is root


def test_configure_logging_is_idempotent():
    a = configure_logging()
    n = len(a.handlers)
    configure_logging()
    assert len(a.handlers) == n


# --------------------------------------------------------------------- #
# Executor strict execute mode
# --------------------------------------------------------------------- #
def _trivial_plan() -> tuple[OptimizationPlan, EvaluationResult]:
    baseline = EvaluationResult(
        retrieval={"context_recall": 0.5},
        generation={"faithfulness": 0.5},
        e2e={"answer_correctness": 0.5},
    )
    exp = OptimizationExperiment(
        name="e1",
        tool="langchain",
        target_component="retrieval",
        description="trivial",
        parameters={"dataset_path": "examples/demo_dataset.jsonl"},
        objectives={"answer_correctness": 1.0},
        search_space={"top_k": [4, 6, 8]},
        max_trials=2,
        stage="retrieval",
    )
    plan = OptimizationPlan(objective_metric="answer_correctness", experiments=[exp])
    return plan, baseline


def test_strict_execute_fails_loudly_without_runner(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("RAGDX_LANGCHAIN_RUNNER_CMD", raising=False)
    monkeypatch.delenv("RAGDX_FALLBACK_SIMULATE_ON_MISSING_RUNNER", raising=False)
    plan, baseline = _trivial_plan()
    executor = OptimizationExecutor(root=tmp_path, strict_execute=True)
    session = executor.execute_plan(plan, baseline=baseline, mode="execute")
    assert session.status == "failed"
    assert all(t.status == "failed" for t in session.trials)
    assert any("strict execute" in (t.notes or "").lower() or "no runner" in " ".join(t.logs).lower()
               for t in session.trials)


def test_lenient_execute_falls_back_to_simulate(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("RAGDX_LANGCHAIN_RUNNER_CMD", raising=False)
    plan, baseline = _trivial_plan()
    executor = OptimizationExecutor(root=tmp_path, strict_execute=False)
    session = executor.execute_plan(plan, baseline=baseline, mode="execute")
    # With strict_execute=False, trials should complete with simulated scores.
    assert session.status == "completed"
    assert all(t.status == "done" and t.objective_scores for t in session.trials)
    assert all(
        any("simulated scoring" in line.lower() or "scored by simulator" in line.lower()
            for line in t.logs)
        for t in session.trials
    )


# --------------------------------------------------------------------- #
# LangChain in-process runner
# --------------------------------------------------------------------- #
def _toy_pipeline_factory(**params: Any):
    """A trivial 'pipeline' for testing — echoes ground truth as the answer
    with a configurable token-dropping probability based on top_k."""
    keep_ratio = min(1.0, params.get("top_k", 6) / 8.0)

    def call(record: DatasetRecord) -> dict[str, Any]:
        tokens = (record.ground_truth or "").split()
        keep = max(1, int(len(tokens) * keep_ratio))
        return {
            "answer": " ".join(tokens[:keep]),
            "contexts": [record.ground_truth or ""],
        }

    return call


def test_langchain_in_process_runner_e2e(tmp_path: Path, monkeypatch):
    records = [
        DatasetRecord(question="q1", ground_truth="the quick brown fox", answer=None,
                      contexts=["the quick brown fox"]),
        DatasetRecord(question="q2", ground_truth="lazy dog jumps", answer=None,
                      contexts=["lazy dog jumps"]),
    ]
    adapter = LangChainAdapter()

    # Stash the toy factory in this module so import-by-string resolves it.
    import sys
    sys.modules[__name__].create_pipeline = _toy_pipeline_factory  # type: ignore[attr-defined]
    factory_spec = f"{__name__}:create_pipeline"

    runner = adapter.make_in_process_runner(records=records)
    plan, baseline = _trivial_plan()
    plan.experiments[0].parameters["pipeline_module"] = factory_spec

    executor = OptimizationExecutor(
        root=tmp_path,
        in_process_runners={"langchain": runner},
    )
    session = executor.execute_plan(plan, baseline=baseline, mode="execute")
    assert session.status == "completed"
    assert all(t.status == "done" for t in session.trials)
    # Token recall should track top_k roughly monotonically: higher top_k → more tokens preserved.
    by_topk = sorted(session.trials, key=lambda t: t.parameters.get("top_k", 0))
    if len(by_topk) >= 2:
        assert by_topk[0].objective_scores["answer_correctness"] <= by_topk[-1].objective_scores["answer_correctness"] + 1e-9


def test_in_process_runner_handles_factory_error(tmp_path: Path):
    """A broken pipeline factory must mark the trial failed, not crash the session."""
    adapter = LangChainAdapter()
    bad_runner = adapter.make_in_process_runner(records=[
        DatasetRecord(question="q", ground_truth="g"),
    ])
    plan, baseline = _trivial_plan()
    plan.experiments[0].parameters["pipeline_module"] = "does_not_exist:create"
    executor = OptimizationExecutor(
        root=tmp_path, in_process_runners={"langchain": bad_runner},
    )
    session = executor.execute_plan(plan, baseline=baseline, mode="execute")
    assert session.status == "failed"
    assert all(t.status == "failed" for t in session.trials)


# --------------------------------------------------------------------- #
# Ragas adapter normalization-only path stays cheap and pure
# --------------------------------------------------------------------- #
def test_ragas_adapter_normalize_scores_no_install_needed():
    from ragdx.engines.ragas_adapter import RagasAdapter
    r = RagasAdapter().normalize_scores({"context_precision": 0.7, "faithfulness": 0.8})
    assert r.retrieval["context_precision"] == 0.7
    assert r.generation["faithfulness"] == 0.8
    assert r.raw_tool_outputs["ragas"] == {"context_precision": 0.7, "faithfulness": 0.8}
