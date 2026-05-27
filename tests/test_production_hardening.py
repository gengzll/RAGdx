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


def test_in_process_runner_refuses_missing_dataset(tmp_path: Path):
    """No silent fallback to examples/demo_dataset.jsonl: when neither
    explicit records, explicit dataset_path, nor experiment.parameters
    carry a dataset, the runner must raise — not score the user's
    pipeline against demo data."""
    from ragdx.optim.langchain_adapter import LangChainAdapter
    from ragdx.schemas.models import (
        EvaluationResult as _ER,
    )
    from ragdx.schemas.models import (
        OptimizationExperiment as _OE,
    )
    from ragdx.schemas.models import (
        OptimizationTrial as _OT,
    )

    adapter = LangChainAdapter()
    runner = adapter.make_in_process_runner()  # no records, no dataset_path
    exp = _OE(
        name="e", tool="langchain", target_component="retrieval",
        description="d", parameters={"pipeline_module": "x:y"},
        objectives={"answer_correctness": 1.0},
        search_space={"top_k": [4]}, max_trials=1, stage="retrieval",
    )
    trial = _OT(
        trial_id="t1", experiment_name="e",
        tool="langchain", strategy="bayesian",
        parameters={"top_k": 4},
    )
    with pytest.raises(ValueError, match="no dataset configured"):
        runner(experiment=exp, trial=trial, baseline=_ER())


def test_in_process_runner_refuses_missing_pipeline_module(tmp_path: Path):
    """Same guarantee for the pipeline factory: no silent fallback to
    the bundled examples module."""
    from ragdx.optim.langchain_adapter import LangChainAdapter
    from ragdx.schemas.models import (
        EvaluationResult as _ER,
    )
    from ragdx.schemas.models import (
        OptimizationExperiment as _OE,
    )
    from ragdx.schemas.models import (
        OptimizationTrial as _OT,
    )

    adapter = LangChainAdapter()
    runner = adapter.make_in_process_runner(records=[DatasetRecord(question="q", ground_truth="g")])
    exp = _OE(
        name="e", tool="langchain", target_component="retrieval",
        description="d", parameters={},  # no pipeline_module
        objectives={"answer_correctness": 1.0},
        search_space={"top_k": [4]}, max_trials=1, stage="retrieval",
    )
    trial = _OT(
        trial_id="t1", experiment_name="e",
        tool="langchain", strategy="bayesian",
        parameters={"top_k": 4},
    )
    with pytest.raises(ValueError, match="pipeline_module"):
        runner(experiment=exp, trial=trial, baseline=_ER())


def test_simulate_mode_sets_session_notes_banner(tmp_path: Path, monkeypatch):
    """Simulate mode is convenient for plan-smoke-testing but its scores
    are meaningless. The executor must mark the session with a visible
    notes banner so users / dashboards know not to trust the numbers."""
    plan, baseline = _trivial_plan()
    # pipeline_module isn't needed for simulate (no runner is invoked).
    executor = OptimizationExecutor(root=tmp_path, strict_execute=False)
    session = executor.execute_plan(plan, baseline=baseline, mode="simulate")
    assert session.mode == "simulate"
    assert "SIMULATED" in session.notes
    assert "hash-based" in session.notes or "stub" in session.notes


def _capture_ragdx_logs(level: int = 0) -> tuple[logging.Handler, list[logging.LogRecord]]:
    """Attach a list-capturing handler directly to the ragdx root logger.

    The ragdx logger uses ``propagate=False`` so pytest's ``caplog``
    fixture can't see records via the root logger. This helper hooks
    into the namespace directly and returns the (handler, records)
    pair — caller is responsible for ``removeHandler`` cleanup.
    """
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    h = _ListHandler(level=level)
    logging.getLogger("ragdx").addHandler(h)
    return h, records


def test_simulate_mode_logs_warning(tmp_path: Path):
    """The simulate banner must hit the logger at WARNING level so it
    surfaces in standard log destinations (CI, file sinks, …)."""
    plan, baseline = _trivial_plan()
    executor = OptimizationExecutor(root=tmp_path, strict_execute=False)
    handler, records = _capture_ragdx_logs(level=logging.WARNING)
    try:
        executor.execute_plan(plan, baseline=baseline, mode="simulate")
    finally:
        logging.getLogger("ragdx").removeHandler(handler)
    matched = [r for r in records if r.levelno >= logging.WARNING and "SIMULATED" in r.getMessage()]
    assert matched, "simulate mode must emit a WARNING-level SIMULATED banner"


def test_execute_mode_does_not_emit_simulate_banner(tmp_path: Path, monkeypatch):
    """The simulate banner is gated on mode='simulate' only. Real
    execute runs (even failed ones) must NOT carry the SIMULATED note."""
    monkeypatch.delenv("RAGDX_LANGCHAIN_RUNNER_CMD", raising=False)
    monkeypatch.delenv("RAGDX_FALLBACK_SIMULATE_ON_MISSING_RUNNER", raising=False)
    plan, baseline = _trivial_plan()
    plan.experiments[0].parameters["pipeline_module"] = "x:y"  # any value
    executor = OptimizationExecutor(root=tmp_path, strict_execute=True)
    handler, records = _capture_ragdx_logs(level=logging.WARNING)
    try:
        session = executor.execute_plan(plan, baseline=baseline, mode="execute")
    finally:
        logging.getLogger("ragdx").removeHandler(handler)
    assert "SIMULATED" not in session.notes
    matched = [r for r in records if "SIMULATED" in r.getMessage()]
    assert not matched


def test_build_runner_spec_does_not_silently_default_dataset(tmp_path: Path):
    """``build_runner_spec`` (used to render the external runner config)
    must NOT inject examples/demo_dataset.jsonl when dataset_path is
    missing — the downstream runner should see None and fail loudly."""
    from ragdx.optim.langchain_adapter import LangChainAdapter
    from ragdx.schemas.models import OptimizationExperiment as _OE

    exp = _OE(
        name="e", tool="langchain", target_component="retrieval",
        description="d", parameters={},  # no dataset_path, no pipeline_module
        objectives={"answer_correctness": 1.0},
        search_space={"top_k": [4]}, max_trials=1, stage="retrieval",
    )
    spec = LangChainAdapter().build_runner_spec(exp, {"top_k": 4})
    assert spec["program_contract"]["dataset_path"] is None
    assert spec["program_contract"]["pipeline_module"] is None


# --------------------------------------------------------------------- #
# Ragas adapter normalization-only path stays cheap and pure
# --------------------------------------------------------------------- #
def test_ragas_adapter_normalize_scores_no_install_needed():
    from ragdx.engines.ragas_adapter import RagasAdapter
    r = RagasAdapter().normalize_scores({"context_precision": 0.7, "faithfulness": 0.8})
    assert r.retrieval["context_precision"] == 0.7
    assert r.generation["faithfulness"] == 0.8
    assert r.raw_tool_outputs["ragas"] == {"context_precision": 0.7, "faithfulness": 0.8}
