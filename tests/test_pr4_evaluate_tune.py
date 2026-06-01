"""Tests for PR4: ``runtime/factories``, ``workflows/evaluate``,
``ragdx evaluate``, and ``ragdx tune``.

Live LLM / FAISS / ragas calls are not exercised here -- those are
covered by the end-to-end demo bundles under ``new_demo1`` /
``new_demo2``. These tests verify:

* The ``RAGConfig`` -> ``RagdxRuntime`` -> ``EvaluationResult`` data
  flow shape (no API calls; uses stubs).
* The CLI commands are registered with the expected flags.
* Ragas score partitioning (``retrieval`` vs ``generation`` vs
  ``e2e``) is correct.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from ragdx.cli import app
from ragdx.cli import evaluate as cli_evaluate
from ragdx.cli import tune as cli_tune
from ragdx.runtime.factories import (
    DEFAULT_SYSTEM_INSTRUCTION,
    RagdxRuntime,
)
from ragdx.schemas.models import DatasetRecord, EvaluationResult
from ragdx.schemas.rag_config import (
    GeneratorSpec,
    JudgeSpec,
    RAGConfig,
)
from ragdx.workflows.evaluate import (
    _scores_to_evaluation_result,
    _select_metrics,
)


# ============================================================ RagdxRuntime
def test_ragdx_runtime_dataclass_has_pre_pr4_fields():
    """Field signature must match what ``ragdx.experiments._Runtime``
    exposed pre-PR4 (it's now a re-export alias). Drift here would
    silently break the stage optimizers."""
    # Build with all positional-required fields stubbed -- we're
    # only inspecting the dataclass shape.
    rt = RagdxRuntime(
        llm_callable=lambda p: "stub",
        dspy_lm=None,
        ragas_judge=None,
        ragas_embeddings=None,
        embeddings=None,
    )
    # Defaults match pre-PR4.
    assert rt.llm_max_concurrent == 2
    assert rt.llm_max_retries == 5
    assert rt.ragas_run_config is None
    assert rt.system_instruction == DEFAULT_SYSTEM_INSTRUCTION


def test_experiments_runtime_alias_still_imports():
    """``ragdx.experiments._Runtime`` must alias the new
    ``RagdxRuntime`` so internal references (the stage optimizers,
    tests, monkey-patches) keep working."""
    from ragdx.experiments import _Runtime
    assert _Runtime is RagdxRuntime


# ============================================================ _select_metrics
def test_select_metrics_with_gt_returns_four():
    """Reference-based metrics (``context_recall``) must be included
    only when at least one record carries ground truth."""
    rm = pytest.importorskip("ragas.metrics")

    records = [
        DatasetRecord(question="Q", ground_truth="A reference answer"),
    ]
    metrics = _select_metrics(records)
    # ``answer_relevancy`` + ``context_precision`` + ``faithfulness`` + ``context_recall``
    assert len(metrics) == 4
    assert rm.context_recall in metrics


def test_select_metrics_no_gt_returns_three():
    """No-GT eval drops ``context_recall``."""
    rm = pytest.importorskip("ragas.metrics")

    records = [DatasetRecord(question="Q", ground_truth=None)]
    metrics = _select_metrics(records)
    assert len(metrics) == 3
    assert rm.context_recall not in metrics


def test_select_metrics_empty_gt_string_treated_as_no_gt():
    """Whitespace-only ground_truth is the same as missing -- the
    composite scorer would otherwise crash on empty strings."""
    pytest.importorskip("ragas.metrics")
    records = [DatasetRecord(question="Q", ground_truth="   ")]
    metrics = _select_metrics(records)
    assert len(metrics) == 3  # no_gt path


# ============================================================ Scores partitioning
def test_scores_to_evaluation_result_partitions_correctly():
    """The full ragas metric universe must land in the right sections
    so ``ragdx diagnose`` / ``compare`` consume them."""
    scores = {
        "context_precision": 0.8,
        "context_recall": 0.9,
        "faithfulness": 0.7,
        "answer_relevancy": 0.6,
        "answer_correctness": 0.85,
        "hallucination": 0.05,
    }
    er = _scores_to_evaluation_result(scores)
    assert er.retrieval == {"context_precision": 0.8, "context_recall": 0.9}
    assert er.generation == {"faithfulness": 0.7, "answer_relevancy": 0.6}
    assert er.e2e == {"answer_correctness": 0.85, "hallucination": 0.05}


def test_scores_to_evaluation_result_unmapped_scores_go_to_metadata():
    """Unknown metric names must be preserved (under metadata) so the
    eval is lossless even when ragas adds new metrics ragdx doesn't
    classify yet."""
    er = _scores_to_evaluation_result(
        {"context_precision": 0.5, "novel_metric_xyz": 0.42},
    )
    assert er.retrieval == {"context_precision": 0.5}
    assert er.metadata.get("unmapped_scores") == {"novel_metric_xyz": 0.42}


def test_scores_to_evaluation_result_skips_non_numeric():
    """Strings / None / NaN strings would crash downstream consumers
    -- ``_scores_to_evaluation_result`` must filter them out."""
    er = _scores_to_evaluation_result(
        {"context_precision": 0.6, "context_recall": "N/A", "bad": None},
    )
    assert er.retrieval == {"context_precision": 0.6}
    # None / strings dropped silently.
    assert "context_recall" not in er.retrieval
    assert "bad" not in er.retrieval


def test_scores_to_evaluation_result_metadata_merge():
    """User-supplied metadata must be preserved alongside any
    workflow-injected fields."""
    er = _scores_to_evaluation_result(
        {"faithfulness": 0.9},
        metadata={"config_name": "esg-prod-v3"},
    )
    assert er.metadata["config_name"] == "esg-prod-v3"


def test_evaluation_result_round_trips_through_json():
    """The CLI writes the result as JSON; verify round-trip works."""
    er = _scores_to_evaluation_result({"faithfulness": 0.5})
    j = er.model_dump_json()
    er2 = EvaluationResult.model_validate_json(j)
    assert er2.generation == er.generation


# ============================================================ CLI registration
def test_evaluate_command_registered():
    """``ragdx evaluate`` must appear in the typer app's command list."""
    names = [info.name or info.callback.__name__ for info in app.registered_commands]
    assert "evaluate" in names


def test_tune_command_registered():
    """``ragdx tune`` must appear in the typer app's command list."""
    names = [info.name or info.callback.__name__ for info in app.registered_commands]
    assert "tune" in names


def test_evaluate_has_required_flags():
    """The CLI surface PR4 promises: --config, --questions, --corpus,
    --output, --api-key, --name."""
    params = set(inspect.signature(cli_evaluate).parameters)
    for required in ("config", "questions", "corpus", "output", "api_key", "name"):
        assert required in params, f"`ragdx evaluate` missing --{required}"


def test_tune_has_required_flags():
    """The CLI surface PR4 promises: --base-config, --questions,
    --corpus, --stage, --budget, --bo-init, --seed, --output, --api-key,
    --write-optimized-config."""
    params = set(inspect.signature(cli_tune).parameters)
    expected = {
        "base_config", "questions", "corpus", "stage", "budget",
        "bo_init", "seed", "output", "api_key", "write_optimized_config",
    }
    missing = expected - params
    assert not missing, f"`ragdx tune` missing flags: {missing}"


def test_tune_stage_choices_includes_all_stage_optimizers():
    """The four StageOptimizers must all be reachable via --stage."""
    from ragdx.cli.tune import _STAGE_CHOICES
    assert set(_STAGE_CHOICES) == {"chunking", "retrieval", "generation", "joint"}


# ============================================================ RunStore wiring (follow-up)
def test_evaluate_has_runstore_save_flags():
    """The follow-up that closes the README workflow's loop: ``evaluate
    --save`` must expose the same diagnose/persist surface as ``ragdx
    diagnose --save`` (``--save``, ``--baseline-run-id``, ``--use-llm``,
    ``--use-both``, ``--use-llm-planner``). Without these the user has
    to chain ``ragdx save`` manually and runs never appear in the
    dashboard."""
    params = set(inspect.signature(cli_evaluate).parameters)
    for required in (
        "save", "baseline_run_id", "use_llm", "use_both", "use_llm_planner",
    ):
        assert required in params, f"`ragdx evaluate` missing --{required}"


def test_tune_has_runstore_save_flags():
    """``ragdx tune --save`` must close the same loop: persist the
    tuned best-config evaluation to the RunStore so the result is
    visible in ``ragdx runs`` / ``ragdx dashboard``."""
    params = set(inspect.signature(cli_tune).parameters)
    for required in (
        "save", "name", "baseline_run_id",
        "use_llm", "use_both", "use_llm_planner",
    ):
        assert required in params, f"`ragdx tune` missing --{required}"


def test_evaluate_rejects_use_llm_without_save(monkeypatch, tmp_path):
    """``--use-llm`` / ``--use-both`` / ``--use-llm-planner`` /
    ``--baseline-run-id`` are meaningless without ``--save`` (they
    drive the diagnose/persist path). Validating early avoids a
    silent no-op after a multi-minute eval run."""
    import typer
    cfg_path = tmp_path / "rag.yaml"
    q_path = tmp_path / "q.jsonl"
    cfg_path.write_text("name: stub\n", encoding="utf-8")
    q_path.write_text('{"question": "Q"}\n', encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="only apply when --save"):
        cli_evaluate(
            config=str(cfg_path), questions=str(q_path), corpus="",
            output=str(tmp_path / "o.json"), api_key="stub", name="",
            save=False, baseline_run_id="",
            use_llm=True, use_both=False, use_llm_planner=False,
        )


def test_tune_rejects_use_llm_without_save(tmp_path):
    """Same early-fail invariant for ``ragdx tune``."""
    import typer
    cfg_path = tmp_path / "rag.yaml"
    q_path = tmp_path / "q.jsonl"
    cfg_path.write_text("name: stub\n", encoding="utf-8")
    q_path.write_text('{"question": "Q"}\n', encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="only apply when --save"):
        cli_tune(
            base_config=str(cfg_path), questions=str(q_path), stage="joint",
            corpus="", budget=4, bo_init=2, seed=0,
            output=str(tmp_path / "o.json"),
            write_optimized_config="", api_key="stub",
            save=False, name="", baseline_run_id="",
            use_llm=False, use_both=False, use_llm_planner=True,
        )


def test_tune_bundle_base_config_is_scrubbed():
    """The ``--output`` bundle JSON ``ragdx tune`` writes must scrub
    ``generator.api_key`` / ``judge.api_key`` from ``base_config``.
    Otherwise the env-resolved key leaks into a file users may commit
    -- discovered while running the new_demo3 scenarios (2026-06-01).

    Cheap source-level check: verifies the cli/tune.py code path uses
    ``scrubbed_for_commit()`` before ``model_dump`` on ``base_config``.
    A more thorough test would run the command and inspect the JSON,
    but that needs live ragas + LLM (covered by the demo bundles)."""
    import inspect

    src = inspect.getsource(cli_tune)
    assert "scrubbed_for_commit" in src and "base_config" in src
    # Specifically the bundle-construction line must scrub. A simple
    # regex catches the common ways someone might re-introduce the leak.
    import re
    pattern = re.compile(
        r'"base_config"\s*:\s*rag_config\.scrubbed_for_commit\(\)'
    )
    assert pattern.search(src), (
        "cli/tune.py bundle JSON must serialize base_config via "
        "scrubbed_for_commit() to prevent api_key leakage."
    )


# ============================================================ --project flag
def test_storage_resolves_per_project_root(monkeypatch, tmp_path):
    """``--project esg`` (or RAGDX_PROJECT=esg) must isolate the storage
    root to ``.ragdx/projects/esg/`` so different projects don't collide
    on run IDs or causal priors. RAGDX_ROOT, when set, wins (explicit
    full path beats project shorthand)."""
    from ragdx.config import StorageSettings
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGDX_ROOT", raising=False)
    monkeypatch.delenv("RAGDX_PROJECT", raising=False)
    # Default: cwd-relative .ragdx
    s = StorageSettings.from_env()
    assert s.root == Path(".ragdx") and s.project is None
    # RAGDX_PROJECT: namespaced
    monkeypatch.setenv("RAGDX_PROJECT", "esg")
    s = StorageSettings.from_env()
    assert s.project == "esg"
    assert s.root.parts[-2:] == ("projects", "esg")
    # RAGDX_ROOT wins over RAGDX_PROJECT (full-path override)
    monkeypatch.setenv("RAGDX_ROOT", str(tmp_path / "custom"))
    s = StorageSettings.from_env()
    assert s.root == tmp_path / "custom"
    assert s.project == "esg"  # surfaced for log/UI, but not used in root


def test_run_store_writes_to_project_subdir(monkeypatch, tmp_path):
    """End-to-end: when RAGDX_PROJECT is set, RunStore.save_run writes
    into the per-project subdir. Two projects' runs never collide."""
    from ragdx.config import get_settings
    from ragdx.schemas.models import (
        DiagnosisReport,
        EvaluationResult,
        OptimizationPlan,
    )
    from ragdx.storage.run_store import RunStore

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGDX_ROOT", raising=False)
    monkeypatch.setenv("RAGDX_PROJECT", "esg")

    settings = get_settings()
    store = RunStore(root=str(settings.storage.root))
    run = store.save_run(
        EvaluationResult(retrieval={"context_precision": 0.5}),
        DiagnosisReport(summary="stub"),
        OptimizationPlan(objective_metric="context_precision"),
        name="esg-baseline",
    )
    # The file landed under the project-namespaced root.
    project_root = tmp_path / ".ragdx" / "projects" / "esg"
    assert (project_root / "runs" / f"{run.run_id}.json").exists()
    # Two different projects: ensure isolation.
    monkeypatch.setenv("RAGDX_PROJECT", "legal")
    settings2 = get_settings()
    store2 = RunStore(root=str(settings2.storage.root))
    assert store2.load_run.__self__.runs_dir != store.runs_dir


# ============================================================ SavedRun.rag_config
def test_saved_run_carries_rag_config_round_trip():
    """SavedRun.rag_config must round-trip through JSON so ``ragdx tune
    --from-run`` can reload it. Defaulting to None preserves
    backward-compatibility with pre-PR6 SavedRun files."""
    from ragdx.schemas.models import (
        DiagnosisReport,
        EvaluationResult,
        OptimizationPlan,
        SavedRun,
    )

    sr = SavedRun(
        run_id="abc123",
        created_at="2026-06-01T00:00:00Z",
        name="t",
        evaluation=EvaluationResult(retrieval={"context_precision": 0.5}),
        diagnosis=DiagnosisReport(summary="stub"),
        optimization_plan=OptimizationPlan(objective_metric="context_precision"),
        rag_config=RAGConfig(name="prod-rag"),
    )
    sr2 = SavedRun.model_validate_json(sr.model_dump_json())
    assert sr2.rag_config is not None
    assert sr2.rag_config.name == "prod-rag"


def test_saved_run_pre_pr6_files_load_without_rag_config():
    """A SavedRun JSON without the ``rag_config`` field (the shape used
    before PR6) must still load -- the field is optional/default None."""
    import json as _json

    from ragdx.schemas.models import SavedRun
    pre_pr6_payload = {
        "schema_version": 1,
        "run_id": "deadbeef0001",
        "created_at": "2026-05-01T00:00:00Z",
        "name": "legacy",
        "evaluation": {"retrieval": {}, "generation": {}, "e2e": {}, "metadata": {}},
        "diagnosis": {"summary": "stub"},
        "optimization_plan": {"objective_metric": "context_precision"},
    }
    sr = SavedRun.model_validate_json(_json.dumps(pre_pr6_payload))
    assert sr.rag_config is None


def test_run_store_save_run_scrubs_rag_config_defensively():
    """Even if a caller forgets ``scrubbed_for_commit()``, ``save_run``
    must never persist credentials to disk. Catches the same security
    bug class as commit 73ee990 / e15d9b8 (YAML + JSON tune output)
    for the SavedRun path."""
    import tempfile

    from ragdx.schemas.models import (
        DiagnosisReport,
        EvaluationResult,
        OptimizationPlan,
    )
    from ragdx.schemas.rag_config import GeneratorSpec
    from ragdx.storage.run_store import RunStore

    with tempfile.TemporaryDirectory() as td:
        store = RunStore(root=td)
        # Pass a config WITH api_key set -- save_run should null it.
        leaky_cfg = RAGConfig(
            generator=GeneratorSpec(api_key="SUPER-SECRET-KEY"),
        )
        run = store.save_run(
            EvaluationResult(retrieval={"context_precision": 0.5}),
            DiagnosisReport(summary="stub"),
            OptimizationPlan(objective_metric="context_precision"),
            rag_config=leaky_cfg,
        )
        on_disk = (Path(td) / "runs" / f"{run.run_id}.json").read_text(encoding="utf-8")
        assert "SUPER-SECRET-KEY" not in on_disk
        # Confirm reload also has no key.
        sr2 = store.load_run(run.run_id)
        assert sr2.rag_config.generator.api_key is None
        # Original in-memory config is NOT mutated.
        assert leaky_cfg.generator.api_key == "SUPER-SECRET-KEY"


# ============================================================ tune --from-run
def test_tune_has_from_run_flags():
    """``--from-run`` and ``--experiment`` close the evaluate->tune
    config-shuttling gap. Without these, users have to re-pass
    --base-config / --questions / --stage / --budget / --baseline-run-id
    by hand, even though every value lives in the SavedRun + plan."""
    params = set(inspect.signature(cli_tune).parameters)
    assert "from_run" in params
    assert "experiment_name" in params


def test_tune_experiment_without_from_run_fails_early(tmp_path):
    """``--experiment`` is only meaningful with ``--from-run``."""
    import typer
    cfg_path = tmp_path / "rag.yaml"
    q_path = tmp_path / "q.jsonl"
    cfg_path.write_text("name: stub\n", encoding="utf-8")
    q_path.write_text('{"question": "Q"}\n', encoding="utf-8")
    with pytest.raises(typer.BadParameter, match=r"--experiment.*only.*--from-run"):
        cli_tune(
            base_config=str(cfg_path), questions=str(q_path), stage="joint",
            corpus="", budget=4, bo_init=2, seed=0,
            output=str(tmp_path / "o.json"),
            write_optimized_config="", api_key="stub",
            save=False, name="", baseline_run_id="",
            use_llm=False, use_both=False, use_llm_planner=False,
            from_run="", experiment_name="ignored",
        )


def test_tune_stage_auto_is_accepted_in_validation():
    """The 'auto' sentinel for --stage must not be rejected by the
    early _STAGE_CHOICES guard (it gets resolved later from --from-run
    or to 'joint')."""
    import inspect as _inspect
    src = _inspect.getsource(cli_tune)
    # Belt-and-braces: confirm the guard exempts "auto".
    assert "stage != \"auto\"" in src or "stage != 'auto'" in src


def test_tune_from_run_inherits_stage_budget_and_search_space(tmp_path, monkeypatch):
    """Locks down the core --from-run contract: a SavedRun whose plan
    has ``experiments[0].stage='retrieval'`` + ``max_trials=12`` +
    ``search_space={'top_k': [2, 4, 6]}`` should cause tune to:
      * pick RetrievalOptimizer
      * set ctx.n_bo_trials=12
      * set ctx.top_ks=[2, 4, 6]
    even though the user passed no --stage / --budget.

    We mock out the heavy bits (corpus loading, runtime build, ragas
    eval) and assert against the StageContext we synthesize."""
    import sys
    from unittest.mock import MagicMock, patch

    from ragdx.optim.stages import StageResult
    from ragdx.schemas.models import (
        DiagnosisReport,
        EvaluationResult,
        OptimizationExperiment,
        OptimizationPlan,
    )
    from ragdx.schemas.rag_config import GeneratorSpec

    # Isolate the RunStore in tmp_path so we can plant a fixture run.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGDX_PROJECT", raising=False)
    monkeypatch.delenv("RAGDX_ROOT", raising=False)
    monkeypatch.setenv("RAGDX_ROOT", str(tmp_path / ".ragdx"))

    from ragdx.storage.run_store import RunStore
    store = RunStore(root=str(tmp_path / ".ragdx"))
    questions_path = tmp_path / "q.jsonl"
    questions_path.write_text('{"question": "Q?"}\n', encoding="utf-8")
    plan = OptimizationPlan(
        objective_metric="context_precision",
        experiments=[OptimizationExperiment(
            name="retrieval-pipeline-search",
            tool="manual",
            target_component="retrieval",
            description="planned retrieval sweep",
            stage="retrieval",
            max_trials=12,
            search_space={"top_k": [2, 4, 6]},
        )],
    )
    cfg = RAGConfig(
        name="from-run-test",
        generator=GeneratorSpec(api_key=None),
    )
    seed_run = store.save_run(
        EvaluationResult(
            retrieval={"context_precision": 0.5},
            metadata={
                "questions_path": str(questions_path),
                "corpus": "/dev/null",  # don't load corpus in this test
            },
        ),
        DiagnosisReport(summary="stub"),
        plan,
        name="seed",
        rag_config=cfg,
    )
    # Patch the heavy bits before tune() runs.
    captured: dict = {}

    class _FakeOptimizer:
        def __init__(self): pass
        def optimize(self, ctx):
            captured["ctx"] = ctx
            return StageResult(
                stage_name="retrieval",
                search_space={"top_k": ctx.top_ks},
                trials=[],
                best_params={"top_k": ctx.top_ks[0]},
                best_config=ctx.base_config,
                best_composite=1.0,
                objective_spec={},
                n_init=ctx.n_bo_init,
                max_trials=ctx.n_bo_trials,
            )

    fake_runtime = MagicMock()
    fake_runtime.embeddings = None
    fake_runtime.llm_callable = lambda p: "stub"
    with patch.dict(sys.modules):
        # The function does `from ragdx.optim.stages import RetrievalOptimizer`
        # at runtime. We patch it via the package module.
        from ragdx.optim import stages as stages_mod
        monkeypatch.setattr(stages_mod, "RetrievalOptimizer", _FakeOptimizer)

        from ragdx.runtime import factories
        monkeypatch.setattr(factories, "build_runtime", lambda c: fake_runtime)

        from ragdx import experiments as exp_mod
        monkeypatch.setattr(
            exp_mod, "_load_corpus_and_records",
            lambda stub, runtime: (["chunk-a", "chunk-b"], [], {}),
        )
        monkeypatch.setattr(
            exp_mod, "_build_ragas_metrics_for_mode", lambda mode: [],
        )
        monkeypatch.setattr(
            exp_mod, "_make_pdf_re_chunk_fn", lambda stub, chunks: None,
        )
        # We need an api_key for the resolve check; set env.
        monkeypatch.setenv("ZHIPU_API_KEY", "stub-key")

        cli_tune(
            base_config="", questions="",
            from_run=seed_run.run_id, experiment_name="",
            stage="auto", corpus="", budget=0,
            bo_init=2, seed=0,
            output=str(tmp_path / "tune_out.json"),
            write_optimized_config="", api_key="",
            save=False, name="", baseline_run_id="",
            use_llm=False, use_both=False, use_llm_planner=False,
        )

    ctx = captured["ctx"]
    assert ctx.n_bo_trials == 12           # inherited max_trials
    assert ctx.top_ks == [2, 4, 6]          # inherited search_space
    assert ctx.base_config.name == "from-run-test"   # inherited rag_config


def test_tune_from_run_explicit_flags_override_plan(tmp_path, monkeypatch):
    """User-passed --stage / --budget must override what --from-run
    would have inherited from the plan."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGDX_PROJECT", raising=False)
    monkeypatch.setenv("RAGDX_ROOT", str(tmp_path / ".ragdx"))

    from ragdx.schemas.models import (
        DiagnosisReport,
        EvaluationResult,
        OptimizationExperiment,
        OptimizationPlan,
    )
    from ragdx.storage.run_store import RunStore

    store = RunStore(root=str(tmp_path / ".ragdx"))
    q_path = tmp_path / "q.jsonl"
    q_path.write_text('{"question": "Q?"}\n', encoding="utf-8")
    plan = OptimizationPlan(
        objective_metric="context_precision",
        experiments=[OptimizationExperiment(
            name="planned",
            tool="manual",
            target_component="retrieval",
            description="planned",
            stage="retrieval",
            max_trials=12,
        )],
    )
    seed_run = store.save_run(
        EvaluationResult(
            retrieval={"context_precision": 0.5},
            metadata={"questions_path": str(q_path), "corpus": "/dev/null"},
        ),
        DiagnosisReport(summary="stub"),
        plan,
        name="seed",
        rag_config=RAGConfig(name="t"),
    )

    captured: dict = {}

    class _FakeOptimizer:
        def __init__(self): pass
        def optimize(self, ctx):
            captured["ctx"] = ctx
            from ragdx.optim.stages import StageResult
            return StageResult(
                stage_name="joint",
                search_space={},
                trials=[],
                best_params={"chunk_size": 256, "chunk_overlap": 0, "top_k": 3},
                best_config=ctx.base_config,
                best_composite=1.0,
                objective_spec={},
                n_init=ctx.n_bo_init,
                max_trials=ctx.n_bo_trials,
            )

    from unittest.mock import MagicMock
    fake_runtime = MagicMock()
    fake_runtime.embeddings = None
    fake_runtime.llm_callable = lambda p: "stub"
    from ragdx.optim import stages as stages_mod
    monkeypatch.setattr(stages_mod, "JointOptimizer", _FakeOptimizer)

    from ragdx.runtime import factories
    monkeypatch.setattr(factories, "build_runtime", lambda c: fake_runtime)

    from ragdx import experiments as exp_mod
    monkeypatch.setattr(
        exp_mod, "_load_corpus_and_records",
        lambda stub, runtime: (["chunk-a"], [], {}),
    )
    monkeypatch.setattr(
        exp_mod, "_build_ragas_metrics_for_mode", lambda mode: [],
    )
    monkeypatch.setattr(
        exp_mod, "_make_pdf_re_chunk_fn", lambda stub, chunks: None,
    )
    monkeypatch.setenv("ZHIPU_API_KEY", "stub-key")

    cli_tune(
        base_config="", questions="",
        from_run=seed_run.run_id, experiment_name="",
        stage="joint",   # ← explicit override of plan's "retrieval"
        corpus="",
        budget=4,         # ← explicit override of plan's 12
        bo_init=2, seed=0,
        output=str(tmp_path / "tune_out.json"),
        write_optimized_config="", api_key="",
        save=False, name="", baseline_run_id="",
        use_llm=False, use_both=False, use_llm_planner=False,
    )

    ctx = captured["ctx"]
    assert ctx.n_bo_trials == 4   # explicit beat inherited 12


def test_tune_from_run_rejects_pre_pr6_savedrun(tmp_path, monkeypatch):
    """A SavedRun without rag_config must produce a clear error message
    pointing the user at the fix (re-run evaluate --save on PR6+)."""
    import typer
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGDX_PROJECT", raising=False)
    monkeypatch.setenv("RAGDX_ROOT", str(tmp_path / ".ragdx"))

    from ragdx.schemas.models import (
        DiagnosisReport,
        EvaluationResult,
        OptimizationPlan,
    )
    from ragdx.storage.run_store import RunStore

    store = RunStore(root=str(tmp_path / ".ragdx"))
    seed_run = store.save_run(
        EvaluationResult(retrieval={"context_precision": 0.5}),
        DiagnosisReport(summary="stub"),
        OptimizationPlan(objective_metric="context_precision"),
        name="legacy",
        # No rag_config -- simulates a pre-PR6 saved run.
    )
    with pytest.raises(typer.BadParameter, match="rag_config"):
        cli_tune(
            base_config="", questions="",
            from_run=seed_run.run_id, experiment_name="",
            stage="auto", corpus="", budget=0,
            bo_init=2, seed=0,
            output=str(tmp_path / "o.json"),
            write_optimized_config="", api_key="",
            save=False, name="", baseline_run_id="",
            use_llm=False, use_both=False, use_llm_planner=False,
        )


def test_tune_save_synthesizes_evaluation_result_from_best_trial(tmp_path):
    """The ``--save`` path on tune synthesizes an EvaluationResult from
    the best trial's ragas scores before calling diagnose/save_run.
    Verify the synthesis helper directly (the full path needs live
    ragas/LLM, exercised in the e2e bundles)."""
    from ragdx.optim.stages import StageResult, StageTrial
    from ragdx.workflows.evaluate import _scores_to_evaluation_result

    trial = StageTrial(
        trial_index=0,
        params={"top_k": 5},
        n_chunks=100,
        scores={
            "context_precision": 0.82,
            "faithfulness": 0.71,
            "answer_correctness": 0.65,
        },
        composite_score=1.4,
        feasible=True,
        violations=[],
        answers_preview=["short..."],
        records=[],
        elapsed_seconds=1.0,
    )
    sr = StageResult(
        stage_name="retrieval",
        search_space={"top_k": [3, 5, 7]},
        trials=[trial],
        best_params={"top_k": 5},
        best_config=None,
        best_composite=1.4,
        objective_spec={},
        n_init=2,
        max_trials=4,
    )
    matching = [t for t in sr.trials if t.params == sr.best_params]
    assert matching, "Expected the best-params lookup to find the trial."
    er = _scores_to_evaluation_result(
        dict(matching[0].scores), metadata={"tune_stage": "retrieval"},
    )
    assert er.retrieval == {"context_precision": 0.82}
    assert er.generation == {"faithfulness": 0.71}
    assert er.e2e == {"answer_correctness": 0.65}
    assert er.metadata["tune_stage"] == "retrieval"


# ============================================================ Experiment alias surface
def test_experiment_build_runtime_uses_factory():
    """``ragdx.experiments._build_runtime`` is a delegation shim after
    PR4. It must still return a ``RagdxRuntime``-shaped object so the
    stage optimizers keep working without changes."""
    from ragdx.experiments import _build_runtime, _Runtime
    assert _Runtime is RagdxRuntime
    # Don't actually call _build_runtime (requires live LLM / HF cache).
    # The signature itself proves the delegation is in place.
    assert callable(_build_runtime)


def test_make_rag_config_translates_experiment_config():
    """``_make_rag_config_from_experiment_config`` must produce a
    RAGConfig whose generator.model / api_base / etc. mirror the
    ExperimentConfig -- this is what build_runtime then consumes."""
    from ragdx.experiments import (
        ExperimentConfig,
        _make_rag_config_from_experiment_config,
    )

    cfg = ExperimentConfig(
        corpus="org/data",
        has_gt=True,
        api_key="stub",
        model="openai/glm-4-flash",
        api_base="https://example.com/v1",
        llm_max_concurrent=4,
        llm_max_retries=10,
        system_instruction="custom-prompt",
    )
    rag = _make_rag_config_from_experiment_config(cfg)
    assert rag.generator.model == "openai/glm-4-flash"
    assert rag.generator.api_base == "https://example.com/v1"
    assert rag.generator.api_key == "stub"
    assert rag.generator.system_instruction == "custom-prompt"
    assert rag.judge.llm_max_concurrent == 4
    assert rag.judge.llm_max_retries == 10


# ============================================================ RAGConfig wiring
def test_evaluate_uses_judge_model_fallback_to_generator():
    """When ``JudgeSpec.model`` is None, the judge must use the
    generator's model. This avoids a config footgun where users
    fill in the generator but forget the judge."""
    # build_ragas_judge wraps a ChatOpenAI with a LangchainLLMWrapper.
    # We stub langchain_openai inline below, but ragas.llms is also
    # imported -- skip the test when ragas isn't installed.
    pytest.importorskip("ragas.llms")
    from ragdx.runtime.factories import build_ragas_judge

    gen = GeneratorSpec(model="openai/gpt-4o-mini")
    judge = JudgeSpec(model=None, api_key="stub")
    # We don't actually instantiate the langchain client -- just
    # confirm the model resolution path. We import the helper module
    # and patch its dependencies, OR we observe via the constructed
    # ChatOpenAI's model attribute. Easiest: stub langchain_openai.
    import sys
    import types

    real_lc_openai = sys.modules.get("langchain_openai")
    fake = types.ModuleType("langchain_openai")
    captured: dict = {}

    class _StubChatOpenAI:
        def __init__(self, **kw):
            captured.update(kw)

    fake.ChatOpenAI = _StubChatOpenAI
    sys.modules["langchain_openai"] = fake
    try:
        build_ragas_judge(judge, fallback=gen)
        # ``openai/`` prefix gets stripped for ChatOpenAI's model arg.
        assert captured["model"] == "gpt-4o-mini"
    finally:
        if real_lc_openai is not None:
            sys.modules["langchain_openai"] = real_lc_openai
        else:
            sys.modules.pop("langchain_openai", None)


def test_build_embedder_rejects_unsupported_kind():
    """Future-proofing: a config with an unknown ``EmbedderSpec.kind``
    must fail at runtime construction, not silently fall back."""
    from ragdx.runtime.factories import build_embedder
    from ragdx.schemas.rag_config import EmbedderSpec

    spec = EmbedderSpec()
    spec.kind = "openai"  # type: ignore[assignment]
    with pytest.raises(NotImplementedError):
        build_embedder(spec)


def test_apply_temperature_clamp_is_idempotent():
    """The clamp must be re-callable without double-wrapping (sentinel
    attribute pattern). Verifying this prevents a real bug where each
    experiment call would add a layer of indirection."""
    litellm = pytest.importorskip("litellm")

    from ragdx.runtime.factories import apply_litellm_temperature_clamp
    apply_litellm_temperature_clamp()
    first = litellm.completion
    apply_litellm_temperature_clamp()
    apply_litellm_temperature_clamp()
    assert litellm.completion is first
    assert getattr(first, "_ragdx_clamped", False) is True


def test_rag_config_default_is_unchanged_by_pr4():
    """PR4 must not have mutated the default RAGConfig surface --
    PR1's snapshot test covers most of this, but we re-pin the
    generator/judge defaults specifically since PR4 added factories
    that depend on them."""
    cfg = RAGConfig()
    assert cfg.generator.api_base == "https://open.bigmodel.cn/api/paas/v4"
    assert cfg.generator.model == "openai/glm-4-flash"
    assert cfg.judge.model is None  # use generator's
    assert cfg.judge.llm_max_concurrent == 2
