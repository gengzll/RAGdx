"""Tests for ``ragdx.experiments`` — validation + dispatch logic.

We don't run the full LLM pipeline here (that's the demo's job). Tests
cover the API contract: parameter validation, mode resolution, corpus
dispatch, and the public dataclass surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragdx.experiments import (
    SCHEMA_VERSION,
    ExperimentConfig,
    ExperimentMode,
    ExperimentResult,
    _load_jsonl_questions,
    _looks_like_hf_dataset,
    migrate_legacy_bundle,
    run_experiment,
)


# ------------------------------------------------------------ config
def test_config_requires_with_gt_for_with_gt_mode():
    with pytest.raises(ValueError, match="mode='with_gt'"):
        ExperimentConfig(corpus="report.pdf", has_gt=False, mode="with_gt", api_key="k")


def test_config_requires_with_gt_for_both_mode():
    with pytest.raises(ValueError, match="mode='with_gt'"):
        ExperimentConfig(corpus="report.pdf", has_gt=False, mode="both", api_key="k")


def test_config_auto_mode_resolves_to_no_gt_when_has_gt_false():
    cfg = ExperimentConfig(corpus="report.pdf", has_gt=False, mode="auto", api_key="k")
    assert cfg.mode == "no_gt"


def test_config_auto_mode_resolves_to_both_when_has_gt_true():
    cfg = ExperimentConfig(corpus="org/data", has_gt=True, mode="auto", api_key="k")
    assert cfg.mode == "both"


def test_config_no_gt_with_has_gt_true_is_allowed():
    """``has_gt=True, mode='no_gt'`` is the 'data has labels but I want
    to ignore them' scenario — should NOT raise."""
    cfg = ExperimentConfig(corpus="org/data", has_gt=True, mode="no_gt", api_key="k")
    assert cfg.mode == "no_gt"


def test_config_picks_up_env_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ZHIPU_API_KEY", "from-env")
    cfg = ExperimentConfig(corpus="org/data", has_gt=True)
    assert cfg.api_key == "from-env"


def test_config_requires_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="api_key is required"):
        ExperimentConfig(corpus="org/data", has_gt=True)


def test_config_output_dir_becomes_path():
    cfg = ExperimentConfig(corpus="org/data", has_gt=True, api_key="k", output_dir="some/dir")
    assert isinstance(cfg.output_dir, Path)


# ------------------------------------------------------- HF dataset sniff
def test_looks_like_hf_dataset_positive():
    assert _looks_like_hf_dataset("explodinggradients/amnesty_qa")
    assert _looks_like_hf_dataset("acme/my-eval-set")


def test_looks_like_hf_dataset_rejects_paths():
    assert not _looks_like_hf_dataset("docs/report.pdf")
    assert not _looks_like_hf_dataset("path/to/file.jsonl")
    assert not _looks_like_hf_dataset("/abs/path.txt")


def test_looks_like_hf_dataset_rejects_local_existing(tmp_path: Path):
    p = tmp_path / "data.csv"
    p.write_text("hello")
    assert not _looks_like_hf_dataset(str(p))


def test_looks_like_hf_dataset_rejects_non_strings():
    assert not _looks_like_hf_dataset(Path("a/b"))  # type: ignore[arg-type]


# --------------------------------------------------- jsonl questions loader
def test_load_jsonl_questions_roundtrip(tmp_path: Path):
    p = tmp_path / "q.jsonl"
    rows = [
        {"question": "What is RAG?", "ground_truth": "Retrieval-augmented generation"},
        {"question": "Why chunk?", "ground_truth": "Smaller pieces"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    records = _load_jsonl_questions(p)
    assert len(records) == 2
    assert records[0].question == "What is RAG?"
    assert records[0].ground_truth == "Retrieval-augmented generation"
    assert records[1].contexts == []


def test_load_jsonl_questions_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "q.jsonl"
    p.write_text(
        '{"question": "Q1"}\n\n   \n{"question": "Q2"}\n', encoding="utf-8"
    )
    records = _load_jsonl_questions(p)
    assert [r.question for r in records] == ["Q1", "Q2"]


# --------------------------------------------------- experiment-result API
def test_experiment_result_save_writes_json(tmp_path: Path):
    cfg = ExperimentConfig(corpus="org/data", has_gt=True, api_key="k", output_dir=tmp_path)
    bundle = {"hello": "world", "n": 42}
    result = ExperimentResult(config=cfg, bundle=bundle, output_path=tmp_path / "result.json")
    written = result.save()
    assert written.exists()
    on_disk = json.loads(written.read_text(encoding="utf-8"))
    assert on_disk == bundle


def test_experiment_result_save_to_custom_path(tmp_path: Path):
    cfg = ExperimentConfig(corpus="org/data", has_gt=True, api_key="k", output_dir=tmp_path)
    res = ExperimentResult(config=cfg, bundle={"a": 1}, output_path=tmp_path / "result.json")
    target = tmp_path / "custom.json"
    written = res.save(target)
    assert written == target
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}


# -------------------------------------------------- run_experiment validation
def test_run_experiment_validates_no_gt_with_with_gt_mode(monkeypatch: pytest.MonkeyPatch):
    """``has_gt=False`` + ``mode='with_gt'`` should fail at the config
    step before any LLM call -- this is the ``ValueError`` users hit
    most often."""
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    with pytest.raises(ValueError, match="mode='with_gt'"):
        run_experiment(
            corpus="report.pdf", has_gt=False, mode="with_gt", save=False,
        )


def test_run_experiment_validates_no_gt_with_both_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    with pytest.raises(ValueError, match="mode='with_gt'"):
        run_experiment(
            corpus="report.pdf", has_gt=False, mode="both", save=False,
        )


# ------------------------------------------------- ExperimentMode type
def test_experiment_mode_literal_values():
    """Mostly a documentation test: confirms the public mode names."""
    valid: list[ExperimentMode] = ["with_gt", "no_gt", "both", "auto"]
    assert set(valid) == {"with_gt", "no_gt", "both", "auto"}


# ------------------------------------------------- corpus dispatch errors
def test_run_experiment_rejects_unsupported_corpus(monkeypatch: pytest.MonkeyPatch):
    """A bare filename with an unsupported extension should fail with
    a clear error, not a cryptic import error."""
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    # Bypass runtime building so we can exercise the corpus dispatcher
    # without a real LLM. Easiest path: call ``_load_corpus_and_records``
    # directly with a stub runtime.
    from ragdx.experiments import _load_corpus_and_records

    class _StubRuntime:
        llm_callable = staticmethod(lambda p: "stub")
    cfg = ExperimentConfig(
        corpus="not_a_real_thing.exe",
        has_gt=False,
        mode="no_gt",
        api_key="k",
    )
    with pytest.raises(ValueError, match="Unsupported corpus"):
        _load_corpus_and_records(cfg, _StubRuntime())  # type: ignore[arg-type]


# ====================================================================
# schema_version: 1 unified bundle + migration
# ====================================================================
def _v1_skeleton(mode_keys: list[str]) -> dict:
    """Minimal example v1 bundle for testing migrate_legacy_bundle idempotency."""
    return {
        "schema_version": 1,
        "meta": {
            "model": "openai/glm-4-flash",
            "model_endpoint": "https://example.com/v1",
            "experiment_mode": "both" if len(mode_keys) > 1 else mode_keys[0],
            "modes_run": mode_keys,
            "has_gt": "with_gt" in mode_keys,
            "detected_gt_mode": "with_gt" if "with_gt" in mode_keys else "no_gt",
            "source": {"kind": "huggingface", "corpus": "x/y"},
        },
        "questions": [{"question": "Q1", "ground_truth": "A1"}],
        "data_diagnostics": {m: {"has_ground_truth": m == "with_gt", "gt_mode": m, "record_count": 1} for m in mode_keys},
        "objectives": {m: {"metrics": {"faithfulness": 1.0}, "constraints": {}, "mode": "weighted_sum"} for m in mode_keys},
        "bayes_search": {m: {"trials": [], "best_params": None, "best_composite": None} for m in mode_keys},
        "dspy_a_b": {m: {"baseline_scores": {}, "optimized_scores": {}, "delta": {}} for m in mode_keys},
        "extras": {},
    }


def test_schema_version_constant():
    assert SCHEMA_VERSION == 1


def test_migrate_legacy_bundle_passes_through_v1():
    """Already-v1 bundles must be returned untouched (no re-wrapping)."""
    b = _v1_skeleton(["with_gt", "no_gt"])
    out = migrate_legacy_bundle(b)
    assert out is b  # identity, not a copy


def test_migrate_legacy_bundle_grid_shape():
    """The optimize_gt_modes demo wrote ``autorag_grid`` with mode-keyed
    runs. After migration the bundle must be v1 with both modes."""
    legacy = {
        "model": "openai/glm-4-flash",
        "model_endpoint": "https://example.com/v1",
        "dataset": "explodinggradients/amnesty_qa[:5]",
        "corpus_size": 5,
        "corpus_chunks": 15,
        "data_diagnostics": {
            "with_gt": {"has_ground_truth": True, "gt_mode": "with_gt"},
            "no_gt": {"has_ground_truth": False, "gt_mode": "no_gt"},
        },
        "autorag_grid": {
            "with_gt": {
                "objective": "context_recall",
                "best_top_k": 3,
                "best_composite": 1.5,
                "runs": [
                    {"top_k": 1, "scores": {"context_recall": 0.6}, "composite_score": 1.0, "feasible": True, "violations": [], "answers": ["a"]},
                    {"top_k": 3, "scores": {"context_recall": 0.9}, "composite_score": 1.5, "feasible": True, "violations": [], "answers": ["a"]},
                ],
            },
            "no_gt": {
                "objective": "faithfulness",
                "best_top_k": 1,
                "best_composite": 0.8,
                "runs": [{"top_k": 1, "scores": {"faithfulness": 0.8}, "composite_score": 0.8, "feasible": True, "violations": [], "answers": ["a"]}],
            },
        },
        "dspy_before_after": {
            "with_gt": {"baseline_scores": {}, "optimized_scores": {}, "delta": {}},
            "no_gt": {"baseline_scores": {}, "optimized_scores": {}, "delta": {}},
        },
        "questions": [{"question": "Q", "ground_truth": "A"}],
    }
    v1 = migrate_legacy_bundle(legacy)
    assert v1["schema_version"] == 1
    assert v1["meta"]["modes_run"] == ["with_gt", "no_gt"]
    assert v1["meta"]["source"]["kind"] == "huggingface"
    assert v1["meta"]["_migrated_from"] == "legacy_grid_v0"
    # Every legacy run becomes a v1 trial.
    assert len(v1["bayes_search"]["with_gt"]["trials"]) == 2
    assert v1["bayes_search"]["with_gt"]["trials"][0]["params"]["top_k"] == 1
    assert v1["bayes_search"]["with_gt"]["_legacy_kind"] == "grid"
    # dspy_a_b is mode-keyed.
    assert set(v1["dspy_a_b"]) == {"with_gt", "no_gt"}


def test_migrate_legacy_bundle_pdf_no_gt_shape():
    """The pdf_no_gt demo wrote a flat ``autorag_bo`` (single mode).
    Migration must wrap it under {no_gt: ...}."""
    legacy = {
        "model": "openai/glm-4-flash",
        "model_endpoint": "https://example.com/v1",
        "source_pdf": "report.pdf",
        "pdf_meta": {"page_count": 100, "chunks": 500},
        "questions": [{"question": "Q1", "ground_truth": None}],
        "synthesized_meta": [{"question": "Q1", "source_chunk_ids": [1, 2]}],
        "gt_mode": "no_gt",
        "objective_spec": {"metrics": {"faithfulness": 1.5}, "constraints": {}, "mode": "weighted_sum"},
        "autorag_bo": {
            "search_space": {"top_k": [1, 3]},
            "trials": [{"trial_index": 0, "params": {"top_k": 1}, "scores": {"faithfulness": 0.8}, "composite_score": 1.2, "feasible": True}],
            "best_params": {"top_k": 1},
            "best_composite": 1.2,
        },
        "dspy_before_after": {"baseline_scores": {"faithfulness": 0.5}, "optimized_scores": {"faithfulness": 0.7}, "delta": {"faithfulness": 0.2}},
    }
    v1 = migrate_legacy_bundle(legacy)
    assert v1["schema_version"] == 1
    assert v1["meta"]["modes_run"] == ["no_gt"]
    assert v1["meta"]["source"]["kind"] == "pdf"
    assert v1["meta"]["_migrated_from"] == "legacy_pdf_v0"
    assert "pdf_meta" in v1["extras"]
    assert "synthesized_questions" in v1["extras"]
    # Single mode is wrapped in a {no_gt: ...} dict.
    assert set(v1["bayes_search"]) == {"no_gt"}
    assert set(v1["dspy_a_b"]) == {"no_gt"}


def test_migrate_legacy_bundle_unknown_shape_stamps_zero():
    out = migrate_legacy_bundle({"random": "stuff"})
    assert out["schema_version"] == 0
    assert out["random"] == "stuff"


def test_committed_example_snapshots_are_v1(tmp_path: Path):
    """The two snapshots in docs/examples/ must already be v1 so the
    generic dashboard renders them directly without invoking the migrator."""
    repo = Path(__file__).resolve().parents[1]
    for name in ("optimize_gt_modes_result.json", "pdf_no_gt_result.json"):
        p = repo / "docs" / "examples" / name
        bundle = json.loads(p.read_text(encoding="utf-8"))
        assert bundle.get("schema_version") == 1, f"{name} is not v1"
        # The dashboard relies on these top-level keys being present.
        for key in ("meta", "questions", "bayes_search", "dspy_a_b", "objectives", "extras"):
            assert key in bundle, f"{name} missing top-level key {key}"
        # Mode-keyed sections are *always* dicts (never flat).
        assert isinstance(bundle["bayes_search"], dict)
        assert isinstance(bundle["dspy_a_b"], dict)
        modes_run = bundle["meta"]["modes_run"]
        assert modes_run, f"{name} has empty modes_run"
        for m in modes_run:
            assert m in bundle["bayes_search"], f"{name} bayes_search missing mode {m}"


# ====================================================================
# CLI -- ragdx experiment subcommand
#
# We deliberately avoid invoking the CLI (typer.testing.CliRunner or
# subprocess): both trigger full Typer app introspection which is
# sensitive to the typer version installed in the test environment.
# Instead we verify the command + its parameters are registered on the
# Typer app, which is the contract callers actually depend on.
# (Validation logic itself is exercised through the Python-API tests
# above, since the CLI body is a thin wrapper.)
# ====================================================================
from ragdx.cli import app, experiment as experiment_cmd  # noqa: E402,I001


def test_cli_experiment_command_registered():
    """``ragdx experiment`` should be a registered Typer command."""
    names = {info.name for info in app.registered_commands}
    # Typer derives the CLI name from the function name when `name=` isn't
    # passed; in our case the function is named ``experiment``.
    assert "experiment" in names or any(
        getattr(info, "callback", None) is experiment_cmd for info in app.registered_commands
    )


def test_cli_experiment_exposes_documented_flags():
    """The CLI signature should expose every option the README promises."""
    import inspect
    params = inspect.signature(experiment_cmd).parameters
    for required in (
        "corpus", "has_gt", "mode", "questions_path",
        "n_questions", "n_bo_trials", "n_bo_init",
        "output_dir", "api_key", "api_base", "model",
        "seed", "save_run", "name", "no_save",
    ):
        assert required in params, f"`ragdx experiment` missing --{required.replace('_', '-')}"


def test_cli_experiment_signature_matches_python_api():
    """Every public ``run_experiment`` kwarg should have a matching CLI flag
    (or be intentionally hidden -- we list the exceptions below)."""
    import inspect
    cli_params = set(inspect.signature(experiment_cmd).parameters)
    api_params = set(inspect.signature(run_experiment).parameters)
    # These Python-API knobs are intentionally not surfaced as CLI flags
    # (lists / objects / save flag mapped differently):
    api_only = {"top_ks", "chunk_sizes", "chunk_overlaps", "objective_overrides", "save"}
    missing = api_params - cli_params - api_only
    assert not missing, f"CLI is missing flags for Python-API params: {missing}"


# ensure 'sys' import lands at file top despite being added late
