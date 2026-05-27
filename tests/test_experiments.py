"""Tests for ``ragdx.experiments`` — validation + dispatch logic.

We don't run the full LLM pipeline here (that's the demo's job). Tests
cover the API contract: parameter validation, mode resolution, corpus
dispatch, and the public dataclass surface.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ragdx.experiments import (
    ExperimentConfig,
    ExperimentMode,
    ExperimentResult,
    _load_jsonl_questions,
    _looks_like_hf_dataset,
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
# CLI -- ragdx experiment subcommand
#
# Use subprocess rather than typer.testing.CliRunner: CliRunner walks
# the full Typer app and triggers PEP 604 / future-annotations parsing
# that's sensitive to the installed Typer version. Subprocess calls
# match what users actually do (`ragdx experiment ...` or `python -m
# ragdx ...`) and exercise the real binary entry point.
# ====================================================================

ROOT = Path(__file__).resolve().parents[1]


def _ragdx(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """Invoke ``python -m ragdx`` with PYTHONPATH set to the local src/."""
    full_env = os.environ.copy()
    full_env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + full_env.get("PYTHONPATH", "")
    full_env["OPENAI_API_KEY"] = full_env.get("OPENAI_API_KEY", "stub")
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "ragdx", *args],
        capture_output=True, text=True, env=full_env, timeout=60,
    )


def test_cli_experiment_help_runs():
    """``--help`` must render without pulling in dspy / ragas / langchain."""
    r = _ragdx("experiment", "--help")
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert "end-to-end RAG optimization experiment" in r.stdout
    assert "--has-gt" in r.stdout
    assert "--mode" in r.stdout


def test_cli_experiment_invalid_mode_rejected():
    r = _ragdx("experiment", "report.pdf", "--no-gt", "--mode", "nonsense")
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "mode must be one of" in combined or "Invalid value" in combined


def test_cli_experiment_with_gt_required_for_with_gt_mode():
    """``--no-gt --mode with_gt`` is the most common user mistake."""
    r = _ragdx("experiment", "report.pdf", "--no-gt", "--mode", "with_gt")
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "mode='with_gt'" in combined or "has_gt" in combined


def test_cli_experiment_both_mode_requires_has_gt():
    r = _ragdx("experiment", "report.pdf", "--no-gt", "--mode", "both")
    assert r.returncode != 0


def test_cli_experiment_listed_in_help():
    """The new subcommand should appear in the main ``ragdx --help`` listing."""
    # Force a wide terminal so the help table doesn't truncate the name.
    r = _ragdx("--help", env={"COLUMNS": "200"})
    assert r.returncode == 0, f"stderr={r.stderr!r}"
    assert "experiment" in r.stdout


# ensure 'sys' import lands at file top despite being added late
