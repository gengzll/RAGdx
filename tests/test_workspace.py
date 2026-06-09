"""Tests for the per-experiment workspace module.

Covers the local-only behaviour (manifest read/write, history log,
``current`` pointer updates, path resolution). Doesn't touch the CLI
subcommands that need a live LLM (those have manual e2e demos under
``demo_*_v2/``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ragdx.workspace import (
    HistoryEntry,
    WorkspaceError,
    init_workspace,
    list_workspaces,
    load_workspace,
    workspaces_root,
)


# --------------------------------------------------------------------- helpers
@pytest.fixture
def isolated_workspaces_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin ``workspaces_root()`` to a per-test tmp dir.

    Without this the tests would write into the real ``./workspaces/``
    folder of whoever's checkout we're in.
    """
    monkeypatch.setenv("RAGDX_WORKSPACES", str(tmp_path))
    return tmp_path


# ============================================================ workspaces_root
def test_workspaces_root_respects_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAGDX_WORKSPACES", str(tmp_path))
    assert workspaces_root() == tmp_path.resolve()


def test_workspaces_root_falls_back_to_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv("RAGDX_WORKSPACES", raising=False)
    monkeypatch.chdir(tmp_path)
    assert workspaces_root() == tmp_path / "workspaces"


# =================================================================== init
def test_init_workspace_creates_manifest(isolated_workspaces_root: Path) -> None:
    ws = init_workspace("alpha", evaluator="deepeval", mode="no_gt")
    assert ws.root == isolated_workspaces_root / "alpha"
    assert ws.manifest_path.exists()
    assert ws.evaluator == "deepeval"
    assert ws.mode == "no_gt"
    # Re-load round-trips identically.
    ws2 = load_workspace("alpha")
    assert ws2.name == "alpha"
    assert ws2.evaluator == "deepeval"
    assert ws2.created_at == ws.created_at


def test_init_workspace_rejects_bad_names(isolated_workspaces_root: Path) -> None:
    for bad in ("", ".hidden", "with/slash", "with\\backslash"):
        with pytest.raises(WorkspaceError, match="Invalid workspace name"):
            init_workspace(bad)


def test_init_workspace_double_init_blocks_without_overwrite(
    isolated_workspaces_root: Path,
) -> None:
    init_workspace("dup")
    with pytest.raises(WorkspaceError, match="already exists"):
        init_workspace("dup")
    # Overwrite resets the manifest but keeps the directory.
    ws2 = init_workspace("dup", overwrite=True, evaluator="deepeval")
    assert ws2.evaluator == "deepeval"


def test_init_workspace_rejects_bad_evaluator_via_cli_layer(
    isolated_workspaces_root: Path,
) -> None:
    """The core ``init_workspace`` doesn't enforce evaluator-validity --
    that's the CLI layer's job. We just verify the core stores whatever
    we pass so a hand-edited YAML can roundtrip experimental backends."""
    ws = init_workspace("any-evaluator", evaluator="some-future-evaluator")
    assert ws.evaluator == "some-future-evaluator"


# =================================================================== load
def test_load_workspace_missing_raises(isolated_workspaces_root: Path) -> None:
    with pytest.raises(WorkspaceError, match="No workspace found"):
        load_workspace("does-not-exist")


def test_load_workspace_malformed_yaml_raises(
    isolated_workspaces_root: Path,
) -> None:
    """A truncated / malformed manifest must surface as WorkspaceError
    (not a bare yaml.YAMLError) so the CLI's error wrapper catches it
    cleanly. The simplest reliable corruption: write non-mapping YAML
    (a list), which the loader rejects on type."""
    bad = isolated_workspaces_root / "broken"
    bad.mkdir()
    (bad / "workspace.yaml").write_text(
        "- one\n- two\n- three\n", encoding="utf-8",
    )
    with pytest.raises(WorkspaceError):
        load_workspace("broken")


def test_load_workspace_accepts_absolute_path(
    isolated_workspaces_root: Path,
) -> None:
    init_workspace("beta")
    ws = load_workspace(str(isolated_workspaces_root / "beta"))
    assert ws.name == "beta"


# =================================================================== paths
def test_path_resolves_relative_and_absolute(
    isolated_workspaces_root: Path,
) -> None:
    ws = init_workspace("paths")
    rel = ws.path("foo.json")
    assert rel == ws.root / "foo.json"
    # Absolute paths pass through unchanged.
    abs_in = isolated_workspaces_root / "elsewhere" / "x.json"
    assert ws.path(abs_in) == abs_in


def test_corpus_path_falls_back_to_yaml(
    isolated_workspaces_root: Path,
) -> None:
    """When ``corpus`` is unset, ``corpus_path`` peeks at the YAML."""
    ws = init_workspace("with-yaml-corpus")
    yaml_cfg = {
        "name": "tmp", "runtime": "langchain",
        "corpus": {"kind": "pdf", "path": "stub.pdf"},
        "chunker": {"chunk_size": 256, "chunk_overlap": 0},
        "retriever": {"top_k": 3},
        "generator": {"model": "openai/x"},
    }
    ws.rag_config_path().write_text(yaml.safe_dump(yaml_cfg), encoding="utf-8")
    assert ws.corpus_path() == ws.root / "stub.pdf"


def test_corpus_path_returns_none_when_unresolvable(
    isolated_workspaces_root: Path,
) -> None:
    """No corpus override + no YAML on disk = None (CLI prompts the user)."""
    ws = init_workspace("no-corpus")
    assert ws.corpus_path() is None


# =================================================================== log / current
def test_log_updates_current_pointer(isolated_workspaces_root: Path) -> None:
    ws = init_workspace("log-test")
    out = ws.path("baseline.eval.json")
    out.write_text("{}", encoding="utf-8")  # so resolve() works

    entry = ws.log("eval", output=out, run_id="abc12345")
    assert entry.command == "eval"
    assert entry.output == "baseline.eval.json"
    assert entry.run_id == "abc12345"
    assert ws.current["latest_eval"] == "baseline.eval.json"
    assert ws.current["baseline_run_id"] == "abc12345"

    # Persists across reloads.
    ws2 = load_workspace("log-test")
    assert ws2.history[-1].command == "eval"
    assert ws2.current["latest_eval"] == "baseline.eval.json"


def test_log_for_each_command_kind_updates_correct_pointer(
    isolated_workspaces_root: Path,
) -> None:
    """``log()`` must drop the artifact under the right ``current.*`` key
    so later commands can pick it up by category. Maps mirror the CLI
    verb set; if a new verb is added the map must learn it."""
    ws = init_workspace("kinds")
    cases = {
        "eval": "latest_eval",
        "diagnose": "latest_diagnose",
        "tune-rag": "latest_tune_rag",
        "tune-prompt": "latest_tune_prompt",
        "report": "latest_report",
    }
    for verb, expected_key in cases.items():
        out = ws.path(f"{verb}.out")
        out.write_text("{}", encoding="utf-8")
        ws.log(verb, output=out)
        assert ws.current[expected_key] == f"{verb}.out", verb


def test_log_with_output_outside_workspace_keeps_absolute(
    isolated_workspaces_root: Path, tmp_path: Path,
) -> None:
    """An artifact written outside the workspace folder is recorded
    with its absolute path (we can't make a sensible relative one)."""
    ws = init_workspace("outside")
    outsider = tmp_path / "elsewhere.json"
    outsider.write_text("{}", encoding="utf-8")
    entry = ws.log("eval", output=outsider)
    assert entry.output is not None
    assert Path(entry.output).is_absolute()


# =================================================================== latest
def test_latest_returns_none_when_no_artifact(
    isolated_workspaces_root: Path,
) -> None:
    ws = init_workspace("empty")
    assert ws.latest("eval") is None


def test_latest_returns_absolute_path(isolated_workspaces_root: Path) -> None:
    ws = init_workspace("latest-test")
    out = ws.path("eval.json")
    out.write_text("{}", encoding="utf-8")
    ws.log("eval", output=out)
    latest = ws.latest("eval")
    assert latest is not None
    assert latest.is_absolute()
    assert latest == out


# =================================================================== list
def test_list_workspaces_sorted(isolated_workspaces_root: Path) -> None:
    init_workspace("zeta")
    init_workspace("alpha")
    init_workspace("mu")
    assert list_workspaces() == ["alpha", "mu", "zeta"]


def test_list_workspaces_skips_non_workspace_dirs(
    isolated_workspaces_root: Path,
) -> None:
    """Directories without workspace.yaml are filtered out so a stray
    ``logs/`` folder under ``workspaces/`` doesn't get listed."""
    init_workspace("real")
    (isolated_workspaces_root / "not-a-ws").mkdir()
    assert list_workspaces() == ["real"]


# =================================================================== HistoryEntry
def test_history_entry_is_picklable_dataclass() -> None:
    """``HistoryEntry`` is a plain frozen-style dataclass; we use it in
    structures that may be pickled (cache invalidation, etc.)."""
    import pickle
    e = HistoryEntry(
        ts="2026-01-01T00:00:00Z", command="eval",
        output="out.json", run_id="r1", extra={"k": "v"},
    )
    e2 = pickle.loads(pickle.dumps(e))
    assert e2 == e


# =================================================================== schema version
def test_manifest_writes_schema_version(isolated_workspaces_root: Path) -> None:
    """Future migrations key off ``schema_version``; the loader doesn't
    enforce it yet, but the writer must include it."""
    ws = init_workspace("schema")
    data = yaml.safe_load(ws.manifest_path.read_text(encoding="utf-8"))
    assert isinstance(data.get("schema_version"), int)
    assert data["schema_version"] >= 1
