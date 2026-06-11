"""Tests for ``ragdx experiment`` checkpoint/resume wiring.

Covers the ``_ExperimentCheckpoints`` registry (group creation, resume
matching, completed-stage inclusion) without any live LLM calls. The
stage-level replay behaviour itself is covered by test_checkpoint.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdx.checkpoint import CheckpointStore
from ragdx.experiments import _ExperimentCheckpoints


@pytest.fixture
def scoped_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the settings-resolved storage root at a tmp dir so the
    CheckpointStore (root=None) lands here, not in the repo's .ragdx."""
    monkeypatch.setenv("RAGDX_ROOT", str(tmp_path / ".ragdx"))
    return tmp_path


def test_disabled_registry_is_inert(scoped_root: Path) -> None:
    ck = _ExperimentCheckpoints(enabled=False)
    assert ck.store is None
    assert ck.get("no_gt", "joint") is None
    ck.complete(None)  # no-op, no crash


def test_fresh_run_creates_group_and_checkpoints(scoped_root: Path) -> None:
    reg = _ExperimentCheckpoints(enabled=True)
    assert reg.group_id.startswith("exp_")

    c1 = reg.get("no_gt", "joint")
    c2 = reg.get("no_gt", "generation")
    assert c1 is not None and c2 is not None
    assert c1.checkpoint_id != c2.checkpoint_id
    assert c1.kind == "experiment"
    assert c1.cli_args["experiment_group"] == reg.group_id
    assert c1.cli_args["mode"] == "no_gt"
    assert c1.cli_args["stage"] == "joint"
    # Same key returns the same object (idempotent).
    assert reg.get("no_gt", "joint") is c1

    # Both persisted on disk under the scoped root.
    store = CheckpointStore()
    ids = {c.checkpoint_id for c in store.list()}
    assert {c1.checkpoint_id, c2.checkpoint_id} <= ids


def test_complete_marks_checkpoint(scoped_root: Path) -> None:
    reg = _ExperimentCheckpoints(enabled=True)
    c1 = reg.get("no_gt", "joint")
    reg.complete(c1)
    store = CheckpointStore()
    loaded = store.load(c1.checkpoint_id)
    assert loaded.status == "completed"


def test_resume_auto_picks_latest_interrupted_group(scoped_root: Path) -> None:
    # Group A: fully completed (not resumable).
    reg_a = _ExperimentCheckpoints(enabled=True)
    ca = reg_a.get("no_gt", "joint")
    reg_a.complete(ca)

    # Group B: joint completed, generation interrupted (resumable).
    reg_b = _ExperimentCheckpoints(enabled=True)
    cb_joint = reg_b.get("no_gt", "joint")
    reg_b.complete(cb_joint)
    cb_gen = reg_b.get("no_gt", "generation")
    store = CheckpointStore()
    cb_gen.status = "interrupted"
    cb_gen.interrupted_reason = "network_timeout"
    store.save(cb_gen)

    resumed = _ExperimentCheckpoints(resume="auto", enabled=True)
    assert resumed.group_id == reg_b.group_id
    # BOTH checkpoints of the group are loaded -- the completed joint
    # one replays instantly, the interrupted generation one continues.
    r_joint = resumed.get("no_gt", "joint")
    r_gen = resumed.get("no_gt", "generation")
    assert r_joint.checkpoint_id == cb_joint.checkpoint_id
    assert r_joint.status == "completed"  # left as-is
    assert r_gen.checkpoint_id == cb_gen.checkpoint_id
    assert r_gen.status == "running"  # re-marked for continuation
    assert r_gen.interrupted_reason == ""


def test_resume_explicit_group_id(scoped_root: Path) -> None:
    reg = _ExperimentCheckpoints(enabled=True)
    c = reg.get("with_gt", "joint")
    store = CheckpointStore()
    c.status = "interrupted"
    store.save(c)

    resumed = _ExperimentCheckpoints(resume=reg.group_id, enabled=True)
    assert resumed.group_id == reg.group_id
    assert resumed.get("with_gt", "joint").checkpoint_id == c.checkpoint_id


def test_resume_auto_without_candidates_raises(scoped_root: Path) -> None:
    with pytest.raises(ValueError, match="No interrupted experiment"):
        _ExperimentCheckpoints(resume="auto", enabled=True)


def test_resume_unknown_group_raises(scoped_root: Path) -> None:
    with pytest.raises(ValueError, match="No experiment checkpoints"):
        _ExperimentCheckpoints(resume="exp_deadbeef", enabled=True)


def test_run_experiment_signature_has_resume() -> None:
    """The public entry point exposes resume/no_checkpoint so the CLI
    passthrough can't silently drop them."""
    import inspect

    from ragdx.experiments import run_experiment
    params = inspect.signature(run_experiment).parameters
    assert "resume" in params
    assert "no_checkpoint" in params
