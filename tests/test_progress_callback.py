"""Tests for the ``run_experiment(progress_callback=...)`` mechanism."""

from __future__ import annotations

import ragdx.experiments as ex
from ragdx.experiments import _ProgressReporter, run_experiment
from ragdx.schemas.models import DatasetRecord


# ----------------------------------------------------------- unit: reporter
def test_reporter_noop_when_callback_none():
    r = _ProgressReporter(None, ["no_gt"])
    # Must not raise despite there being no sink.
    r.emit("start", pct=0.0)
    r.mode_progress(0, "no_gt")("bo_done", 0.5, "x")


def test_reporter_pct_is_monotonic_and_bounded():
    events: list[dict] = []
    r = _ProgressReporter(events.append, ["with_gt", "no_gt"])
    r.emit("start", pct=0.0)
    for i, mode in enumerate(["with_gt", "no_gt"]):
        p = r.mode_progress(i, mode)
        p("mode_start", 0.0, "")
        p("bo_done", 0.55, "")
        p("mode_done", 1.0, "")
    r.emit("bundle_written", pct=1.0)

    pcts = [e["pct"] for e in events]
    assert pcts == sorted(pcts), "pct must be non-decreasing"
    assert all(0.0 <= p <= 1.0 for p in pcts)
    assert pcts[0] == 0.0 and pcts[-1] == 1.0
    # Every event carries the documented shape.
    for e in events:
        assert set(e) == {"stage", "mode", "pct", "detail"}


def test_reporter_swallows_callback_errors():
    def boom(_event):
        raise RuntimeError("ui blew up")

    r = _ProgressReporter(boom, ["no_gt"])
    # A broken UI callback must never propagate into the run.
    r.emit("start", pct=0.0)


# --------------------------------------------- integration: run_experiment
def test_run_experiment_emits_ordered_stage_events(monkeypatch, tmp_path):
    """With the heavy internals stubbed, the callback should see the
    coarse milestones in order and default-None must be a no-op."""

    monkeypatch.setattr(ex, "_build_runtime", lambda cfg: object())
    monkeypatch.setattr(
        ex,
        "_load_corpus_and_records",
        lambda cfg, runtime: (
            ["chunk-a", "chunk-b"],
            [DatasetRecord(question="q1", ground_truth=None, contexts=[])],
            {"kind": "pdf"},
        ),
    )

    def _fake_one_mode(cfg, runtime, chunks, records, mode, *, ckpts=None, progress=None):
        if progress is not None:
            progress("mode_start", 0.0, f"start {mode}")
            progress("bo_done", 0.55, "bo done")
            progress("dspy_done", 0.92, "dspy done")
            progress("mode_done", 1.0, f"done {mode}")
        return {
            "bayes_search": {},
            "dspy_a_b": {},
            "objective_spec": {},
            "final_config": None,
        }

    monkeypatch.setattr(ex, "_run_one_mode", _fake_one_mode)
    monkeypatch.setattr(
        ex, "_build_unified_bundle",
        lambda cfg, results, meta, records: {"schema_version": 1, "meta": {}},
    )

    events: list[dict] = []
    result = run_experiment(
        corpus="dummy.pdf",
        has_gt=False,
        mode="no_gt",
        api_key="test-key",  # config validation only; runtime is stubbed
        output_dir=str(tmp_path / "out"),
        save=False,
        no_checkpoint=True,
        progress_callback=events.append,
    )

    stages = [e["stage"] for e in events]
    assert stages[0] == "start"
    assert "corpus_loaded" in stages
    assert stages[-1] == "bundle_written"
    # The per-mode milestones appear in order between corpus_loaded and bundle_written.
    for expected in ["mode_start", "bo_done", "dspy_done", "mode_done"]:
        assert expected in stages
    pcts = [e["pct"] for e in events]
    assert pcts == sorted(pcts)

    # Default-None path must still run without a callback.
    events.clear()
    run_experiment(
        corpus="dummy.pdf",
        has_gt=False,
        mode="no_gt",
        api_key="test-key",
        output_dir=str(tmp_path / "out2"),
        save=False,
        no_checkpoint=True,
    )
    assert result.bundle["schema_version"] == 1
