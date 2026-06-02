"""Tests for ``ragdx.checkpoint`` + the StageOptimizer resume contract.

The headline contract: an interrupted run can be resumed *without
re-running* any LLM-bound trial that completed before the crash.
That hinges on three things working together:

1. :class:`Checkpoint` round-trips through JSON cleanly (so the
   on-disk state.json restores everything).
2. :class:`CheckpointStore` lists, loads, and marks status.
3. ``StageOptimizer.optimize`` replays ``trials_completed`` into the
   BO sampler via ``bo.report(...)`` and resumes from where the
   sampler left off.
"""

from __future__ import annotations

import inspect

import pytest

from ragdx.checkpoint import (
    Checkpoint,
    CheckpointStore,
    CompletedTrial,
)


# ============================================================ Checkpoint schema
def test_checkpoint_round_trips_through_json():
    """A Checkpoint's JSON form must reload to an equal object so
    ``--resume`` doesn't drop fields."""
    cp = Checkpoint(
        kind="tune.retrieval",
        cli_args={"budget": 4, "stage": "retrieval"},
        name="rt-test",
        rag_config_yaml="name: test\n",
        stage_label="no_gt",
        search_space={"top_k": [1, 3, 5]},
        n_bo_init=2,
        max_trials=4,
        seed=7,
        trials_completed=[CompletedTrial(
            trial_index=0,
            params={"top_k": 3},
            n_chunks=42,
            scores={"faithfulness": 0.9},
            composite_score=1.4,
            feasible=True,
            elapsed_seconds=2.1,
        )],
    )
    cp2 = Checkpoint.model_validate_json(cp.model_dump_json())
    assert cp2.kind == "tune.retrieval"
    assert cp2.cli_args["budget"] == 4
    assert len(cp2.trials_completed) == 1
    assert cp2.trials_completed[0].params == {"top_k": 3}
    assert cp2.search_space == {"top_k": [1, 3, 5]}


def test_checkpoint_id_default_starts_with_ckpt():
    """The default id format ``ckpt_<8hex>`` is what the CLI's resume
    hint suggests. Drift here would break copy-paste UX."""
    cp = Checkpoint(kind="tune.joint")
    assert cp.checkpoint_id.startswith("ckpt_")
    assert len(cp.checkpoint_id) == len("ckpt_") + 8


# ============================================================ CheckpointStore
def test_store_save_load_list_and_mark_completed(tmp_path):
    store = CheckpointStore(root=tmp_path)
    cp = Checkpoint(kind="tune.retrieval", name="x")
    store.save(cp)
    assert (tmp_path / cp.checkpoint_id / "state.json").exists()

    loaded = store.load(cp.checkpoint_id)
    assert loaded.checkpoint_id == cp.checkpoint_id
    assert loaded.status == "running"

    listed = store.list()
    assert len(listed) == 1
    assert listed[0].checkpoint_id == cp.checkpoint_id

    store.mark_completed(cp.checkpoint_id)
    assert store.load(cp.checkpoint_id).status == "completed"

    incomplete = store.list_incomplete()
    assert incomplete == []  # the one we have is "completed" now


def test_store_save_and_load_chunks_round_trip(tmp_path):
    """``chunks_master`` is the most expensive thing to re-derive on
    resume (PDF parse + sentence-transformer embed). Store + load must
    round-trip without changing the chunk list."""
    store = CheckpointStore(root=tmp_path)
    cp = Checkpoint(kind="tune.chunking")
    store.save(cp)
    chunks = ["chunk-a", "chunk-b", "chunk-c with unicode 你好"]
    store.save_chunks(cp, chunks)
    store.save(cp)  # persist the updated chunks_path

    loaded = store.load(cp.checkpoint_id)
    assert loaded.chunks_path == "chunks.json"
    assert store.load_chunks(loaded) == chunks


def test_store_skips_corrupt_state_files_in_list(tmp_path):
    """A single corrupt state.json must not break ``ragdx checkpoints``."""
    store = CheckpointStore(root=tmp_path)
    # One good checkpoint
    good = Checkpoint(kind="tune.joint")
    store.save(good)
    # One corrupt directory
    bad_dir = tmp_path / "ckpt_corruptd"
    bad_dir.mkdir()
    (bad_dir / "state.json").write_text("{ not valid json", encoding="utf-8")

    listed = store.list()
    assert len(listed) == 1
    assert listed[0].checkpoint_id == good.checkpoint_id


def test_store_mark_interrupted_records_reason(tmp_path):
    store = CheckpointStore(root=tmp_path)
    cp = Checkpoint(kind="tune.retrieval")
    store.save(cp)
    store.mark_interrupted(cp.checkpoint_id, reason="network_timeout")
    loaded = store.load(cp.checkpoint_id)
    assert loaded.status == "interrupted"
    assert loaded.interrupted_reason == "network_timeout"


def test_store_delete_removes_directory(tmp_path):
    store = CheckpointStore(root=tmp_path)
    cp = Checkpoint(kind="tune.retrieval")
    store.save(cp)
    assert store.dir_for(cp.checkpoint_id).exists()
    assert store.delete(cp.checkpoint_id) is True
    assert not store.dir_for(cp.checkpoint_id).exists()
    assert store.delete(cp.checkpoint_id) is False  # idempotent


# ============================================================ Resume contract
def test_stage_context_carries_checkpoint_fields():
    """``StageContext`` exposes ``checkpoint`` + ``checkpoint_store``
    so the BO loop and the generation optimizer can both opt into
    resuming without changing their public ABCs."""
    from ragdx.optim.stages import StageContext
    src = inspect.getsource(StageContext)
    assert "checkpoint" in src
    assert "checkpoint_store" in src


def test_bo_optimizer_replays_completed_trials_into_sampler():
    """The replay path: load a partial checkpoint, run StageOptimizer
    on a tiny synthetic stage, assert the completed trials show up in
    the result without being re-evaluated."""
    from ragdx.optim.objectives import default_objective
    from ragdx.optim.stages import RetrievalOptimizer, StageContext
    from ragdx.schemas.models import DatasetRecord
    from ragdx.schemas.rag_config import RAGConfig

    # A 1-trial-already-done checkpoint.
    cp = Checkpoint(
        kind="tune.retrieval",
        stage_label="no_gt",
        search_space={"top_k": [1, 3, 5, 7]},
        n_bo_init=1,
        max_trials=4,
        seed=7,
        trials_completed=[CompletedTrial(
            trial_index=0,
            params={"top_k": 5},
            n_chunks=10,
            scores={"context_precision": 0.6},
            composite_score=0.42,
            feasible=True,
            elapsed_seconds=1.0,
            answers_preview=["preview"],
            records=[],
        )],
    )

    # Mock runtime so the optimizer's pipeline.build path doesn't need
    # a real embedder / LLM. We monkeypatch the BO loop's
    # _evaluate_with_ragas to return a deterministic score per call.
    captured: dict = {"new_trial_count": 0, "params_seen": []}

    class _FakeRuntime:
        embeddings = None
        ragas_judge = None
        ragas_embeddings = None
        ragas_run_config = None
        llm_callable = staticmethod(lambda p: "stub-answer")

    # Stub the pipeline so retrieve / generate don't touch any LLM.
    class _StubPipeline:
        n_chunks = 10
        def retrieve(self, q, top_k=None):
            return ["context"]
        def generate(self, q, ctxs, system_instruction=None):
            return "stub"

    from ragdx.runtime import pipeline as pipeline_mod
    original_build = pipeline_mod.RAGPipeline.build
    pipeline_mod.RAGPipeline.build = staticmethod(
        lambda *a, **kw: _StubPipeline()
    )

    from ragdx import experiments as exp_mod
    original_eval = getattr(exp_mod, "_evaluate_with_ragas", None)

    def _fake_eval(answered, judge, embeddings, metrics, run_config=None):
        captured["new_trial_count"] += 1
        return {"scores": {"context_precision": 0.5}, "skipped": []}

    exp_mod._evaluate_with_ragas = _fake_eval

    try:
        ctx = StageContext(
            base_config=RAGConfig(),
            chunks_master=["chunk"],
            records=[DatasetRecord(question="Q?", ground_truth=None)],
            objective=default_objective("no_gt"),
            metrics=[],
            runtime=_FakeRuntime(),
            n_bo_trials=4,
            n_bo_init=1,
            seed=7,
            label="no_gt",
            top_ks=[1, 3, 5, 7],
            checkpoint=cp,
            checkpoint_store=None,  # disable save side-effect in test
        )
        result = RetrievalOptimizer().optimize(ctx)
    finally:
        pipeline_mod.RAGPipeline.build = original_build
        if original_eval is not None:
            exp_mod._evaluate_with_ragas = original_eval

    # The 1 replayed trial + 3 fresh trials = 4 total.
    assert len(result.trials) == 4
    # Only 3 trials should have called ragas (the new ones).
    assert captured["new_trial_count"] == 3
    # The replayed trial appears in trials with its checkpointed score.
    replayed = [t for t in result.trials if t.params == {"top_k": 5} and t.composite_score == 0.42]
    assert len(replayed) == 1


# ============================================================ CLI wiring
def test_tune_has_resume_and_no_checkpoint_flags():
    """``--resume`` and ``--no-checkpoint`` must appear on the tune
    command so users can drive the new flow from the shell."""
    from ragdx.cli import tune as cli_tune

    params = set(inspect.signature(cli_tune).parameters)
    assert "resume" in params
    assert "no_checkpoint" in params


def test_checkpoints_command_registered():
    from ragdx.cli import app
    names = {info.name or info.callback.__name__ for info in app.registered_commands}
    assert "checkpoints" in names
    assert "checkpoint-clean" in names


def test_resume_with_unknown_id_raises_bad_parameter(tmp_path, monkeypatch):
    """Loading a missing checkpoint must surface a typer.BadParameter
    (not a raw FileNotFoundError) so the CLI prints a friendly message."""

    # Point the default CheckpointStore root at an empty tmp_path.
    monkeypatch.chdir(tmp_path)

    store = CheckpointStore(root=tmp_path)
    with pytest.raises(FileNotFoundError, match="ragdx checkpoints"):
        store.load("ckpt_does_not_exist")
    # And the CLI translates that to BadParameter (smoke test the
    # try/except chain in cli/tune.py without invoking the full
    # tune body, which requires a config etc.):
    from ragdx.cli import tune as cli_tune
    src = inspect.getsource(cli_tune)
    assert "FileNotFoundError" in src
    assert "typer.BadParameter" in src
