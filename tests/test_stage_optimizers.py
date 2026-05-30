"""Tests for the PR3 stage optimizers.

These cover the pure-function contracts (``search_space``,
``build_trial_config``, ``cache_key``, ``needs_re_chunk``,
``per_call_overrides``) for each stage, plus the
``StageResult.to_bayes_search_bundle`` serialization.

The full ``optimize()`` loop is exercised end-to-end by the live
experiment runs (committed bundles under ``new_demo1`` /
``new_demo2``) and by the existing ``tests/test_experiments.py``
contract tests. Stubbing ragas + DSPy well enough to assert on the
loop output here would require more scaffolding than the test
provides confidence -- the integration coverage is better.
"""

from __future__ import annotations

import pytest

from ragdx.optim.stages import (
    ChunkingOptimizer,
    GenerationOptimizer,
    JointOptimizer,
    RetrievalOptimizer,
    StageContext,
    StageResult,
    StageTrial,
)
from ragdx.schemas.models import DatasetRecord
from ragdx.schemas.rag_config import ChunkerSpec, RAGConfig, RetrieverSpec


def _make_ctx(**overrides) -> StageContext:
    """Minimal StageContext for pure-function tests."""
    base = dict(
        base_config=RAGConfig(),
        chunks_master=["chunk-a", "chunk-b", "chunk-c"],
        records=[DatasetRecord(question="Q", ground_truth="GT")],
        objective=None,  # type: ignore[arg-type]
        metrics=[],
        runtime=None,
        n_bo_trials=4,
        n_bo_init=2,
        seed=0,
        label="test",
    )
    base.update(overrides)
    return StageContext(**base)


# ============================================================ Joint
def test_joint_search_space_has_three_axes():
    """Joint varies chunk_size + chunk_overlap + top_k together."""
    ctx = _make_ctx(
        chunk_sizes=[256, 512], chunk_overlaps=[0, 50], top_ks=[3, 5],
    )
    space = JointOptimizer().search_space(ctx)
    assert set(space.keys()) == {"chunk_size", "chunk_overlap", "top_k"}
    assert space["chunk_size"] == [256, 512]
    assert space["chunk_overlap"] == [0, 50]
    assert space["top_k"] == [3, 5]


def test_joint_build_trial_config_applies_both_chunker_and_retriever():
    """The trial config must reflect ALL three knobs the BO varies."""
    base = RAGConfig()
    trial = JointOptimizer().build_trial_config(
        base, {"chunk_size": 1024, "chunk_overlap": 100, "top_k": 7},
    )
    assert trial.chunker.chunk_size == 1024
    assert trial.chunker.chunk_overlap == 100
    assert trial.retriever.top_k == 7
    # The base must NOT have been mutated.
    assert base.chunker.chunk_size == 512


def test_joint_cache_key_invalidates_on_chunker_change_only():
    """top_k changes must NOT rebuild the vstore (forwarded per-call instead)."""
    j = JointOptimizer()
    k1 = j.cache_key({"chunk_size": 512, "chunk_overlap": 50, "top_k": 3})
    k2 = j.cache_key({"chunk_size": 512, "chunk_overlap": 50, "top_k": 7})
    k3 = j.cache_key({"chunk_size": 1024, "chunk_overlap": 50, "top_k": 3})
    assert k1 == k2  # same chunker -> same pipeline
    assert k1 != k3  # different chunk_size -> rebuild


def test_joint_per_call_overrides_top_k():
    """``top_k`` flows through pipeline.retrieve(top_k=...) per call."""
    assert JointOptimizer().per_call_overrides(
        {"chunk_size": 512, "chunk_overlap": 50, "top_k": 5}
    ) == {"top_k": 5}


def test_joint_needs_re_chunk_true():
    """Joint changes the chunker -> must signal re-chunking."""
    assert JointOptimizer().needs_re_chunk({"chunk_size": 256, "chunk_overlap": 0}) is True


# ============================================================ Chunking
def test_chunking_search_space_is_two_axes():
    """Chunking sweeps only chunker dims; top_k is held at base config."""
    ctx = _make_ctx(chunk_sizes=[256, 512], chunk_overlaps=[0, 50])
    space = ChunkingOptimizer().search_space(ctx)
    assert set(space.keys()) == {"chunk_size", "chunk_overlap"}
    assert "top_k" not in space


def test_chunking_build_trial_config_leaves_retriever_alone():
    """Holding the retriever fixed at base values is the whole point."""
    base = RAGConfig(retriever=RetrieverSpec(top_k=7))
    trial = ChunkingOptimizer().build_trial_config(
        base, {"chunk_size": 256, "chunk_overlap": 0},
    )
    assert trial.chunker.chunk_size == 256
    assert trial.retriever.top_k == 7  # unchanged
    assert base.retriever.top_k == 7   # unchanged


def test_chunking_cache_key_matches_joint_for_chunker_dims():
    """Same chunker -> same vstore. Different (chunk_size, overlap) -> rebuild."""
    c = ChunkingOptimizer()
    assert c.cache_key({"chunk_size": 512, "chunk_overlap": 50}) \
        != c.cache_key({"chunk_size": 1024, "chunk_overlap": 50})


# ============================================================ Retrieval
def test_retrieval_search_space_is_top_k_only():
    """Retrieval varies only the retrieval policy."""
    ctx = _make_ctx(top_ks=[1, 3, 5, 7])
    space = RetrievalOptimizer().search_space(ctx)
    assert set(space.keys()) == {"top_k"}


def test_retrieval_build_trial_config_leaves_chunker_alone():
    """Holding the chunker fixed = no re-chunking = much faster trials."""
    base = RAGConfig(chunker=ChunkerSpec(chunk_size=1024, chunk_overlap=100))
    trial = RetrievalOptimizer().build_trial_config(base, {"top_k": 10})
    assert trial.retriever.top_k == 10
    assert trial.chunker.chunk_size == 1024  # unchanged
    assert trial.chunker.chunk_overlap == 100


def test_retrieval_cache_key_is_constant():
    """A single pipeline serves every retrieval trial. ``cache_key``
    returning the same value for every call is what makes that work."""
    r = RetrievalOptimizer()
    assert r.cache_key({"top_k": 3}) == r.cache_key({"top_k": 7})


def test_retrieval_per_call_overrides_top_k():
    """top_k flows through pipeline.retrieve(top_k=...)."""
    assert RetrievalOptimizer().per_call_overrides({"top_k": 8}) == {"top_k": 8}


def test_retrieval_does_not_signal_re_chunk():
    """Retrieval trials must reuse the master chunks pool."""
    assert RetrievalOptimizer().needs_re_chunk({"top_k": 3}) is False


# ============================================================ Generation
def test_generation_build_trial_config_overrides_system_instruction():
    """Generation injects the MIPROv2-discovered instruction into the
    GeneratorSpec, leaving every other knob untouched."""
    base = RAGConfig()
    trial = GenerationOptimizer().build_trial_config(
        base, {"system_instruction": "You are a legal assistant."},
    )
    assert trial.generator.system_instruction == "You are a legal assistant."
    # Everything else preserved.
    assert trial.generator.model == base.generator.model
    assert trial.generator.api_base == base.generator.api_base
    assert trial.chunker == base.chunker
    assert trial.retriever == base.retriever


def test_generation_search_space_documents_its_non_grid_nature():
    """``search_space`` returns a documentation-only sentinel; MIPROv2's
    candidates aren't a discrete grid the framework can enumerate."""
    space = GenerationOptimizer().search_space(_make_ctx())
    assert "instruction_candidates" in space


# ============================================================ StageResult / StageTrial
def test_stage_trial_to_bundle_dict_round_trips_fields():
    """Trial -> dict shape must match the bundle's
    ``bayes_search.trials[i]`` schema the dashboard renders."""
    t = StageTrial(
        trial_index=2,
        params={"top_k": 5},
        n_chunks=42,
        scores={"faithfulness": 0.9},
        composite_score=1.5,
        feasible=True,
        violations=[],
        answers_preview=["short..."],
        records=[{"question": "Q", "answer": "A"}],
        elapsed_seconds=3.5,
    )
    d = t.to_bundle_dict()
    assert d["trial_index"] == 2
    assert d["params"] == {"top_k": 5}
    assert d["n_chunks"] == 42
    assert d["scores"] == {"faithfulness": 0.9}
    assert d["composite_score"] == 1.5
    assert d["feasible"] is True
    assert d["violations"] == []
    assert d["answers_preview"] == ["short..."]
    assert d["records"] == [{"question": "Q", "answer": "A"}]
    assert d["elapsed_seconds"] == 3.5


def test_stage_result_to_bayes_search_bundle_matches_expected_keys():
    """The bundle's ``bayes_search[mode]`` section relies on these keys.
    Locking them down prevents accidental schema drift."""
    res = StageResult(
        stage_name="joint",
        search_space={"top_k": [1, 3, 5]},
        trials=[],
        best_params={"top_k": 3},
        best_config=None,
        best_composite=1.2,
        objective_spec={"metrics": {"faithfulness": 1.5}, "constraints": {}, "mode": "weighted_sum"},
        n_init=2,
        max_trials=4,
    )
    bundle = res.to_bayes_search_bundle()
    assert set(bundle.keys()) == {
        "search_space", "n_init", "max_trials", "objective_spec",
        "trials", "best_params", "best_composite",
    }
    assert bundle["best_params"] == {"top_k": 3}
    assert bundle["best_composite"] == 1.2
    assert bundle["trials"] == []


# ============================================================ Cross-stage invariants
@pytest.mark.parametrize(
    "stage_cls", [JointOptimizer, ChunkingOptimizer, RetrievalOptimizer]
)
def test_bo_stages_share_stage_optimizer_interface(stage_cls):
    """Every BO-driven stage must implement the abstract methods so the
    default loop can drive it. Catching this here is cheaper than
    discovering it inside a multi-minute live run."""
    s = stage_cls()
    ctx = _make_ctx()
    assert isinstance(s.name, str) and s.name
    assert isinstance(s.search_space(ctx), dict)
    assert isinstance(s.build_trial_config(RAGConfig(), next(
        iter([{"chunk_size": 256, "chunk_overlap": 0, "top_k": 3}])
    )), RAGConfig)


def test_with_override_does_not_mutate_base_in_any_stage():
    """A non-negotiable invariant: stage optimizers must not mutate the
    base config (callers reuse it across modes)."""
    base = RAGConfig()
    snapshot = base.model_dump()
    JointOptimizer().build_trial_config(base, {"chunk_size": 256, "chunk_overlap": 0, "top_k": 3})
    ChunkingOptimizer().build_trial_config(base, {"chunk_size": 256, "chunk_overlap": 0})
    RetrievalOptimizer().build_trial_config(base, {"top_k": 11})
    GenerationOptimizer().build_trial_config(base, {"system_instruction": "X"})
    assert base.model_dump() == snapshot
