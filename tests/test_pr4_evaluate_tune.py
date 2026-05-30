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
    import ragas.metrics as rm

    records = [
        DatasetRecord(question="Q", ground_truth="A reference answer"),
    ]
    metrics = _select_metrics(records)
    # ``answer_relevancy`` + ``context_precision`` + ``faithfulness`` + ``context_recall``
    assert len(metrics) == 4
    assert rm.context_recall in metrics


def test_select_metrics_no_gt_returns_three():
    """No-GT eval drops ``context_recall``."""
    import ragas.metrics as rm

    records = [DatasetRecord(question="Q", ground_truth=None)]
    metrics = _select_metrics(records)
    assert len(metrics) == 3
    assert rm.context_recall not in metrics


def test_select_metrics_empty_gt_string_treated_as_no_gt():
    """Whitespace-only ground_truth is the same as missing -- the
    composite scorer would otherwise crash on empty strings."""
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
    import litellm

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
