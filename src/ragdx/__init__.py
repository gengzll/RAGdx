"""ragdx — RAG Diagnosis & Optimization Library.

Top-level exports expose the most commonly used entry points so users can
write::

    from ragdx import (
        UnifiedEvaluator,
        RAGDiagnosisEngine,
        OptimizationPlanner,
        RunStore,
        get_settings,
        get_llm_callable,
        run_experiment,
    )

For provider-specific or lower-level APIs (custom LLM providers, engines,
adapters), import from the relevant subpackage directly — e.g.
``from ragdx.llm import register_provider`` or
``from ragdx.engines.embedding_eval import EmbeddingEvaluator``.
"""

from __future__ import annotations

__version__ = "0.9.0"


def _install_vertexai_shims() -> None:
    """Pre-import shim for ragas 0.3.x on langchain-community 0.4.x.

    Modern ``langchain-community`` (≥ 0.3.6) moved the Google Vertex
    chat / completion classes out into the separate
    ``langchain-google-vertexai`` package, but ragas 0.3.2 still does
    a hard ``from langchain_community.chat_models.vertexai import
    ChatVertexAI`` at import time. The class is only used inside ragas
    when the caller explicitly picks Vertex as their judge LLM -- our
    GLM / OpenAI flows never trigger it -- so a thin stub module that
    ``ChatVertexAI`` / ``VertexAI`` resolve to satisfies the import
    without pulling in google-cloud-aiplatform.

    No-op when:
    * ``langchain_community`` isn't installed at all (no ragas user
      flow possible from this env, so no shim needed).
    * The real ``vertexai`` submodule is present (don't shadow a
      real install).
    """
    import importlib.util
    import sys
    import types

    # Bail early if ``langchain_community`` itself isn't installed.
    # ``find_spec("langchain_community.x.y")`` imports the parent
    # package as a side-effect, which would raise ModuleNotFoundError
    # on a minimal install (e.g. ragdx[dev] CI without the langchain
    # extras). Probing the top-level spec by itself is safe.
    if importlib.util.find_spec("langchain_community") is None:
        return

    for fullname, attr in (
        ("langchain_community.chat_models.vertexai", "ChatVertexAI"),
        ("langchain_community.llms.vertexai", "VertexAI"),
    ):
        if fullname in sys.modules:
            continue
        try:
            if importlib.util.find_spec(fullname) is not None:
                continue
        except (ImportError, ValueError):
            # Defensive: a broken intermediate package shouldn't
            # block ragdx from importing. Fall through to the stub.
            pass
        stub = types.ModuleType(fullname)
        setattr(stub, attr, type(attr, (), {}))
        sys.modules[fullname] = stub


_install_vertexai_shims()

from ragdx.config import (
    ExecutionSettings,
    LLMSettings,
    Settings,
    StorageSettings,
    get_settings,
)
from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.core.evaluator import UnifiedEvaluator
from ragdx.engines.embedding_eval import EmbeddingEvaluator
from ragdx.engines.llm_diagnosis import LLMDiagnosisExplainer
from ragdx.errors import (
    ConfigError,
    DependencyError,
    EvaluationError,
    LLMConfigError,
    LLMError,
    RagdxError,
    RunnerError,
    RunnerMissingError,
    StorageError,
)
from ragdx.experiments import (
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
)
from ragdx.llm import (
    DEFAULT_MODELS,
    LLMCallable,
    LLMProvider,
    build_provider,
    get_llm_callable,
    list_providers,
    register_provider,
)
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import (
    CausalSignal,
    DatasetRecord,
    DiagnosisHypothesis,
    DiagnosisReport,
    EvaluationResult,
    FeedbackEvent,
    OptimizationExperiment,
    OptimizationPlan,
    OptimizationSession,
    OptimizationTrial,
    QueryTrace,
    SavedRun,
    ToolRunResult,
    TraceSpan,
)
from ragdx.storage.run_store import RunStore

__all__ = [  # noqa: RUF022 — grouped by category, not alphabetical
    "__version__",
    # config
    "Settings",
    "StorageSettings",
    "LLMSettings",
    "ExecutionSettings",
    "get_settings",
    # errors
    "RagdxError",
    "ConfigError",
    "DependencyError",
    "LLMError",
    "LLMConfigError",
    "EvaluationError",
    "RunnerError",
    "RunnerMissingError",
    "StorageError",
    # llm
    "LLMProvider",
    "LLMCallable",
    "DEFAULT_MODELS",
    "build_provider",
    "get_llm_callable",
    "list_providers",
    "register_provider",
    # engines / evaluators
    "UnifiedEvaluator",
    "EmbeddingEvaluator",
    "LLMDiagnosisExplainer",
    "RAGDiagnosisEngine",
    # optim
    "OptimizationPlanner",
    # experiments (one-call end-to-end driver)
    "run_experiment",
    "ExperimentConfig",
    "ExperimentResult",
    # storage
    "RunStore",
    # schemas
    "DatasetRecord",
    "EvaluationResult",
    "DiagnosisHypothesis",
    "DiagnosisReport",
    "OptimizationExperiment",
    "OptimizationPlan",
    "OptimizationSession",
    "OptimizationTrial",
    "SavedRun",
    "ToolRunResult",
    "QueryTrace",
    "TraceSpan",
    "FeedbackEvent",
    "CausalSignal",
]
