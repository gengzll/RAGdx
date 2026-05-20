"""ragdx — RAG Diagnosis & Optimization Library.

Top-level exports expose the most commonly used entry points so users can
write::

    from ragdx import (
        UnifiedEvaluator,
        RAGDiagnosisEngine,
        OptimizationPlanner,
        OptimizationExecutor,
        RunStore,
        get_settings,
        get_llm_callable,
    )

For provider-specific or lower-level APIs (custom LLM providers, engines,
adapters), import from the relevant subpackage directly — e.g.
``from ragdx.llm import register_provider`` or
``from ragdx.engines.embedding_eval import EmbeddingEvaluator``.
"""

from __future__ import annotations

__version__ = "0.9.0"

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
from ragdx.llm import (
    DEFAULT_MODELS,
    LLMCallable,
    LLMProvider,
    build_provider,
    get_llm_callable,
    list_providers,
    register_provider,
)
from ragdx.optim.executor import OptimizationExecutor
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

__all__ = [
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
    "OptimizationExecutor",
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
