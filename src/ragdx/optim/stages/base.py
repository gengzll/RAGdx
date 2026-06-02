"""Shared types for stage-targeted optimizers.

A :class:`StageOptimizer` is a small, focused thing: take a base RAG
config, an evaluation context, and a budget; return the best
single-stage override it found. The default loop is plain Bayesian
search; stages that don't fit (e.g. generation-prompt tuning via
DSPy) override :meth:`optimize` directly.

The context object is intentionally a thin wrapper around what the
experiment runtime already builds (``_Runtime`` in
:mod:`ragdx.experiments`). Stages reach in for the embedder, LLM
callable, judge LLM, and ragas knobs without re-deriving them.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ragdx.optim.bayes_search import BayesianSearch
from ragdx.optim.objectives import CompositeObjective
from ragdx.runtime.pipeline import RAGPipeline
from ragdx.schemas.models import DatasetRecord
from ragdx.schemas.rag_config import RAGConfig


@dataclass
class StageContext:
    """What every stage needs to evaluate a candidate config.

    Built once per ``ragdx experiment`` / ``ragdx optimize`` call by
    the orchestrating workflow and reused across stages.
    """

    base_config: RAGConfig
    """Starting point. Stages produce overrides on top of this."""

    chunks_master: list[str]
    """Pre-chunked corpus pool. ChunkingOptimizer / JointOptimizer
    re-chunk per trial (when the chunker spec changes); the others
    treat this as immutable."""

    records: list[DatasetRecord]
    """The eval suite. The optimizer scores each candidate config
    against these records."""

    objective: CompositeObjective
    """Composite scorer turning ragas metrics into one number."""

    metrics: list = field(default_factory=list)
    """The ragas metric instances to evaluate against (filtered by GT mode)."""

    runtime: Any = None
    """The ``_Runtime`` instance from :mod:`ragdx.experiments`.
    Provides ``embeddings``, ``llm_callable``, ``ragas_judge``,
    ``ragas_embeddings``, ``ragas_run_config``, ``system_instruction``,
    ``llm_max_concurrent``. We pass it as ``Any`` to avoid a forward
    reference into ``experiments.py``."""

    n_bo_trials: int = 8
    n_bo_init: int = 3
    seed: int = 7

    re_chunk_fn: Callable[[Any], list[str]] | None = None
    """Optional callable that takes a ``ChunkerSpec`` and returns the
    chunks pool re-built at that chunking. Used by ChunkingOptimizer /
    JointOptimizer for PDF corpora (where chunk_size affects the
    pool). If ``None``, stages that need re-chunking fall back to
    ``chunks_master`` (the chunk_size knob still varies the cache key
    but the actual chunks don't change -- meaningful for HF / JSONL
    sources)."""

    label: str = ""
    """Free-form mode label (``"with_gt"`` / ``"no_gt"``) used in log lines."""

    # --- search axes (per-stage default sweep ranges) ----------------
    chunk_sizes: list[int] = field(default_factory=lambda: [256, 512, 1024])
    chunk_overlaps: list[int] = field(default_factory=lambda: [0, 50, 100])
    top_ks: list[int] = field(default_factory=lambda: [1, 3, 5, 7])

    # --- GenerationOptimizer (DSPy MIPROv2) knobs --------------------
    mipro_auto: str = "light"
    """``optimizer_kwargs["auto"]`` for MIPROv2: ``"light"`` (~3
    candidates, ~5 min), ``"medium"`` (~10-20 candidates, ~30 min), or
    ``"heavy"`` (~30+ candidates, ~90 min). Larger budgets give the
    grounded proposer more room to find a prompt that beats the seed."""

    # --- Checkpoint (PR6+) -------------------------------------------
    checkpoint: Any = None
    """Optional :class:`ragdx.checkpoint.Checkpoint`. When set, the BO
    loop replays its completed trials into the sampler and saves an
    updated state after every new trial. ``None`` keeps the
    pre-checkpoint behaviour exactly."""

    checkpoint_store: Any = None
    """Optional :class:`ragdx.checkpoint.CheckpointStore` for the
    per-trial save. Ignored when ``checkpoint`` is ``None``."""

    dspy_optimizer: str = "mipro"
    """Which DSPy teleprompter ``--stage generation`` should run:

    * ``"mipro"`` (default): MIPROv2 — Bayesian search over
      (instruction x few-shot demo) candidates. Strongest but
      slowest; writes BOTH ``system_instruction`` AND
      ``few_shot_demos`` to ``best_config``.
    * ``"copro"``: COPRO — iterative LLM-driven instruction
      rewriting. Demos unchanged. Writes ONLY ``system_instruction``.
      Fastest. Best fit when production deployment ignores demos.
    * ``"bootstrap_fewshot"``: BootstrapFewShot — generates few-shot
      demos by running the seed program on the trainset. Instruction
      unchanged. Writes ONLY ``few_shot_demos``.

    The chosen optimizer determines which fields of the winning
    ``RAGConfig`` get populated. See PR7's
    ``GenerationOptimizer.optimize`` for the dispatch.
    """

    dspy_metric: str = "auto"
    """Which DSPy-side inner-loop metric to use for the MIPROv2 search.

    ``"auto"`` (default): pick ``token_f1`` when records carry ground
    truth, ``ragas`` otherwise (PR6+).
    ``"ragas"``: use the ragas composite (context_precision +
    faithfulness + answer_relevancy) as a single weighted score per
    example. Discriminative even in no-GT mode where ``llm_judge`` saturates.
    ``"llm_judge"``: legacy single-call LLM faithfulness judge from
    :class:`DSPyAdapter` (pre-PR6 default). Cheap but saturates on
    permissive judge LMs (e.g. GLM-4-Flash returns 1.0 for everything).
    ``"token_f1"``: token-overlap F1 against ground truth. Requires
    GT-populated records.
    """


@dataclass
class StageTrial:
    """A single trial's record. Convertible to the bundle's
    ``bayes_search.trials[i]`` shape."""

    trial_index: int
    params: dict[str, Any]
    n_chunks: int
    scores: dict[str, float]
    composite_score: float
    feasible: bool
    violations: list[str]
    answers_preview: list[str]
    records: list[dict[str, Any]]
    elapsed_seconds: float

    def to_bundle_dict(self) -> dict[str, Any]:
        """Shape consumed by the experiment dashboard / report."""
        return {
            "trial_index": self.trial_index,
            "params": self.params,
            "n_chunks": self.n_chunks,
            "scores": self.scores,
            "composite_score": self.composite_score,
            "feasible": self.feasible,
            "violations": self.violations,
            "answers_preview": self.answers_preview,
            "records": self.records,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass
class StageResult:
    """What a stage optimizer returns."""

    stage_name: str
    search_space: dict[str, list]
    trials: list[StageTrial]
    best_params: dict[str, Any] | None
    best_config: RAGConfig | None
    best_composite: float | None
    objective_spec: dict[str, Any]
    n_init: int
    max_trials: int
    extras: dict[str, Any] = field(default_factory=dict)
    """Stage-specific extras (e.g. GenerationOptimizer's
    ``baseline_scores`` / ``optimized_scores`` / DSPy instructions)
    that don't fit the standard trial schema."""

    def to_bayes_search_bundle(self) -> dict[str, Any]:
        """Render as the ``bayes_search[mode]`` bundle section. Used by
        the BO-style stages (Joint, Chunking, Retrieval)."""
        return {
            "search_space": self.search_space,
            "n_init": self.n_init,
            "max_trials": self.max_trials,
            "objective_spec": self.objective_spec,
            "trials": [t.to_bundle_dict() for t in self.trials],
            "best_params": self.best_params,
            "best_composite": self.best_composite,
        }


class StageOptimizer(ABC):
    """Vary one slice of a :class:`RAGConfig`; hold the others fixed.

    Subclasses define **what** to vary (``search_space``,
    ``build_trial_config``) and **when** to invalidate the vstore
    cache (``cache_key`` / ``needs_re_chunk``). The default
    :meth:`optimize` runs ragdx's Bayesian search using these.
    """

    name: ClassVar[str]
    """Short label used in logs and in the bundle's
    ``meta.stages_run[]`` (PR4)."""

    @abstractmethod
    def search_space(self, ctx: StageContext) -> dict[str, list]:
        """Return the axes the optimizer varies, in
        ``{name: [discrete_values, ...]}`` form."""

    @abstractmethod
    def build_trial_config(self, base: RAGConfig, params: dict[str, Any]) -> RAGConfig:
        """Apply ``params`` (a point from ``search_space``) to ``base``."""

    def cache_key(self, params: dict[str, Any]) -> tuple:
        """Pipeline-cache key. Default: re-use the same pipeline for
        every trial (good for stages that only change retrieval params).
        Override when changing ``params`` requires rebuilding the
        vstore (chunking changes)."""
        return ()

    def needs_re_chunk(self, params: dict[str, Any]) -> bool:
        """Whether this trial's params change the chunking. Stages
        that vary chunker return True; others return False."""
        return False

    def chunks_for_trial(
        self, ctx: StageContext, trial_config: RAGConfig, params: dict[str, Any]
    ) -> list[str]:
        """Resolve the chunks pool for this trial. If the stage
        signals re-chunking and the context provides a re-chunk
        callable, use it; otherwise fall back to ``chunks_master``."""
        if self.needs_re_chunk(params) and ctx.re_chunk_fn is not None:
            return ctx.re_chunk_fn(trial_config.chunker)
        return ctx.chunks_master

    def per_call_overrides(self, params: dict[str, Any]) -> dict[str, Any]:
        """Per-call overrides forwarded to ``pipeline.retrieve`` /
        ``generate``. Default: empty (use config defaults). Override
        to surface BO params that don't change the pipeline's identity
        (e.g. RetrievalOptimizer varying ``top_k`` per-call rather than
        per-pipeline)."""
        return {}

    # ------------------------------------------------------------------
    # Default BO-driven optimize loop
    # ------------------------------------------------------------------
    def optimize(self, ctx: StageContext) -> StageResult:
        """Run a Bayesian search over ``search_space``.

        Subclasses can override for non-BO paths (Generation uses
        DSPy's MIPROv2 instead).

        Checkpointing
        -------------
        If ``ctx.checkpoint`` is set, the optimizer:

        * Replays each :class:`CompletedTrial` from
          ``checkpoint.trials_completed`` into the BO sampler via
          ``bo.report(...)`` so the sampler's posterior matches what
          the previous (interrupted) run had learned.
        * Re-uses the recorded ``StageTrial`` objects directly in
          ``trials_out`` (no LLM re-call).
        * Saves the updated checkpoint after every completed trial,
          so a crash mid-loop loses at most one trial.

        Without ``ctx.checkpoint``: behaviour is unchanged from PR3.
        """
        # Lazy imports so stage-only callers don't drag in the
        # checkpoint / ragas modules unless needed.
        from ragdx.checkpoint import CompletedTrial
        from ragdx.experiments import _evaluate_with_ragas

        space = self.search_space(ctx)
        bo = BayesianSearch(
            space, n_init=ctx.n_bo_init, max_trials=ctx.n_bo_trials, seed=ctx.seed,
        )

        # Pipeline cache keyed by whichever params actually affect the
        # vstore identity. Stages that don't change the vstore return
        # the same ``cache_key()`` every time -> single pipeline reused.
        pipeline_cache: dict[tuple, RAGPipeline] = {}
        trials_out: list[StageTrial] = []

        # --- Replay completed trials from the checkpoint, if any ----
        # ``BayesianSearch.report()`` requires a prior ``next_params()``
        # to populate ``_pending_params``. For replay we already know
        # both the params and the score, so we bypass the guard by
        # priming ``_pending_params`` directly. Equivalent to telling the
        # sampler "I'm continuing trial N with params X and score S",
        # which is exactly what resume needs.
        checkpoint = getattr(ctx, "checkpoint", None)
        checkpoint_store = getattr(ctx, "checkpoint_store", None)
        if checkpoint is not None and checkpoint.trials_completed:
            for ct in checkpoint.trials_completed:
                bo._pending_params = dict(ct.params)
                bo.report(dict(ct.params), float(ct.composite_score))
                trials_out.append(StageTrial(
                    trial_index=ct.trial_index,
                    params=dict(ct.params),
                    n_chunks=ct.n_chunks,
                    scores=dict(ct.scores),
                    composite_score=ct.composite_score,
                    feasible=ct.feasible,
                    violations=list(ct.violations),
                    answers_preview=list(ct.answers_preview),
                    records=list(ct.records),
                    elapsed_seconds=ct.elapsed_seconds,
                ))

        while bo.has_next():
            params = bo.next_params()
            t_start = time.time()

            trial_config = self.build_trial_config(ctx.base_config, params)
            key = self.cache_key(params)
            if key not in pipeline_cache:
                chunks = self.chunks_for_trial(ctx, trial_config, params)
                pipeline_cache[key] = RAGPipeline.build(
                    trial_config,
                    chunks,
                    embedder=ctx.runtime.embeddings,
                    llm_callable=ctx.runtime.llm_callable,
                )
            pipeline = pipeline_cache[key]
            n_chunks = pipeline.n_chunks

            # Per-call overrides let stages vary parameters that don't
            # change pipeline identity (e.g. RetrievalOptimizer's top_k).
            overrides = self.per_call_overrides(params)
            answered: list[DatasetRecord] = []
            for r in ctx.records:
                ctxs = pipeline.retrieve(
                    r.question, top_k=overrides.get("top_k"),
                )
                ans = pipeline.generate(
                    r.question, ctxs,
                    system_instruction=overrides.get("system_instruction"),
                )
                answered.append(
                    DatasetRecord(
                        question=r.question,
                        ground_truth=r.ground_truth,
                        contexts=ctxs,
                        answer=ans,
                    )
                )

            ev = _evaluate_with_ragas(
                answered,
                ctx.runtime.ragas_judge,
                ctx.runtime.ragas_embeddings,
                ctx.metrics,
                run_config=ctx.runtime.ragas_run_config,
            )
            scores = ev.get("scores", {})
            comp_eval = ctx.objective.evaluate(scores)
            bo.report(params, comp_eval["score"])

            new_trial = StageTrial(
                trial_index=len(bo.trials) - 1,
                params=dict(params),
                n_chunks=n_chunks,
                scores=scores,
                composite_score=comp_eval["score"],
                feasible=comp_eval["feasible"],
                violations=comp_eval["violations"],
                answers_preview=[(a.answer or "")[:200] for a in answered],
                records=[
                    {
                        "question": a.question,
                        "ground_truth": a.ground_truth,
                        "contexts": list(a.contexts or []),
                        "answer": a.answer or "",
                    }
                    for a in answered
                ],
                elapsed_seconds=round(time.time() - t_start, 2),
            )
            trials_out.append(new_trial)

            # Persist after every completed trial so a crash mid-loop
            # loses at most this trial. Defensive: a checkpoint write
            # failure must never sink the whole optimization.
            if checkpoint is not None and checkpoint_store is not None:
                try:
                    checkpoint.trials_completed.append(CompletedTrial(
                        trial_index=new_trial.trial_index,
                        params=new_trial.params,
                        n_chunks=new_trial.n_chunks,
                        scores=new_trial.scores,
                        composite_score=new_trial.composite_score,
                        feasible=new_trial.feasible,
                        violations=new_trial.violations,
                        answers_preview=new_trial.answers_preview,
                        records=new_trial.records,
                        elapsed_seconds=new_trial.elapsed_seconds,
                    ))
                    checkpoint.search_space = dict(space)
                    checkpoint.n_bo_init = ctx.n_bo_init
                    checkpoint.max_trials = ctx.n_bo_trials
                    checkpoint.seed = ctx.seed
                    checkpoint.objective_spec = ctx.objective.to_dict()
                    checkpoint.stage_label = ctx.label
                    checkpoint_store.save(checkpoint)
                except Exception as exc:  # pragma: no cover - defensive
                    import logging
                    logging.getLogger(__name__).warning(
                        "checkpoint save failed (continuing optimization): %s",
                        exc,
                    )

        best = bo.best_trial
        best_config = (
            self.build_trial_config(ctx.base_config, best.params) if best else None
        )
        return StageResult(
            stage_name=self.name,
            search_space=space,
            trials=trials_out,
            best_params=best.params if best else None,
            best_config=best_config,
            best_composite=best.score if best else None,
            objective_spec=ctx.objective.to_dict(),
            n_init=ctx.n_bo_init,
            max_trials=ctx.n_bo_trials,
        )


__all__ = [
    "StageContext",
    "StageOptimizer",
    "StageResult",
    "StageTrial",
]
