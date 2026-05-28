"""End-to-end experiment driver for the ragdx pipeline.

``run_experiment`` is the importable equivalent of the demo scripts in
``examples/``: one function call that takes a corpus + a few flags,
runs a Bayesian RAG-config search, runs DSPy before/after for each
requested GT mode, and returns a bundle you can save / inspect / feed
into the dashboards.

Bundle schema
-------------

The bundle output uses ``schema_version: 1`` — a stable, mode-keyed
shape that the generic dashboard renders directly. See
:func:`_build_unified_bundle` for the layout. Legacy bundles produced
by the older demo scripts can be normalized via
:func:`migrate_legacy_bundle`.

Usage::

    from ragdx import run_experiment

    # with-GT (amnesty_qa demo equivalent) -- run both modes side-by-side
    result = run_experiment(
        corpus="explodinggradients/amnesty_qa",
        has_gt=True,
        mode="both",
        api_key="<glm-key>",
    )

    # no-GT (PDF demo equivalent)
    result = run_experiment(
        corpus="<path/to/your.pdf>",
        has_gt=False,
        api_key="<glm-key>",
    )

    print(result.bundle["bayes_search"]["with_gt"]["best_params"])

The bundle is also written to ``output_dir/result.json`` so
``ragdx.ui.experiment_dashboard`` (the generic dashboard) can render it
directly.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ragdx.core.evaluator import UnifiedEvaluator
from ragdx.datasets import synthesize_questions
from ragdx.loaders import load_pdf_chunks
from ragdx.optim._gt_helpers import gt_mode as detect_gt_mode
from ragdx.optim.bayes_search import BayesianSearch
from ragdx.optim.dspy_adapter import DSPyAdapter
from ragdx.optim.objectives import CompositeObjective, default_objective
from ragdx.schemas.models import DatasetRecord

ExperimentMode = Literal["with_gt", "no_gt", "both", "auto"]
logger = logging.getLogger(__name__)


# =====================================================================
# Public dataclasses
# =====================================================================
@dataclass
class ExperimentConfig:
    """Configuration for :func:`run_experiment`.

    Fields are deliberately tuned so the default values reproduce the
    PDF / amnesty_qa demos when fed the matching ``corpus`` argument.
    """

    corpus: str | Path
    has_gt: bool
    mode: ExperimentMode = "auto"
    questions_path: str | Path | None = None
    n_questions: int = 5
    n_bo_trials: int = 8
    n_bo_init: int = 3
    top_ks: list[int] = field(default_factory=lambda: [1, 3, 5, 7])
    chunk_sizes: list[int] = field(default_factory=lambda: [256, 512, 1024])
    chunk_overlaps: list[int] = field(default_factory=lambda: [0, 50, 100])
    objective_overrides: dict[str, CompositeObjective] | None = None
    output_dir: str | Path = ".ragdx_experiment"
    api_key: str | None = None
    api_base: str = "https://open.bigmodel.cn/api/paas/v4"
    model: str = "openai/glm-4-flash"
    seed: int = 7
    # --- LLM endpoint call management (applies to every LLM-talking
    # subsystem in the pipeline -- ragas judge, DSPy MIPROv2, BO
    # generation). Two knobs, named for *what they control* (an LLM
    # endpoint's rate budget) rather than which library consumes them.
    llm_max_concurrent: int = 2
    """Max in-flight LLM calls per evaluation batch. Default ``2`` is
    tuned for strict rate-limited endpoints (e.g. GLM-4-Flash). Bump
    to ``8-16`` for OpenAI / Anthropic to speed up trials. Propagates
    to ragas ``RunConfig.max_workers`` and DSPy MIPROv2 ``num_threads``."""

    llm_max_retries: int = 5
    """Per-call transport-layer retry budget. Propagates to the openai
    client (used by ragas judge) and ``litellm.completion`` (used by
    BO generation and DSPy). Default ``5`` should survive most transient
    rate-limit windows; raise if you see frequent ``NaN`` scores."""

    def __post_init__(self) -> None:
        # mode validation -- can't request with-GT runs when no GT exists.
        if not self.has_gt and self.mode in ("with_gt", "both"):
            raise ValueError(
                "mode='with_gt' / 'both' requires has_gt=True; received "
                f"has_gt=False, mode={self.mode!r}. Set mode='no_gt' or "
                "provide GT data."
            )
        if self.mode == "auto":
            self.mode = "both" if self.has_gt else "no_gt"
        if self.api_key is None:
            self.api_key = os.environ.get("ZHIPU_API_KEY") or os.environ.get(
                "OPENAI_API_KEY"
            )
        if not self.api_key:
            raise ValueError(
                "api_key is required (or set ZHIPU_API_KEY / OPENAI_API_KEY "
                "in the environment)."
            )
        self.output_dir = Path(self.output_dir)


@dataclass
class ExperimentResult:
    """Output of :func:`run_experiment`."""

    config: ExperimentConfig
    bundle: dict
    output_path: Path

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the bundle JSON. Returns the actual write path."""
        target = Path(path) if path else self.output_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.bundle, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return target


# =====================================================================
# DSPy MIPROv2 trial-score capture (matches the demos)
# =====================================================================
class _MIPROTrialScoreCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.scores_so_far: list[float] = []
        self.best_scores: list[float] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "Scores so far:" in msg:
            tail = msg.split("Scores so far:", 1)[1].strip()
            try:
                self.scores_so_far = json.loads(tail)
            except Exception:
                pass
        elif "Best score so far:" in msg:
            tail = msg.split("Best score so far:", 1)[1].strip()
            try:
                self.best_scores.append(float(tail))
            except Exception:
                pass


# =====================================================================
# Runtime helpers (LLM, judge, embeddings) -- one place per provider
# =====================================================================
@dataclass
class _Runtime:
    """Holds the live LLM clients shared across all experiment stages."""

    llm_callable: Callable[[str], str]
    dspy_lm: Any
    ragas_judge: Any
    ragas_embeddings: Any
    embeddings: Any  # langchain embeddings for FAISS
    # Throttle knobs propagated from ExperimentConfig so downstream call
    # sites don't need to re-thread cfg.
    llm_max_concurrent: int = 2
    llm_max_retries: int = 5
    ragas_run_config: Any = None  # built from llm_max_concurrent (see _build_runtime)


def _apply_litellm_temperature_clamp(min_temp: float = 0.01) -> None:
    """GLM-4-Flash rejects temperature<0.01. DSPy / ragas internally
    override the temperature to ~1e-8 for deterministic output; we
    clamp at *every* upstream entry point so every downstream call is
    safe regardless of which library makes it.

    Clamps three layers:

    1. ``litellm.completion`` / ``batch_completion`` — used by DSPy and
       our own ``llm_callable``.
    2. ``openai.resources.chat.completions.Completions.create`` —
       sync OpenAI client (covers ``langchain_openai.ChatOpenAI``'s
       sync path).
    3. ``openai.resources.chat.completions.AsyncCompletions.create`` —
       async OpenAI client (ragas' judge calls hit this).

    Idempotent on each layer."""
    import litellm

    def _clamp(kw):
        t = kw.get("temperature")
        if t is not None and 0 < t < min_temp:
            kw["temperature"] = min_temp
        return kw

    # ---- litellm layer
    if not getattr(litellm.completion, "_ragdx_clamped", False):
        orig_completion = litellm.completion
        orig_batch = litellm.batch_completion

        def comp(*a, **kw):
            return orig_completion(*a, **_clamp(kw))

        def batch(*a, **kw):
            return orig_batch(*a, **_clamp(kw))

        comp._ragdx_clamped = True  # type: ignore[attr-defined]
        litellm.completion = comp
        litellm.batch_completion = batch

    # ---- openai client layer (covers ChatOpenAI -> ragas judge)
    try:
        from openai.resources.chat.completions import (
            AsyncCompletions,
            Completions,
        )
    except ImportError:  # pragma: no cover - openai pinning floor
        return

    if not getattr(Completions.create, "_ragdx_clamped", False):
        _orig_sync = Completions.create

        def _sync_create(self, *a, **kw):
            return _orig_sync(self, *a, **_clamp(kw))

        _sync_create._ragdx_clamped = True  # type: ignore[attr-defined]
        Completions.create = _sync_create  # type: ignore[method-assign]

    if not getattr(AsyncCompletions.create, "_ragdx_clamped", False):
        _orig_async = AsyncCompletions.create

        async def _async_create(self, *a, **kw):
            return await _orig_async(self, *a, **_clamp(kw))

        _async_create._ragdx_clamped = True  # type: ignore[attr-defined]
        AsyncCompletions.create = _async_create  # type: ignore[method-assign]


def _build_runtime(cfg: ExperimentConfig) -> _Runtime:
    """Construct every piece of LLM infrastructure once."""
    # Ensure env vars for downstream libs that read them directly.
    os.environ["OPENAI_API_KEY"] = cfg.api_key  # type: ignore[assignment]
    os.environ["OPENAI_API_BASE"] = cfg.api_base
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    _apply_litellm_temperature_clamp()

    import litellm

    def llm_callable(prompt: str) -> str:
        resp = litellm.completion(
            model=cfg.model,
            messages=[{"role": "user", "content": prompt}],
            api_key=cfg.api_key,
            api_base=cfg.api_base,
            temperature=0.01,
            max_tokens=350,
            timeout=60,
            num_retries=cfg.llm_max_retries,
        )
        return resp.choices[0].message.content or ""

    import dspy

    dspy_lm = dspy.LM(
        cfg.model,
        api_key=cfg.api_key,
        api_base=cfg.api_base,
        temperature=0.01,
        max_tokens=400,
        cache=False,
    )

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    # Strip "openai/" prefix that LiteLLM uses for ChatOpenAI's model arg.
    chat_model = cfg.model.split("/", 1)[-1] if "/" in cfg.model else cfg.model
    ragas_judge = LangchainLLMWrapper(
        ChatOpenAI(
            model=chat_model,
            api_key=cfg.api_key,
            base_url=cfg.api_base,
            temperature=0.01,
            timeout=60,
            max_retries=cfg.llm_max_retries,
        )
    )

    # ragas RunConfig: only ``max_workers`` is user-tunable (via the
    # ``llm_max_concurrent`` knob). The other fields (max_retries /
    # max_wait / timeout) are sensible internal defaults -- they're
    # ragas-loop-specific knobs, distinct from the transport-layer
    # retry budget (``llm_max_retries`` above) that actually matters
    # for production tuning.
    ragas_run_config = None
    try:
        from ragas.run_config import RunConfig as _RagasRunConfig
        ragas_run_config = _RagasRunConfig(
            max_workers=cfg.llm_max_concurrent,
            max_retries=8,
            max_wait=30,
            timeout=180,
        )
    except ImportError:
        pass

    return _Runtime(
        llm_callable=llm_callable,
        dspy_lm=dspy_lm,
        ragas_judge=ragas_judge,
        ragas_embeddings=ragas_embeddings,
        embeddings=hf_embeddings,
        llm_max_concurrent=cfg.llm_max_concurrent,
        llm_max_retries=cfg.llm_max_retries,
        ragas_run_config=ragas_run_config,
    )


# =====================================================================
# Corpus + records loading
# =====================================================================
def _looks_like_hf_dataset(s: str) -> bool:
    """``user/repo`` strings with no path separators are likely HF dataset names."""
    if not isinstance(s, str):
        return False
    if Path(s).exists():
        return False
    return "/" in s and "\\" not in s and not s.endswith((".pdf", ".jsonl", ".txt"))


def _load_amnesty_style_hf(name: str, n_records: int) -> tuple[list[str], list[DatasetRecord]]:
    """Load a HuggingFace QA-style dataset (amnesty_qa-shaped).

    Tries amnesty_qa's known config/split combos in order
    (``english_v3`` → ``v2`` → ``v1``) and reads the most likely
    column names for question / GT / contexts. Different config
    versions use different column names — see code for the mapping.
    Falls back to a no-config ``train`` split for HF datasets that
    don't follow the amnesty_qa convention at all.
    """
    from datasets import load_dataset

    ds = None
    for cfg in ("english_v3", "english_v2", "english_v1"):
        try:
            ds = load_dataset(name, cfg, split=f"eval[:{n_records}]")
            break
        except Exception:  # pragma: no cover - schema drift across versions
            continue
    if ds is None:
        ds = load_dataset(name, split=f"train[:{n_records}]")

    def _col(row: dict, *names: str) -> Any:
        for n in names:
            if n in row and row[n] is not None:
                return row[n]
        return None

    corpus_chunks: list[str] = []
    records: list[DatasetRecord] = []
    for row in ds:
        # v3 uses retrieved_contexts; v1/v2 use contexts.
        for passage in _col(row, "retrieved_contexts", "contexts") or []:
            text = (passage or "").strip()
            if text:
                corpus_chunks.append(text)
        # v3 uses reference; v2 uses ground_truth (str); v1 uses ground_truths (list[str]).
        gt = _col(row, "reference", "ground_truth")
        if gt is None:
            gt_list = _col(row, "ground_truths") or []
            gt = gt_list[0] if gt_list else ""
        records.append(
            DatasetRecord(
                question=_col(row, "user_input", "question") or "",
                ground_truth=gt or "",
                contexts=[],
            )
        )
    return corpus_chunks, records


def _load_jsonl_questions(path: Path) -> list[DatasetRecord]:
    """``[{question, ground_truth?, contexts?}, ...]`` -> DatasetRecords."""
    records: list[DatasetRecord] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append(
                DatasetRecord(
                    question=row["question"],
                    ground_truth=row.get("ground_truth"),
                    contexts=list(row.get("contexts") or []),
                )
            )
    return records


def _load_corpus_and_records(
    cfg: ExperimentConfig, runtime: _Runtime
) -> tuple[list[str], list[DatasetRecord], dict]:
    """Dispatch on the ``corpus`` argument and produce
    ``(chunks, records, source_meta)``. Records may have empty GT in
    no-GT mode -- that's intentional and downstream handles it."""
    corpus = cfg.corpus
    source_meta: dict[str, Any] = {"corpus": str(corpus)}

    # ---- HuggingFace dataset (single source for both corpus + records) ----
    if isinstance(corpus, str) and _looks_like_hf_dataset(corpus):
        chunks, records = _load_amnesty_style_hf(corpus, cfg.n_questions)
        source_meta["kind"] = "huggingface"
        source_meta["dataset"] = corpus
        source_meta["records"] = len(records)
        source_meta["corpus_chunks"] = len(chunks)
        return chunks, records, source_meta

    # ---- PDF corpus -- need to synthesize questions or load externally ----
    path = Path(corpus)
    if path.suffix.lower() == ".pdf":
        loader = load_pdf_chunks(path, chunk_size=512, chunk_overlap=50)
        chunks = loader.chunks
        source_meta["kind"] = "pdf"
        source_meta["pdf_meta"] = loader.metadata

        if cfg.questions_path:
            records = _load_jsonl_questions(Path(cfg.questions_path))
            source_meta["questions_path"] = str(cfg.questions_path)
        elif cfg.has_gt:
            raise ValueError(
                "PDF corpus + has_gt=True requires `questions_path` to point "
                "at a JSONL with {question, ground_truth, ...} per line."
            )
        else:
            # No GT mode -- synthesize questions from the corpus.
            synthesised = synthesize_questions(
                chunks,
                n=cfg.n_questions,
                llm_callable=runtime.llm_callable,
                chunks_per_question=2,
            )
            records = [
                DatasetRecord(question=q.question, ground_truth=None, contexts=[])
                for q in synthesised
            ]
            source_meta["synthesized_meta"] = [
                {"question": q.question, "source_chunk_ids": q.source_chunk_ids}
                for q in synthesised
            ]
        return chunks, records, source_meta

    # ---- JSONL corpus path: {text, source} per line ----
    if path.suffix.lower() == ".jsonl":
        chunks = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get("text") or row.get("chunk") or row.get("content")
                if text:
                    chunks.append(text.strip())
        source_meta["kind"] = "jsonl"
        source_meta["corpus_chunks"] = len(chunks)

        if cfg.questions_path:
            records = _load_jsonl_questions(Path(cfg.questions_path))
        elif cfg.has_gt:
            raise ValueError(
                "JSONL corpus + has_gt=True requires `questions_path` for the "
                "question / ground_truth set."
            )
        else:
            synthesised = synthesize_questions(
                chunks,
                n=cfg.n_questions,
                llm_callable=runtime.llm_callable,
                chunks_per_question=2,
            )
            records = [
                DatasetRecord(question=q.question, ground_truth=None, contexts=[])
                for q in synthesised
            ]
        return chunks, records, source_meta

    raise ValueError(
        f"Unsupported corpus: {corpus!r}. Accepts: HuggingFace dataset name "
        "(e.g. 'org/dataset'), .pdf path, or .jsonl path."
    )


# =====================================================================
# RAG primitives (mirrors what the demos do)
# =====================================================================
def _build_vstore(chunks: list[str], embeddings: Any) -> Any:
    from langchain_community.vectorstores import FAISS

    return FAISS.from_texts(chunks, embeddings)


def _retrieve(vstore: Any, question: str, top_k: int) -> list[str]:
    return [d.page_content for d in vstore.similarity_search(question, k=top_k)]


def _generate_answer(question: str, contexts: list[str], lm: Callable[[str], str]) -> str:
    ctx_str = "\n---\n".join(contexts) if contexts else "(no context)"
    prompt = (
        "Answer the question using only the provided context.\n"
        "Be concise. Do not invent facts not in the context.\n\n"
        f"Context:\n{ctx_str}\n\nQuestion: {question}\n\nAnswer:"
    )
    try:
        return lm(prompt)
    except Exception as e:  # pragma: no cover - live LLM
        return f"<generation error: {e}>"


def _build_ragas_metrics_for_mode(mode: str) -> list:
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    if mode == "with_gt":
        return [faithfulness, answer_relevancy, context_precision, context_recall]
    return [faithfulness, answer_relevancy, context_precision]


def _evaluate_with_ragas(
    records: list[DatasetRecord],
    judge: Any,
    embeddings: Any,
    metrics: list,
    *,
    run_config: Any = None,
) -> dict:
    """Run ragas evaluation with an optional ``RunConfig`` throttle.

    ``run_config`` is forwarded verbatim to ``ragas.evaluate``. Pass
    the one built in :func:`_build_runtime` (which honours
    :class:`ExperimentConfig`'s ``ragas_*`` knobs) to throttle
    concurrency on strict rate-limit endpoints; pass ``None`` to let
    ragas use its own defaults (``max_workers=16``).
    """
    evaluator = UnifiedEvaluator()
    try:
        ragas_kwargs: dict[str, Any] = {
            "llm": judge, "embeddings": embeddings, "metrics": metrics,
        }
        if run_config is not None:
            ragas_kwargs["run_config"] = run_config
        result = evaluator.evaluate(
            records,
            use_ragas=True, run_ragas=True,
            use_ragchecker=False, use_embedding=False,
            ragas_kwargs=ragas_kwargs,
        )
        return {
            "scores": {**result.retrieval, **result.generation, **result.e2e},
            "skipped": result.metadata.get("skipped_metrics", {}),
        }
    except Exception as e:  # pragma: no cover - live LLM
        return {"error": f"{type(e).__name__}: {e}", "scores": {}}


# =====================================================================
# AutoRAG BO + DSPy A/B (one helper each, called once per GT mode)
# =====================================================================
def _run_bayes_search(
    cfg: ExperimentConfig,
    runtime: _Runtime,
    chunks_master: list[str],
    records: list[DatasetRecord],
    objective: CompositeObjective,
    metrics: list,
    label: str,
) -> dict:
    search_space = {
        "chunk_size": list(cfg.chunk_sizes),
        "chunk_overlap": list(cfg.chunk_overlaps),
        "top_k": list(cfg.top_ks),
    }
    bo = BayesianSearch(
        search_space, n_init=cfg.n_bo_init, max_trials=cfg.n_bo_trials, seed=cfg.seed,
    )

    vstore_cache: dict[tuple, tuple[int, Any]] = {}
    trials_out: list[dict] = []
    while bo.has_next():
        params = bo.next_params()
        t_start = time.time()
        logger.info("[BO/%s] trial %d %s", label, len(bo.trials) + 1, params)

        cache_key = (params["chunk_size"], params["chunk_overlap"])
        if cache_key not in vstore_cache:
            # Re-chunk the master corpus to this size/overlap. For HF / jsonl
            # sources we already have one fixed chunk list, but BO still wants
            # to vary -- we re-split the joined text. For PDF we re-load.
            if isinstance(cfg.corpus, (str, Path)) and Path(cfg.corpus).exists() and Path(cfg.corpus).suffix.lower() == ".pdf":
                loader = load_pdf_chunks(
                    Path(cfg.corpus),
                    chunk_size=params["chunk_size"],
                    chunk_overlap=params["chunk_overlap"],
                )
                chunks = loader.chunks
            else:
                # For non-PDF sources just use the master chunks as-is (chunk_size
                # variation is only meaningful when we can re-split the raw text).
                chunks = chunks_master
            vstore = _build_vstore(chunks, runtime.embeddings)
            vstore_cache[cache_key] = (len(chunks), vstore)
        n_chunks, vstore = vstore_cache[cache_key]

        answered = []
        for r in records:
            ctxs = _retrieve(vstore, r.question, params["top_k"])
            ans = _generate_answer(r.question, ctxs, runtime.llm_callable)
            answered.append(
                DatasetRecord(
                    question=r.question,
                    ground_truth=r.ground_truth,
                    contexts=ctxs,
                    answer=ans,
                )
            )

        ev = _evaluate_with_ragas(
            answered, runtime.ragas_judge, runtime.ragas_embeddings, metrics,
            run_config=runtime.ragas_run_config,
        )
        scores = ev.get("scores", {})
        comp_eval = objective.evaluate(scores)
        bo.report(params, comp_eval["score"])

        trials_out.append(
            {
                "trial_index": len(bo.trials) - 1,
                "params": params,
                "n_chunks": n_chunks,
                "scores": scores,
                "composite_score": comp_eval["score"],
                "feasible": comp_eval["feasible"],
                "violations": comp_eval["violations"],
                "answers_preview": [a.answer[:200] for a in answered],
                "elapsed_seconds": round(time.time() - t_start, 2),
            }
        )

    best = bo.best_trial
    return {
        "search_space": search_space,
        "n_init": cfg.n_bo_init,
        "max_trials": cfg.n_bo_trials,
        "objective_spec": objective.to_dict(),
        "trials": trials_out,
        "best_params": best.params if best else None,
        "best_composite": best.score if best else None,
    }


def _dspy_before_after(
    runtime: _Runtime,
    records_with_ctxs: list[DatasetRecord],
    objective: CompositeObjective,
    metrics: list,
    label: str,
) -> dict:
    import dspy

    dspy.configure(lm=runtime.dspy_lm)
    adapter = DSPyAdapter()
    baseline_program = adapter.build_program()

    def _run_program(program):
        out = []
        for r in records_with_ctxs:
            ctx_str = "\n".join(r.contexts) if r.contexts else ""
            try:
                with dspy.context(lm=runtime.dspy_lm):
                    pred = program(question=r.question, context=ctx_str)
                ans = str(getattr(pred, "answer", "") or "")
            except Exception as e:  # pragma: no cover - live LLM
                ans = f"<error: {e}>"
            out.append(
                DatasetRecord(
                    question=r.question,
                    ground_truth=r.ground_truth,
                    contexts=r.contexts,
                    answer=ans,
                )
            )
        return out

    logger.info("[DSPy/%s] (a) baseline run", label)
    baseline_answered = _run_program(baseline_program)
    baseline_eval = _evaluate_with_ragas(
        baseline_answered, runtime.ragas_judge, runtime.ragas_embeddings, metrics,
        run_config=runtime.ragas_run_config,
    )

    logger.info("[DSPy/%s] (b) MIPROv2 optimisation", label)
    capture = _MIPROTrialScoreCapture()
    capture.setLevel(logging.INFO)
    mipro_logger = logging.getLogger("dspy.teleprompt.mipro_optimizer_v2")
    mipro_logger.addHandler(capture)
    prior = mipro_logger.level
    mipro_logger.setLevel(logging.INFO)
    try:
        opt_result = adapter.optimize(
            records_with_ctxs,
            student_lm=runtime.dspy_lm,
            judge_lm=runtime.dspy_lm,
            optimizer="MIPROv2",
            optimizer_kwargs={
                "auto": "light",
                # Match the same in-flight budget ragas uses, so we don't
                # accidentally fan out 16 DSPy threads against a strict
                # endpoint after carefully throttling ragas to 2.
                "num_threads": runtime.llm_max_concurrent,
            },
        )
    finally:
        mipro_logger.removeHandler(capture)
        mipro_logger.setLevel(prior)

    logger.info("[DSPy/%s] (c) optimised re-run", label)
    optimised_answered = _run_program(opt_result["optimized_program"])
    opt_eval = _evaluate_with_ragas(
        optimised_answered, runtime.ragas_judge, runtime.ragas_embeddings, metrics,
        run_config=runtime.ragas_run_config,
    )

    baseline_scores = baseline_eval.get("scores", {}) or {}
    optimised_scores = opt_eval.get("scores", {}) or {}
    composite_baseline = objective.evaluate(baseline_scores)
    composite_optimized = objective.evaluate(optimised_scores)
    delta = {
        m: (optimised_scores.get(m, float("nan")) - baseline_scores.get(m, float("nan")))
        for m in sorted(set(baseline_scores) | set(optimised_scores))
    }
    return {
        "baseline_scores": baseline_scores,
        "optimized_scores": optimised_scores,
        "delta": delta,
        "composite": {
            "objective_spec": objective.to_dict(),
            "baseline": composite_baseline,
            "optimized": composite_optimized,
            "delta": composite_optimized["score"] - composite_baseline["score"],
        },
        "baseline_sample_answers": [a.answer for a in baseline_answered],
        "optimized_sample_answers": [a.answer for a in optimised_answered],
        "instructions": dict(opt_result["instructions"]),
        "demos": {n: [dict(d) for d in demos] for n, demos in opt_result["demos"].items()},
        "trial_scores": list(capture.scores_so_far),
        "best_score_progression": list(capture.best_scores),
        "gt_mode": opt_result["gt_mode"],
        "optimizer": opt_result["optimizer"],
        "trainset_size": opt_result["trainset_size"],
    }


# =====================================================================
# Per-mode orchestration
# =====================================================================
def _run_one_mode(
    cfg: ExperimentConfig,
    runtime: _Runtime,
    chunks_master: list[str],
    records: list[DatasetRecord],
    mode: str,
) -> dict:
    """Drive one GT mode through AutoRAG BO + DSPy A/B."""
    objective = (cfg.objective_overrides or {}).get(mode) or default_objective(mode)
    metrics = _build_ragas_metrics_for_mode(mode)

    bo_result = _run_bayes_search(cfg, runtime, chunks_master, records, objective, metrics, mode)

    # Build records pre-retrieved at the BO winner's top_k + chunk size.
    best_params = bo_result["best_params"] or {}
    if isinstance(cfg.corpus, (str, Path)) and Path(str(cfg.corpus)).exists() and Path(str(cfg.corpus)).suffix.lower() == ".pdf":
        chunks_for_dspy = load_pdf_chunks(
            Path(cfg.corpus),
            chunk_size=best_params.get("chunk_size", 512),
            chunk_overlap=best_params.get("chunk_overlap", 50),
        ).chunks
    else:
        chunks_for_dspy = chunks_master
    vstore = _build_vstore(chunks_for_dspy, runtime.embeddings)

    records_dspy = [
        DatasetRecord(
            question=r.question,
            ground_truth=r.ground_truth,
            contexts=_retrieve(vstore, r.question, best_params.get("top_k", 3)),
        )
        for r in records
    ]
    dspy_result = _dspy_before_after(runtime, records_dspy, objective, metrics, mode)

    return {
        "bayes_search": bo_result,
        "dspy_a_b": dspy_result,
        "objective_spec": objective.to_dict(),
    }


# =====================================================================
# Unified bundle (schema_version: 1)
# =====================================================================
SCHEMA_VERSION = 1


def _build_unified_bundle(
    cfg: ExperimentConfig,
    results_by_mode: dict[str, dict],
    source_meta: dict,
    base_records: list[DatasetRecord],
) -> dict:
    """Assemble the canonical ``schema_version: 1`` bundle.

    Layout (every mode-keyed section is *always* a dict, so dashboards
    never have to disambiguate single-mode vs multi-mode runs)::

        {
          "schema_version": 1,
          "meta": {
            "model", "model_endpoint",
            "experiment_mode",            # the resolved cfg.mode
            "modes_run": [<mode>, ...],
            "has_gt", "detected_gt_mode",
            "source": {kind, corpus, ...} # corpus-source descriptor
          },
          "questions": [{"question", "ground_truth"}, ...],
          "data_diagnostics": {<mode>: {has_ground_truth, gt_mode, record_count}},
          "objectives":     {<mode>: <objective spec>},
          "bayes_search":   {<mode>: <BO result>},
          "dspy_a_b":       {<mode>: <DSPy before/after result>},
          "extras": {
            "pdf_meta"?:               # only when corpus is a PDF
            "synthesized_questions"?:  # only when questions were LLM-synthesized
            ...
          }
        }
    """
    detected = detect_gt_mode(base_records)
    modes_run = list(results_by_mode.keys())

    # Pull source-specific extras out so meta.source stays a clean
    # descriptor and the bulky bits live under bundle.extras.
    extras: dict[str, Any] = {}
    if source_meta.get("synthesized_meta"):
        extras["synthesized_questions"] = source_meta["synthesized_meta"]
    if source_meta.get("pdf_meta"):
        extras["pdf_meta"] = source_meta["pdf_meta"]

    source_clean = {
        k: v for k, v in source_meta.items() if k not in {"synthesized_meta"}
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "model": cfg.model,
            "model_endpoint": cfg.api_base,
            "experiment_mode": cfg.mode,
            "modes_run": modes_run,
            "has_gt": cfg.has_gt,
            "detected_gt_mode": detected,
            "source": source_clean,
        },
        "questions": [
            {"question": r.question, "ground_truth": r.ground_truth}
            for r in base_records
        ],
        "data_diagnostics": {
            m: {
                "has_ground_truth": (m == "with_gt"),
                "gt_mode": m,
                "record_count": len(base_records),
            }
            for m in results_by_mode
        },
        "objectives": {m: r["objective_spec"] for m, r in results_by_mode.items()},
        "bayes_search": {m: r["bayes_search"] for m, r in results_by_mode.items()},
        "dspy_a_b": {m: r["dspy_a_b"] for m, r in results_by_mode.items()},
        "extras": extras,
    }


def migrate_legacy_bundle(bundle: dict) -> dict:
    """Best-effort upgrade of a pre-``schema_version: 1`` bundle.

    Recognised legacy shapes:

    * Multi-mode amnesty/grid demo (``optimize_gt_modes_result.json``)
      with top-level ``autorag_grid``, ``autorag_spec``, ``metric_map``,
      ``data_diagnostics``, ``dspy_before_after`` keyed by mode.
    * Single-mode PDF/no-GT demo (``pdf_no_gt_result.json``) with flat
      ``autorag_bo`` / ``dspy_before_after`` and a top-level ``gt_mode``.

    Unrecognised bundles pass through unchanged with a ``schema_version``
    tag of ``0`` so callers can detect the situation. Already-v1 bundles
    are returned untouched.
    """
    if bundle.get("schema_version") == SCHEMA_VERSION:
        return bundle

    # ------ shape A: multi-mode grid (optimize_gt_modes demo) -------------
    if "autorag_grid" in bundle and isinstance(bundle["autorag_grid"], dict):
        grid = bundle["autorag_grid"]
        dspy = bundle.get("dspy_before_after") or {}
        modes_run = [m for m in ("with_gt", "no_gt") if m in grid]

        bayes_search: dict[str, dict] = {}
        objectives: dict[str, dict] = {}
        for m, payload in grid.items():
            runs = payload.get("runs") or []
            trials = []
            for i, run in enumerate(runs):
                trials.append({
                    "trial_index": i,
                    "params": {"top_k": run.get("top_k")},
                    "n_chunks": run.get("n_chunks"),
                    "scores": run.get("scores", {}),
                    "composite_score": run.get("composite_score"),
                    "feasible": run.get("feasible", True),
                    "violations": run.get("violations", []),
                    "answers_preview": [
                        (a or "")[:200] for a in (run.get("answers") or [])
                    ],
                })
            bayes_search[m] = {
                "search_space": {"top_k": sorted({r.get("top_k") for r in runs if r.get("top_k") is not None})},
                "trials": trials,
                "best_params": {"top_k": payload.get("best_top_k")} if payload.get("best_top_k") is not None else None,
                "best_composite": payload.get("best_composite"),
                "objective_spec": payload.get("objective_spec"),
                "_legacy_kind": "grid",  # marker so dashboards can label this honestly
            }
            objectives[m] = payload.get("objective_spec") or {
                "metrics": {payload.get("objective"): 1.0} if payload.get("objective") else {},
                "constraints": {},
                "mode": "weighted_sum",
            }

        dspy_v1 = {m: dspy[m] for m in modes_run if m in dspy}

        diagnostics = bundle.get("data_diagnostics") or {
            m: {"has_ground_truth": (m == "with_gt"), "gt_mode": m,
                "record_count": bundle.get("corpus_size", 0)}
            for m in modes_run
        }

        source: dict[str, Any] = {"kind": "huggingface"}
        if "dataset" in bundle:
            source["dataset"] = bundle["dataset"]
            source["corpus"] = bundle["dataset"]
        if "corpus_chunks" in bundle:
            source["corpus_chunks"] = bundle["corpus_chunks"]
        if "corpus_size" in bundle:
            source["records"] = bundle["corpus_size"]

        out = {
            "schema_version": SCHEMA_VERSION,
            "meta": {
                "model": bundle.get("model"),
                "model_endpoint": bundle.get("model_endpoint"),
                "experiment_mode": "both" if len(modes_run) > 1 else (modes_run[0] if modes_run else "unknown"),
                "modes_run": modes_run,
                "has_gt": "with_gt" in modes_run,
                "detected_gt_mode": "with_gt" if "with_gt" in modes_run else "no_gt",
                "source": source,
                "_migrated_from": "legacy_grid_v0",
            },
            "questions": bundle.get("questions", []),
            "data_diagnostics": diagnostics,
            "objectives": objectives,
            "bayes_search": bayes_search,
            "dspy_a_b": dspy_v1,
            "extras": {},
        }
        if "autorag_spec" in bundle:
            out["extras"]["autorag_spec"] = bundle["autorag_spec"]
        if "metric_map" in bundle:
            out["extras"]["metric_map"] = bundle["metric_map"]
        return out

    # ------ shape B: single-mode flat (pdf_no_gt demo) --------------------
    if "autorag_bo" in bundle and isinstance(bundle.get("autorag_bo"), dict) and "trials" in bundle["autorag_bo"]:
        mode = bundle.get("gt_mode") or ("with_gt" if bundle.get("has_gt") else "no_gt")
        bo = bundle["autorag_bo"]
        dspy = bundle.get("dspy_before_after") or {}
        objective_spec = (
            bundle.get("objective_spec")
            or bo.get("objective_spec")
            or {"metrics": {}, "constraints": {}, "mode": "weighted_sum"}
        )

        source: dict[str, Any] = {"kind": "pdf" if "pdf_meta" in bundle or "source_pdf" in bundle else "unknown"}
        if "source_pdf" in bundle:
            source["corpus"] = bundle["source_pdf"]
        if "pdf_meta" in bundle:
            source["pdf_meta"] = bundle["pdf_meta"]

        extras: dict[str, Any] = {}
        if "pdf_meta" in bundle:
            extras["pdf_meta"] = bundle["pdf_meta"]
        if "synthesized_meta" in bundle:
            extras["synthesized_questions"] = bundle["synthesized_meta"]

        return {
            "schema_version": SCHEMA_VERSION,
            "meta": {
                "model": bundle.get("model"),
                "model_endpoint": bundle.get("model_endpoint"),
                "experiment_mode": mode,
                "modes_run": [mode],
                "has_gt": mode == "with_gt",
                "detected_gt_mode": mode,
                "source": source,
                "_migrated_from": "legacy_pdf_v0",
            },
            "questions": bundle.get("questions", []),
            "data_diagnostics": {
                mode: {
                    "has_ground_truth": mode == "with_gt",
                    "gt_mode": mode,
                    "record_count": len(bundle.get("questions") or []),
                }
            },
            "objectives": {mode: objective_spec},
            "bayes_search": {mode: bo},
            "dspy_a_b": {mode: dspy} if dspy else {},
            "extras": extras,
        }

    # Unknown shape -- stamp with schema_version=0 so callers can detect.
    out = dict(bundle)
    out["schema_version"] = 0
    return out


# =====================================================================
# Public entry point
# =====================================================================
def run_experiment(
    *,
    corpus: str | Path,
    has_gt: bool,
    mode: ExperimentMode = "auto",
    questions_path: str | Path | None = None,
    n_questions: int = 5,
    n_bo_trials: int = 8,
    n_bo_init: int = 3,
    top_ks: list[int] | None = None,
    chunk_sizes: list[int] | None = None,
    chunk_overlaps: list[int] | None = None,
    objective_overrides: dict[str, CompositeObjective] | None = None,
    output_dir: str | Path = ".ragdx_experiment",
    api_key: str | None = None,
    api_base: str = "https://open.bigmodel.cn/api/paas/v4",
    model: str = "openai/glm-4-flash",
    seed: int = 7,
    llm_max_concurrent: int = 2,
    llm_max_retries: int = 5,
    save: bool = True,
) -> ExperimentResult:
    """Run the complete demo pipeline once and return the bundle.

    Parameters
    ----------
    corpus:
        Document source. One of:

        * ``"<org>/<dataset>"`` — a HuggingFace dataset name (auto-loads
          questions and corpus chunks together, e.g. ``"explodinggradients/amnesty_qa"``).
        * Path to a ``.pdf`` — loaded via :func:`ragdx.loaders.load_pdf_chunks`.
        * Path to a ``.jsonl`` containing ``{text, source?}`` per line.
    has_gt:
        Whether the input carries ground-truth answers. When ``True`` and
        the corpus is a PDF or JSONL, supply ``questions_path`` so we
        know where the labelled questions live.
    mode:
        ``"with_gt"`` (requires has_gt), ``"no_gt"``, ``"both"`` (requires
        has_gt), or ``"auto"`` (default — picks ``"both"`` when has_gt,
        ``"no_gt"`` otherwise).
    questions_path:
        Optional JSONL with ``{question, ground_truth, contexts?}`` per
        line. Required when ``has_gt=True`` and ``corpus`` is a PDF or
        plain JSONL corpus.
    n_questions:
        Number of records to use. For HF sources this slices the eval
        split; for synthesised PDF/JSONL questions it controls the call
        count to ``synthesize_questions``.
    n_bo_trials / n_bo_init:
        Bayesian-search budget passed to :class:`BayesianSearch`.
    top_ks / chunk_sizes / chunk_overlaps:
        Search axes. Default matches the demos.
    objective_overrides:
        Optional ``{"with_gt": CompositeObjective, "no_gt": ...}`` to
        replace the production defaults from
        :func:`ragdx.optim.objectives.default_objective`.
    output_dir:
        Where ``result.json`` is written (also used by the Streamlit
        dashboards if you point them at this path).
    api_key:
        GLM / OpenAI API key. Falls back to ``ZHIPU_API_KEY`` /
        ``OPENAI_API_KEY`` env vars.
    api_base / model:
        Override defaults if you're using a different provider.
    seed:
        Deterministic seed for BO + question synthesis.
    llm_max_concurrent:
        Max in-flight LLM calls per evaluation batch. Default ``2`` is
        tuned for strict rate-limited endpoints (e.g. GLM-4-Flash); bump
        to ``8-16`` for OpenAI / Anthropic to speed up trials. Propagates
        to ragas ``RunConfig.max_workers`` and DSPy MIPROv2 ``num_threads``.
    llm_max_retries:
        Per-call transport-layer retry budget. Propagates to the openai
        client (used by the ragas judge) and ``litellm.completion`` (used
        by BO generation and DSPy). Default ``5``.
    save:
        If True (default), write ``output_dir/result.json``.

    Returns
    -------
    ExperimentResult with the full bundle (also written to disk by default).
    """
    cfg = ExperimentConfig(
        corpus=corpus,
        has_gt=has_gt,
        mode=mode,
        questions_path=questions_path,
        n_questions=n_questions,
        n_bo_trials=n_bo_trials,
        n_bo_init=n_bo_init,
        top_ks=top_ks or [1, 3, 5, 7],
        chunk_sizes=chunk_sizes or [256, 512, 1024],
        chunk_overlaps=chunk_overlaps or [0, 50, 100],
        objective_overrides=objective_overrides,
        output_dir=output_dir,
        api_key=api_key,
        api_base=api_base,
        model=model,
        seed=seed,
        llm_max_concurrent=llm_max_concurrent,
        llm_max_retries=llm_max_retries,
    )

    runtime = _build_runtime(cfg)
    chunks_master, base_records, source_meta = _load_corpus_and_records(cfg, runtime)

    results_by_mode: dict[str, dict] = {}
    if cfg.mode in ("with_gt", "both"):
        records_gt = [
            DatasetRecord(
                question=r.question, ground_truth=r.ground_truth, contexts=list(r.contexts)
            )
            for r in base_records
        ]
        results_by_mode["with_gt"] = _run_one_mode(
            cfg, runtime, chunks_master, records_gt, "with_gt"
        )
    if cfg.mode in ("no_gt", "both"):
        records_no = [
            DatasetRecord(
                question=r.question, ground_truth=None, contexts=list(r.contexts)
            )
            for r in base_records
        ]
        results_by_mode["no_gt"] = _run_one_mode(
            cfg, runtime, chunks_master, records_no, "no_gt"
        )

    bundle = _build_unified_bundle(cfg, results_by_mode, source_meta, base_records)

    output_path = cfg.output_dir / "result.json"
    result = ExperimentResult(config=cfg, bundle=bundle, output_path=output_path)
    if save:
        result.save()
    return result


__all__ = [
    "SCHEMA_VERSION",
    "ExperimentConfig",
    "ExperimentMode",
    "ExperimentResult",
    "migrate_legacy_bundle",
    "run_experiment",
]
