"""End-to-end experiment driver for the ragdx demo pipeline.

The ``demo_optimize_gt_modes.py`` and ``demo_pdf_no_gt.py`` scripts in
``examples/`` walk the user through every step of a complete run.
``run_experiment`` is the importable equivalent: one function call that
takes a corpus + a few flags, runs AutoRAG Bayesian search, runs DSPy
before/after for each requested GT mode, and returns a structured
result you can save / inspect / feed into the dashboards.

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
        corpus="docs/asmpt-esg-report.pdf",
        has_gt=False,
        api_key="<glm-key>",
    )

    print(result.bundle["autorag_bo"]["with_gt"]["best_params"])

The bundle is also written to ``output_dir/result.json`` so the
Streamlit dashboards
(``ragdx.ui.optimization_dashboard`` / ``ragdx.ui.pdf_no_gt_dashboard``)
can render it directly.
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


def _apply_litellm_temperature_clamp(min_temp: float = 0.01) -> None:
    """GLM-4-Flash rejects temperature<0.01. DSPy / ragas occasionally send
    1e-5; clamp once at runtime-build time so every downstream call is safe.
    Idempotent (re-applying is a no-op)."""
    import litellm

    if getattr(litellm.completion, "_ragdx_clamped", False):
        return
    orig_completion = litellm.completion
    orig_batch = litellm.batch_completion

    def _clamp(kw):
        t = kw.get("temperature")
        if t is not None and 0 < t < min_temp:
            kw["temperature"] = min_temp
        return kw

    def comp(*a, **kw):
        return orig_completion(*a, **_clamp(kw))

    def batch(*a, **kw):
        return orig_batch(*a, **_clamp(kw))

    comp._ragdx_clamped = True  # type: ignore[attr-defined]
    litellm.completion = comp
    litellm.batch_completion = batch


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
            max_retries=2,
        )
    )

    return _Runtime(
        llm_callable=llm_callable,
        dspy_lm=dspy_lm,
        ragas_judge=ragas_judge,
        ragas_embeddings=ragas_embeddings,
        embeddings=hf_embeddings,
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
    """Load a HuggingFace dataset that follows the amnesty_qa shape:
    {user_input, reference, retrieved_contexts}."""
    from datasets import load_dataset

    # amnesty_qa convention: subset name + eval split
    try:
        ds = load_dataset(name, "english_v3", split=f"eval[:{n_records}]")
    except Exception:  # pragma: no cover - other HF datasets vary
        ds = load_dataset(name, split=f"train[:{n_records}]")

    corpus_chunks: list[str] = []
    records: list[DatasetRecord] = []
    for row in ds:
        for passage in row.get("retrieved_contexts") or []:
            text = (passage or "").strip()
            if text:
                corpus_chunks.append(text)
        records.append(
            DatasetRecord(
                question=row.get("user_input", row.get("question", "")),
                ground_truth=row.get("reference") or "",
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
    records: list[DatasetRecord], judge: Any, embeddings: Any, metrics: list
) -> dict:
    evaluator = UnifiedEvaluator()
    try:
        result = evaluator.evaluate(
            records,
            use_ragas=True, run_ragas=True,
            use_ragchecker=False, use_embedding=False,
            ragas_kwargs={"llm": judge, "embeddings": embeddings, "metrics": metrics},
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

        ev = _evaluate_with_ragas(answered, runtime.ragas_judge, runtime.ragas_embeddings, metrics)
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
        baseline_answered, runtime.ragas_judge, runtime.ragas_embeddings, metrics
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
            optimizer_kwargs={"auto": "light"},
        )
    finally:
        mipro_logger.removeHandler(capture)
        mipro_logger.setLevel(prior)

    logger.info("[DSPy/%s] (c) optimised re-run", label)
    optimised_answered = _run_program(opt_result["optimized_program"])
    opt_eval = _evaluate_with_ragas(
        optimised_answered, runtime.ragas_judge, runtime.ragas_embeddings, metrics
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
        "autorag_bo": bo_result,
        "dspy_before_after": dspy_result,
        "objective_spec": objective.to_dict(),
    }


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

    autorag_bundle = {
        k: v["autorag_bo"] for k, v in results_by_mode.items()
    }
    dspy_bundle = {k: v["dspy_before_after"] for k, v in results_by_mode.items()}

    bundle: dict[str, Any] = {
        "model": cfg.model,
        "model_endpoint": cfg.api_base,
        "experiment_mode": cfg.mode,
        "has_gt": cfg.has_gt,
        "source": source_meta,
        "questions": [
            {"question": r.question, "ground_truth": r.ground_truth} for r in base_records
        ],
        "data_diagnostics": {
            mode_key: {
                "has_ground_truth": (mode_key == "with_gt"),
                "gt_mode": mode_key,
                "record_count": len(base_records),
            }
            for mode_key in results_by_mode
        },
        "autorag_bo": autorag_bundle if len(autorag_bundle) > 1 else next(iter(autorag_bundle.values())),
        "dspy_before_after": dspy_bundle if len(dspy_bundle) > 1 else next(iter(dspy_bundle.values())),
        "_modes_run": list(results_by_mode),
    }
    if "synthesized_meta" in source_meta:
        bundle["synthesized_meta"] = source_meta["synthesized_meta"]
    detected = detect_gt_mode(base_records)
    bundle["detected_gt_mode"] = detected

    output_path = cfg.output_dir / "result.json"
    result = ExperimentResult(config=cfg, bundle=bundle, output_path=output_path)
    if save:
        result.save()
    return result


__all__ = ["ExperimentConfig", "ExperimentMode", "ExperimentResult", "run_experiment"]
