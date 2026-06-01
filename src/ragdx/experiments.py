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
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ragdx.core.evaluator import UnifiedEvaluator
from ragdx.datasets import synthesize_questions
from ragdx.loaders import load_pdf_chunks
from ragdx.optim._gt_helpers import gt_mode as detect_gt_mode
from ragdx.optim.objectives import CompositeObjective, default_objective
from ragdx.optim.stages import (
    GenerationOptimizer,
    JointOptimizer,
    StageContext,
)
from ragdx.runtime.factories import (
    RagdxRuntime,
    apply_litellm_temperature_clamp,
    build_runtime,
)
from ragdx.runtime.pipeline import RAGPipeline
from ragdx.schemas.models import DatasetRecord
from ragdx.schemas.rag_config import (
    ChunkerSpec,
    CorpusSpec,
    EmbedderSpec,
    GeneratorSpec,
    JudgeSpec,
    RAGConfig,
    RetrieverSpec,
)

ExperimentMode = Literal["with_gt", "no_gt", "both", "auto"]
logger = logging.getLogger(__name__)


# =====================================================================
# Default system instruction (single source of truth)
# =====================================================================
DEFAULT_SYSTEM_INSTRUCTION = (
    "Answer the question using only the retrieved context.\n"
    "Be concise. Do not invent facts that are not in the context."
)
"""Default RAG system instruction.

Used by **both** the BO trial generation path (raw prompt assembled in
:func:`_generate_answer`) and the DSPy baseline signature
(:class:`DSPyAdapter.build_program` falls back to this when no
``instruction`` is supplied). Keeping the two paths on the same default
makes the BO-stage scores and the DSPy-baseline scores directly
comparable -- they're scored against the same prompt.

Override per-run via :attr:`ExperimentConfig.system_instruction` (Python
API) or the ``--system-instruction`` / ``--system-instruction-file`` CLI
flags. The actually-used instruction is recorded in
``bundle.meta.run_config.system_instruction`` for reproducibility.
"""


# =====================================================================
# Public dataclasses
# =====================================================================
@dataclass
class ExperimentConfig:
    """Configuration for :func:`run_experiment`.

    Fields are deliberately tuned so the default values reproduce the
    PDF / amnesty_qa demos when fed the matching ``corpus`` argument.
    """

    corpus: str | Path | list[str | Path]
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

    system_instruction: str | None = None
    """RAG system instruction shared by BO generation and DSPy baseline.

    None resolves to :data:`DEFAULT_SYSTEM_INSTRUCTION` at runtime-build
    time. Override to inject domain-specific guidance (legal / medical /
    finance / etc.) -- both the BO trial generator and the DSPy baseline
    program will use this string, keeping their scores comparable.
    The DSPy MIPROv2 optimizer may then evolve this further on top.

    The resolved value is recorded in ``bundle.meta.run_config`` so the
    dashboard can display it and you can reproduce the run from the
    bundle alone."""

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
    """Captures MIPROv2's optimization log so the visualization can
    show per-trial decisions.

    Three buckets:

    * ``scores_so_far`` / ``best_scores`` -- the raw score progression
      DSPy logs ("Scores so far: [...]") for the score plot.
    * ``trials`` -- per-trial records reconstructed from the
      "== Trial N / M ==" + "Score: X ... parameters {...}" log lines.
      Each record has ``trial``, ``score``, ``kind`` (default /
      minibatch / full), and ``params`` (predictor → instruction_idx).
      Combined with the proposed instructions captured separately
      (see ``GenerationOptimizer._propose_instructions`` monkey-patch),
      this gives the user the "what prompt got tried on which trial"
      mapping the HTML report visualizes.
    """

    # Capture patterns. Tolerates two DSPy 3.x header styles:
    #   * ``== Trial 1 / 10 - Full Evaluation of Default Program ==``
    #   * ``===== Trial 2 / 10 =====``
    # The "- <kind> " part is optional; when absent we default to
    # ``"minibatch"`` since that's how MIPROv2 logs every trial after
    # the first.
    _TRIAL_HEADER_RE = re.compile(
        r"==+\s*Trial\s*(\d+)\s*/\s*(\d+)\s*"
        r"(?:-\s*([A-Za-z][\w ]+?)\s*)?==+",
    )
    _SCORE_LINE_RE = re.compile(
        r"^Score:\s*([0-9.\-eE]+)(?:\s+on minibatch[^.]*)?"
        # ``parameters`` is either a dict ``{...}`` (older DSPy) or a
        # list of strings ``[...]`` (DSPy 3.x: e.g.
        # ``['Predictor 0: Instruction 1', 'Predictor 0: Few-Shot Set 3']``).
        r"\s*with parameters\s*([\{\[].*?[\}\]])\.?$",
    )
    _DEFAULT_SCORE_RE = re.compile(
        r"Default program score:\s*([0-9.\-eE]+)",
    )

    def __init__(self) -> None:
        super().__init__()
        self.scores_so_far: list[float] = []
        self.best_scores: list[float] = []
        self.trials: list[dict] = []
        self._current_trial: int | None = None
        self._current_kind: str | None = None

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "Scores so far:" in msg:
            tail = msg.split("Scores so far:", 1)[1].strip()
            try:
                self.scores_so_far = json.loads(tail)
            except Exception:
                pass
            return
        if "Best score so far:" in msg:
            tail = msg.split("Best score so far:", 1)[1].strip()
            try:
                self.best_scores.append(float(tail))
            except Exception:
                pass
            return
        # Trial header: remember the active trial so the next Score line
        # gets attributed correctly. ``kind`` is optional in the
        # ``===== Trial N / M =====`` short form (MIPROv2 only labels
        # the default trial explicitly; the rest are minibatch by
        # convention).
        m = self._TRIAL_HEADER_RE.search(msg)
        if m is not None:
            self._current_trial = int(m.group(1))
            kind_grp = m.group(3)
            self._current_kind = (
                kind_grp.strip().lower() if kind_grp else "minibatch"
            )
            return
        # Default-program baseline score.
        m = self._DEFAULT_SCORE_RE.search(msg)
        if m is not None:
            try:
                self.trials.append({
                    "trial": self._current_trial or 1,
                    "score": float(m.group(1)),
                    "kind": "default",
                    "params": None,
                })
            except Exception:
                pass
            return
        # Per-trial Score line with chosen parameters.
        for line in msg.splitlines():
            m = self._SCORE_LINE_RE.search(line.strip())
            if m is not None:
                try:
                    params_str = m.group(2)
                    self.trials.append({
                        "trial": self._current_trial,
                        "score": float(m.group(1)),
                        "kind": self._current_kind or "minibatch",
                        # Keep as a string -- the dict format varies
                        # across DSPy versions and would require fragile
                        # eval to parse. The HTML renderer just displays
                        # it as a code fragment.
                        "params": params_str,
                    })
                except Exception:
                    pass


# =====================================================================
# Runtime helpers -- delegated to ragdx.runtime.factories
# =====================================================================
# Backward-compat alias: pre-PR4 code referenced ``ragdx.experiments._Runtime``
# and ``_apply_litellm_temperature_clamp``. PR4 hoisted them to
# :mod:`ragdx.runtime.factories` so non-experiment callers (workflows /
# CLI tune) can reach the same factories. The aliases below keep
# internal imports working.
_Runtime = RagdxRuntime
_apply_litellm_temperature_clamp = apply_litellm_temperature_clamp


def _build_runtime(cfg: ExperimentConfig) -> RagdxRuntime:
    """Build the runtime for an ``ExperimentConfig``.

    Delegates to :func:`ragdx.runtime.factories.build_runtime` via the
    ``ExperimentConfig`` -> ``RAGConfig`` mapping in
    :func:`_make_rag_config_from_experiment_config` -- giving the
    experiment workflow the same end state as ``ragdx evaluate`` /
    ``ragdx tune`` would produce. Byte-identical pre/post PR4.
    """
    rag_config = _make_rag_config_from_experiment_config(cfg)
    return build_runtime(rag_config)


def _make_rag_config_from_experiment_config(cfg: ExperimentConfig) -> RAGConfig:
    """Translate the experiment-flavoured config into a
    :class:`RAGConfig` so the runtime factory sees the same fields a
    user would put in their ``rag_config.yaml``.

    Distinct from :func:`_make_rag_config` (which takes a ``runtime``)
    because we need to build the config *before* the runtime exists.
    """
    return RAGConfig(
        corpus=CorpusSpec(
            kind="multi" if isinstance(cfg.corpus, list) else "pdf",
            path=str(cfg.corpus) if not isinstance(cfg.corpus, list) else None,
        ),
        chunker=ChunkerSpec(
            strategy="recursive",
            chunk_size=cfg.chunk_sizes[0] if cfg.chunk_sizes else 512,
            chunk_overlap=cfg.chunk_overlaps[0] if cfg.chunk_overlaps else 50,
        ),
        embedder=EmbedderSpec(
            kind="huggingface",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            normalize=True,
        ),
        retriever=RetrieverSpec(
            vectorstore="faiss",
            search_type="similarity",
            top_k=cfg.top_ks[0] if cfg.top_ks else 5,
            reranker="none",
        ),
        generator=GeneratorSpec(
            provider="litellm",
            model=cfg.model,
            api_base=cfg.api_base,
            api_key=cfg.api_key,
            system_instruction=cfg.system_instruction,
        ),
        judge=JudgeSpec(
            model=None,  # = use generator's model
            llm_max_concurrent=cfg.llm_max_concurrent,
            llm_max_retries=cfg.llm_max_retries,
        ),
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


def _load_single_corpus_item(
    item: str | Path, cfg: ExperimentConfig
) -> tuple[list[str], list[DatasetRecord] | None, dict]:
    """Load one corpus item.

    Returns ``(chunks, records_or_None, source_meta)``:

    * ``records_or_None`` is populated *only* when the item is a
      HuggingFace dataset (which inherently carries Q+GT). For PDFs
      and JSONL corpora the caller is responsible for either loading
      ``cfg.questions_path`` or synthesizing questions from the
      merged chunk pool.
    """
    source_meta: dict[str, Any] = {"corpus": str(item)}

    if isinstance(item, str) and _looks_like_hf_dataset(item):
        chunks, records = _load_amnesty_style_hf(item, cfg.n_questions)
        source_meta["kind"] = "huggingface"
        source_meta["dataset"] = item
        source_meta["records"] = len(records)
        source_meta["corpus_chunks"] = len(chunks)
        return chunks, records, source_meta

    path = Path(item)
    if path.suffix.lower() == ".pdf":
        loader = load_pdf_chunks(path, chunk_size=512, chunk_overlap=50)
        chunks = loader.chunks
        source_meta["kind"] = "pdf"
        source_meta["pdf_meta"] = loader.metadata
        return chunks, None, source_meta

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
        return chunks, None, source_meta

    raise ValueError(
        f"Unsupported corpus: {item!r}. Accepts: HuggingFace dataset name "
        "(e.g. 'org/dataset'), .pdf path, or .jsonl path."
    )


def _normalize_corpus(corpus: Any) -> list[str | Path]:
    """Coerce ``corpus`` into a flat list of items, accepting:

    * a single string / Path  → ``[item]``
    * a list of strings / Paths → ``list(corpus)``
    * a comma-separated string (CLI passes them this way) → ``[a, b, ...]``
    """
    if isinstance(corpus, (list, tuple)):
        return list(corpus)
    if isinstance(corpus, str) and "," in corpus:
        return [p.strip() for p in corpus.split(",") if p.strip()]
    return [corpus]


def _load_corpus_and_records(
    cfg: ExperimentConfig, runtime: _Runtime
) -> tuple[list[str], list[DatasetRecord], dict]:
    """Dispatch on the ``corpus`` argument and produce
    ``(chunks, records, source_meta)``. Records may have empty GT in
    no-GT mode -- that's intentional and downstream handles it.

    ``cfg.corpus`` accepts a single item or a list. For multi-corpus
    runs chunks are concatenated into one pool; records come from
    (in order of precedence) ``cfg.questions_path``, then the first
    HuggingFace item that carries Q+GT, then synthesis from the
    merged chunk pool.
    """
    items = _normalize_corpus(cfg.corpus)
    all_chunks: list[str] = []
    sources: list[dict] = []
    hf_records: list[DatasetRecord] | None = None
    pdf_chunks_used = False

    for item in items:
        chunks_i, records_i, meta_i = _load_single_corpus_item(item, cfg)
        all_chunks.extend(chunks_i)
        sources.append(meta_i)
        if hf_records is None and records_i:
            hf_records = records_i
        if meta_i.get("kind") == "pdf":
            pdf_chunks_used = True

    # Records resolution: explicit JSONL path always wins; otherwise HF
    # records if any; otherwise synthesize / fail.
    if cfg.questions_path:
        records = _load_jsonl_questions(Path(cfg.questions_path))
        for s in sources:
            s["questions_path"] = str(cfg.questions_path)
    elif hf_records is not None:
        records = hf_records
    elif cfg.has_gt:
        raise ValueError(
            "has_gt=True with PDF/JSONL corpus (or multi-corpus including "
            "one) requires `questions_path` -- a JSONL with "
            "{question, ground_truth, ...} per line."
        )
    else:
        synthesised = synthesize_questions(
            all_chunks,
            n=cfg.n_questions,
            llm_callable=runtime.llm_callable,
            chunks_per_question=2,
        )
        records = [
            DatasetRecord(question=q.question, ground_truth=None, contexts=[])
            for q in synthesised
        ]
        synth_meta = [
            {"question": q.question, "source_chunk_ids": q.source_chunk_ids}
            for q in synthesised
        ]
        # Stash on the first PDF source (single-corpus compat) when there
        # is exactly one PDF, otherwise emit at the top-level source meta.
        if len(sources) == 1 and pdf_chunks_used:
            sources[0]["synthesized_meta"] = synth_meta
        else:
            # multi-corpus: attach to the umbrella source_meta below
            pass

    if len(sources) == 1:
        source_meta = sources[0]
    else:
        source_meta = {
            "kind": "multi",
            "corpus": [str(i) for i in items],
            "corpus_chunks": len(all_chunks),
            "records": len(records),
            "sources": sources,
        }
        # If we just synthesized for multi-corpus, attach the synth meta
        # at the top level too.
        if not cfg.questions_path and hf_records is None and not cfg.has_gt:
            synth_meta = [
                {"question": r.question, "source_chunk_ids": []}
                for r in records
            ]
            source_meta["synthesized_meta"] = synth_meta

    return all_chunks, records, source_meta


# =====================================================================
# RAG primitives -- delegated to ragdx.runtime.pipeline.RAGPipeline
# =====================================================================
def _make_rag_config(cfg: ExperimentConfig, runtime: _Runtime) -> RAGConfig:
    """Build the base :class:`RAGConfig` reflecting the experiment cfg.

    This is the "starting point" config for the BO loop; the loop then
    derives per-trial configs via ``base.with_override(chunker=..., retriever=...)``.
    Mapping is mechanical:

    * Corpus / chunker / retriever defaults reflect what BO starts with
      (first item of each search-space axis).
    * Generator inherits ``cfg.model`` / ``cfg.api_base`` /
      ``runtime.system_instruction``.
    * Judge mirrors the generator for now (PR2 will let users pin a
      stronger judge).

    The returned object is mostly a documentation device in PR1 -- BO
    still drives parameters via :class:`BayesianSearch`. PR3 will move
    search-space definitions onto the spec classes themselves.
    """
    corpus = CorpusSpec(
        kind="multi" if isinstance(cfg.corpus, list) else "pdf",
        path=str(cfg.corpus) if not isinstance(cfg.corpus, list) else None,
    )
    return RAGConfig(
        corpus=corpus,
        chunker=ChunkerSpec(
            strategy="recursive",
            chunk_size=cfg.chunk_sizes[0] if cfg.chunk_sizes else 512,
            chunk_overlap=cfg.chunk_overlaps[0] if cfg.chunk_overlaps else 50,
        ),
        embedder=EmbedderSpec(
            kind="huggingface",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            normalize=True,
        ),
        retriever=RetrieverSpec(
            vectorstore="faiss",
            search_type="similarity",
            top_k=cfg.top_ks[0] if cfg.top_ks else 5,
            reranker="none",
        ),
        generator=GeneratorSpec(
            provider="litellm",
            model=cfg.model,
            api_base=cfg.api_base,
            api_key=cfg.api_key,
            system_instruction=runtime.system_instruction,
        ),
        judge=JudgeSpec(
            model=None,  # = use generator's model
            llm_max_concurrent=cfg.llm_max_concurrent,
            llm_max_retries=cfg.llm_max_retries,
        ),
    )


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
# Per-mode orchestration
# =====================================================================
def _make_pdf_re_chunk_fn(
    cfg: ExperimentConfig, chunks_master: list[str]
) -> Callable[[ChunkerSpec], list[str]] | None:
    """For PDF corpora, return a callable that re-chunks every PDF item
    at the requested ``ChunkerSpec`` and pools the chunks. ``None`` for
    HF / JSONL where re-chunking the master pool isn't meaningful.

    Used as ``StageContext.re_chunk_fn`` by stages that vary the
    chunker (Joint, Chunking).
    """
    pdf_paths = [
        Path(i)
        for i in _normalize_corpus(cfg.corpus)
        if isinstance(i, (str, Path)) and Path(i).exists()
        and Path(i).suffix.lower() == ".pdf"
    ]
    if not pdf_paths:
        # No PDFs in the corpus -- return None so stages fall back to
        # ``chunks_master`` (consistent with pre-PR3 behaviour).
        return None

    def _re_chunk(chunker: ChunkerSpec) -> list[str]:
        out: list[str] = []
        for p in pdf_paths:
            out.extend(load_pdf_chunks(
                p, chunk_size=chunker.chunk_size, chunk_overlap=chunker.chunk_overlap,
            ).chunks)
        return out

    return _re_chunk


def _run_one_mode(
    cfg: ExperimentConfig,
    runtime: _Runtime,
    chunks_master: list[str],
    records: list[DatasetRecord],
    mode: str,
) -> dict:
    """Drive one GT mode through Joint (BO) + Generation (DSPy MIPROv2).

    Post-PR3 this is a thin composition of two
    :class:`StageOptimizer` instances. The bundle's ``bayes_search``
    section comes from ``JointOptimizer``; the ``dspy_a_b`` section
    comes from ``GenerationOptimizer``'s extras. Both produce the
    exact same shape they did pre-PR3 (locked in by the
    ``new_demo1`` / ``new_demo2`` snapshot tests).
    """
    objective = (cfg.objective_overrides or {}).get(mode) or default_objective(mode)
    metrics = _build_ragas_metrics_for_mode(mode)
    base_rag_config = _make_rag_config(cfg, runtime)
    re_chunk_fn = _make_pdf_re_chunk_fn(cfg, chunks_master)

    # --- Stage 1: Joint BO over (chunk_size, chunk_overlap, top_k) ----
    joint_ctx = StageContext(
        base_config=base_rag_config,
        chunks_master=chunks_master,
        records=records,
        objective=objective,
        metrics=metrics,
        runtime=runtime,
        n_bo_trials=cfg.n_bo_trials,
        n_bo_init=cfg.n_bo_init,
        seed=cfg.seed,
        re_chunk_fn=re_chunk_fn,
        label=mode,
        chunk_sizes=cfg.chunk_sizes,
        chunk_overlaps=cfg.chunk_overlaps,
        top_ks=cfg.top_ks,
    )
    joint_result = JointOptimizer().optimize(joint_ctx)

    # --- Build the BO winner's pipeline + pre-retrieve records --------
    # GenerationOptimizer operates on records that already carry
    # contexts (it only varies the prompt). Retrieve at the BO winner
    # exactly the way pre-PR3 ``_dspy_before_after`` did.
    best_params = joint_result.best_params or {}
    winner_config = joint_result.best_config or base_rag_config
    winner_chunks = (
        re_chunk_fn(winner_config.chunker) if re_chunk_fn else chunks_master
    )
    winner_pipeline = RAGPipeline.build(
        winner_config,
        winner_chunks,
        embedder=runtime.embeddings,
        llm_callable=runtime.llm_callable,
    )
    records_for_generation = [
        DatasetRecord(
            question=r.question,
            ground_truth=r.ground_truth,
            contexts=winner_pipeline.retrieve(
                r.question, top_k=best_params.get("top_k"),
            ),
        )
        for r in records
    ]

    # --- Stage 2: Generation (DSPy MIPROv2 prompt optimization) -------
    gen_ctx = StageContext(
        base_config=winner_config,
        chunks_master=winner_chunks,
        records=records_for_generation,
        objective=objective,
        metrics=metrics,
        runtime=runtime,
        label=mode,
    )
    gen_result = GenerationOptimizer().optimize(gen_ctx)

    return {
        "bayes_search": joint_result.to_bayes_search_bundle(),
        # GenerationOptimizer parks the dashboard-shaped payload in
        # ``extras`` (the StageResult.trials shape doesn't fit
        # MIPROv2's outputs cleanly). Same content as pre-PR3
        # ``_dspy_before_after``.
        "dspy_a_b": gen_result.extras,
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
            # Reproducibility: persist the LLM endpoint call-management
            # knobs used for this run so the dashboard can display them
            # and ``ragdx experiment ... --llm-max-concurrent N`` can be
            # reconstructed from the bundle alone.
            "run_config": {
                "llm_max_concurrent": cfg.llm_max_concurrent,
                "llm_max_retries": cfg.llm_max_retries,
                "n_bo_trials": cfg.n_bo_trials,
                "n_bo_init": cfg.n_bo_init,
                "n_questions": cfg.n_questions,
                "seed": cfg.seed,
                # Resolved system instruction (cfg.system_instruction or
                # DEFAULT). Recording the resolved value -- not None --
                # so reproducibility doesn't depend on the package
                # default staying stable across versions.
                "system_instruction": (
                    cfg.system_instruction or DEFAULT_SYSTEM_INSTRUCTION
                ),
            },
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
    corpus: str | Path | list[str | Path],
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
    system_instruction: str | None = None,
    save: bool = True,
) -> ExperimentResult:
    """Run the complete demo pipeline once and return the bundle.

    Parameters
    ----------
    corpus:
        Document source(s). One of:

        * ``"<org>/<dataset>"`` — a HuggingFace dataset name (auto-loads
          questions and corpus chunks together, e.g. ``"explodinggradients/amnesty_qa"``).
        * Path to a ``.pdf`` — loaded via :func:`ragdx.loaders.load_pdf_chunks`.
        * Path to a ``.jsonl`` containing ``{text, source?}`` per line.
        * **A list of any combination of the above** for multi-corpus
          runs. Chunks are concatenated into one pool; records come
          from (in precedence order) ``questions_path``, then the
          first HuggingFace item, then synthesis from the merged pool.
          From the CLI, pass comma-separated paths:
          ``ragdx experiment "a.pdf,b.pdf" --no-gt``.
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
    system_instruction:
        RAG system prompt shared by BO generation and the DSPy baseline.
        ``None`` falls back to :data:`DEFAULT_SYSTEM_INSTRUCTION`
        (``"Answer the question using only the retrieved context. Be
        concise. Do not invent facts that are not in the context."``).
        Override per-domain (legal / medical / finance) to steer both
        the BO trial generator and the DSPy baseline with the same
        instruction -- MIPROv2 may then evolve it further on top.
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
        system_instruction=system_instruction,
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
