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


# =====================================================================
# DSPy GEPA log capture (PR8+)
# =====================================================================
class _GEPATrialScoreCapture(logging.Handler):
    """Parse DSPy GEPA's per-iteration log so the HTML report can show
    "what prompts did GEPA actually propose, on which iterations,
    with what scores".

    GEPA logs four shapes of interest, each as a single ``logger.info``
    call (with ``Proposed new text`` carrying the **full multi-line**
    instruction text -- ``record.getMessage()`` returns it verbatim):

    * ``Iteration N: Base program full valset score: V over X / Y examples``
      -- iteration 0 baseline.
    * ``Iteration N: Selected program X score: V``
      -- which earlier candidate this iteration is mutating.
    * ``Iteration N: Proposed new text for predict: <multi-line text>``
      -- the newly-proposed instruction.
    * ``Iteration N: New subsample score V is (not )?better than old score W``
      -- whether the proposal beat the prior best at minibatch scoring.
    * ``Iteration N: Found a better program on the valset with score Z``
      -- full-valset confirmation when subsample passed.

    The accumulator surfaces two fields the existing renderer already
    knows how to draw (we deliberately reuse the MIPROv2 shapes so
    ``_render_dspy_a_b`` needs no GEPA-specific branch):

    * ``proposed_per_iter: list[{"iter": int, "text": str}]`` -- aligned
      with ``proposed_instructions_by_predictor["predict"]``.
    * ``trials: list[{"trial": int, "kind": str, "score": float,
                      "params": str}]`` -- aligned with the MIPROv2
      ``trial_log`` shape.
    """

    # Number patterns avoid the trailing-period gotcha:
    # ``5.592798987337338.`` (end of sentence) -> would otherwise be
    # captured including the dot and crash ``float()``.
    _NUM = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?"
    _BASE_RE = re.compile(
        rf"Iteration\s+(\d+):\s+Base program full valset score:\s*({_NUM})",
    )
    _SELECTED_RE = re.compile(
        rf"Iteration\s+(\d+):\s+Selected program\s+(\d+)\s+score:\s*({_NUM})",
    )
    _PROPOSED_RE = re.compile(
        r"Iteration\s+(\d+):\s+Proposed new text for predict:\s+",
    )
    _SUBSAMPLE_RE = re.compile(
        rf"Iteration\s+(\d+):\s+New subsample score\s+({_NUM})\s+"
        rf"is\s+(not\s+)?better than old score\s+({_NUM})",
    )
    _FULLEVAL_RE = re.compile(
        rf"Iteration\s+(\d+):\s+Found a better program on the valset "
        rf"with score\s+({_NUM})",
    )
    _NEW_PROG_IDX_RE = re.compile(
        r"Iteration\s+(\d+):\s+New program candidate index:\s+(\d+)",
    )

    def __init__(self) -> None:
        super().__init__()
        self.proposed_per_iter: list[dict] = []
        self.trials: list[dict] = []
        # State carried between log lines (one iteration spans
        # several records: Selected -> Proposed -> subsample [-> full]).
        self._current: dict | None = None

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "dspy.teleprompt.gepa" not in record.name and "gepa" not in msg.lower():
            # Belt-and-braces: we only attach to the gepa logger, but
            # if a caller wires us elsewhere we'd rather drop the line
            # than mis-attribute it.
            if "Iteration" not in msg:
                return

        # Iter 0 baseline.
        m = self._BASE_RE.search(msg)
        if m is not None:
            iter_num = int(m.group(1))
            score = float(m.group(2))
            self.trials.append({
                "trial": iter_num,
                "score": score,
                "kind": "default (baseline program)",
                "params": "Program 0 (seed)",
            })
            return

        # Iteration X Selected program Y score Z -- starts a new iter.
        m = self._SELECTED_RE.search(msg)
        if m is not None:
            iter_num = int(m.group(1))
            parent_idx = int(m.group(2))
            parent_score = float(m.group(3))
            self._current = {
                "iter": iter_num,
                "parent_idx": parent_idx,
                "parent_score": parent_score,
            }
            return

        # Iteration X Proposed new text -- multi-line. record.getMessage()
        # returns the full string. Strip the prefix and store.
        m = self._PROPOSED_RE.search(msg)
        if m is not None:
            iter_num = int(m.group(1))
            # The proposed text is everything after the prefix.
            prefix_end = msg.find("Proposed new text for predict:")
            text = msg[prefix_end + len("Proposed new text for predict:"):].strip()
            self.proposed_per_iter.append({"iter": iter_num, "text": text})
            return

        # Iteration X New subsample score Y is (not )?better than Z.
        m = self._SUBSAMPLE_RE.search(msg)
        if m is not None:
            iter_num = int(m.group(1))
            sub_score = float(m.group(2))
            rejected = bool(m.group(3))
            old_score = float(m.group(4))
            kind = "subsample (rejected)" if rejected else "subsample (accepted)"
            params = (
                f"Iter {iter_num}: proposed score {sub_score:.2f} vs prior best {old_score:.2f}"
            )
            self.trials.append({
                "trial": iter_num,
                "score": sub_score,
                "kind": kind,
                "params": params,
            })
            return

        # Iteration X Found a better program on the valset with score Z.
        m = self._FULLEVAL_RE.search(msg)
        if m is not None:
            iter_num = int(m.group(1))
            full_score = float(m.group(2))
            self.trials.append({
                "trial": iter_num,
                "score": full_score,
                "kind": "full valset eval (accepted)",
                "params": f"Iter {iter_num}: confirmed at full valset",
            })
            return

        # Iteration X New program candidate index: I -- annotation only,
        # logged for completeness so the renderer can show "added as
        # program N" in a future revision.
        m = self._NEW_PROG_IDX_RE.search(msg)
        if m is not None:
            iter_num = int(m.group(1))
            new_idx = int(m.group(2))
            # Adjust the latest "full valset eval (accepted)" trial.
            for trial in reversed(self.trials):
                if (
                    trial.get("trial") == iter_num
                    and trial.get("kind") == "full valset eval (accepted)"
                ):
                    trial["params"] = (
                        f"Iter {iter_num}: added as program {new_idx}"
                    )
                    break


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
            # Phase 4a: forward per-record metric rows so trial bundles
            # can carry them through to the HTML report. Empty list when
            # the underlying ragas Result couldn't expose them.
            "per_record_scores": result.metadata.get("per_record_scores", []),
        }
    except Exception as e:  # pragma: no cover - live LLM
        # Surface the error so silent eval failures don't get buried.
        # Without this log line, an upstream ragas / LLM hiccup just
        # produces an empty ``scores: {}`` dict that callers happily
        # accept, and the bundle ends up score-less for no obvious
        # reason. The DEBUG-level log records the full traceback for
        # users who set ``RAGDX_LOG_LEVEL=DEBUG``.
        logger.warning(
            "_evaluate_with_ragas: %s: %s (returning empty scores)",
            type(e).__name__, e,
        )
        logger.debug("ragas eval traceback", exc_info=True)
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


def _supplement_deepeval_metrics(
    records_payload: list[dict],
    runtime: RagdxRuntime,
) -> dict[str, float]:
    """Compute the deepeval-only metrics over the optimized answers.

    Returns ``{metric: mean_score}`` for the deepeval metrics ragas
    doesn't produce: ``hallucination``, ``bias``, ``toxicity``,
    ``g_eval``. Each metric is computed per-record with a per-call
    guard so a judge timeout / parse error degrades to "metric not
    computed" rather than sinking the whole run.

    ``records_payload`` items carry ``question`` / ``contexts`` /
    ``optimized_answer`` (the dspy_a_b record shape). Returns ``{}``
    when deepeval (or its judge) isn't available, or no record could
    be scored.
    """
    judge = getattr(runtime, "deepeval_judge", None)
    if judge is None:
        return {}
    try:
        from deepeval.metrics import (
            BiasMetric,
            GEval,
            HallucinationMetric,
            ToxicityMetric,
        )
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    except Exception:  # pragma: no cover - deepeval optional
        return {}

    # Build the metric objects once; reuse across records.
    geval = GEval(
        name="g_eval",
        criteria=(
            "Holistically rate how well the answer responds to the "
            "question using ONLY the retrieved context: grounded in the "
            "context, complete, and directly on-topic. Higher is better."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=judge,
    )
    # Standalone answer-quality metrics (no context needed for bias /
    # toxicity; hallucination compares answer against the context).
    builders: dict[str, Any] = {
        "hallucination": lambda: HallucinationMetric(model=judge),
        "bias": lambda: BiasMetric(model=judge),
        "toxicity": lambda: ToxicityMetric(model=judge),
        "g_eval": lambda: geval,
    }

    acc: dict[str, list[float]] = {k: [] for k in builders}
    for r in records_payload:
        q = str(r.get("question") or "")
        ans = str(r.get("optimized_answer") or r.get("answer") or "")
        ctxs = [str(c) for c in (r.get("contexts") or []) if c]
        if not ans:
            continue
        # HallucinationMetric uses ``context`` (the ground-truth-ish
        # source) rather than ``retrieval_context``; we pass the
        # retrieved contexts as both so it has something to compare to.
        tc = LLMTestCase(
            input=q, actual_output=ans,
            retrieval_context=ctxs or None, context=ctxs or None,
        )
        for name, build in builders.items():
            try:
                m = build()
                m.measure(tc)
                s = float(getattr(m, "score", None))
                if s == s:  # not NaN
                    acc[name].append(max(0.0, min(1.0, s)))
            except Exception as exc:  # pragma: no cover - judge-dependent
                logger.debug("deepeval %s failed on a record: %s", name, exc)

    out: dict[str, float] = {}
    for name, vals in acc.items():
        if vals:
            out[name] = round(sum(vals) / len(vals), 4)
    if out:
        logger.info("deepeval supplement computed: %s", out)
    return out


def _run_one_mode(
    cfg: ExperimentConfig,
    runtime: _Runtime,
    chunks_master: list[str],
    records: list[DatasetRecord],
    mode: str,
    *,
    ckpts: _ExperimentCheckpoints | None = None,
) -> dict:
    """Drive one GT mode through Joint (BO) + Generation (DSPy MIPROv2).

    Post-PR3 this is a thin composition of two
    :class:`StageOptimizer` instances. The bundle's ``bayes_search``
    section comes from ``JointOptimizer``; the ``dspy_a_b`` section
    comes from ``GenerationOptimizer``'s extras. Both produce the
    exact same shape they did pre-PR3 (locked in by the
    ``new_demo1`` / ``new_demo2`` snapshot tests).

    ``ckpts`` threads the per-(mode, stage) experiment checkpoints into
    the stage optimizers, which save after every BO trial / generation
    phase. ``None`` (the default, and what unit tests pass implicitly)
    keeps the no-checkpoint behaviour.
    """
    objective = (cfg.objective_overrides or {}).get(mode) or default_objective(mode)
    metrics = _build_ragas_metrics_for_mode(mode)
    base_rag_config = _make_rag_config(cfg, runtime)
    re_chunk_fn = _make_pdf_re_chunk_fn(cfg, chunks_master)
    ckpt_store = ckpts.store if ckpts is not None else None

    # --- Stage 1: Joint BO over (chunk_size, chunk_overlap, top_k) ----
    ckpt_joint = ckpts.get(mode, "joint") if ckpts is not None else None
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
        checkpoint=ckpt_joint,
        checkpoint_store=ckpt_store,
    )
    joint_result = JointOptimizer().optimize(joint_ctx)
    if ckpts is not None:
        ckpts.complete(ckpt_joint)

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

    # --- Stage 2: Generation (DSPy prompt optimization) ---------------
    ckpt_gen = ckpts.get(mode, "generation") if ckpts is not None else None
    gen_ctx = StageContext(
        base_config=winner_config,
        chunks_master=winner_chunks,
        records=records_for_generation,
        objective=objective,
        metrics=metrics,
        runtime=runtime,
        label=mode,
        checkpoint=ckpt_gen,
        checkpoint_store=ckpt_store,
    )
    gen_result = GenerationOptimizer().optimize(gen_ctx)
    if ckpts is not None:
        ckpts.complete(ckpt_gen)

    # --- Stage 3: deepeval supplement (item 1) -----------------------
    # ragas gives us context_precision / faithfulness / answer_relevancy.
    # deepeval adds the metrics ragas doesn't compute -- hallucination,
    # bias, toxicity, g_eval -- so the three-layer view shows real
    # numbers instead of "requires deepeval". Computed on the OPTIMIZED
    # answers (the final system). Best-effort: each metric is wrapped so
    # a judge timeout degrades to "not computed" rather than failing the
    # run. No-op when deepeval isn't installed.
    try:
        extra = _supplement_deepeval_metrics(
            gen_result.extras.get("records") or [], runtime,
        )
        if extra:
            opt = dict(gen_result.extras.get("optimized_scores") or {})
            opt.update(extra)
            gen_result.extras["optimized_scores"] = opt
            gen_result.extras["deepeval_supplement"] = extra
    except Exception as exc:  # pragma: no cover - depends on judge LM
        logger.warning("deepeval supplement skipped: %s", exc)

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


def _winner_scores(bayes_bundle: dict) -> dict[str, float]:
    """Pull per-metric scores from the BO winner trial.

    The trials list is the source of truth; ``best_params`` identifies
    the winning row. Falls back to the highest-composite trial if no
    params match, and to ``{}`` if there are no trials. Used by the
    diagnosis synthesis below -- we feed these into a synthetic
    ``EvaluationResult`` so :class:`RAGDiagnosisEngine` can produce a
    DiagnosisReport for the BO winner config.
    """
    trials = bayes_bundle.get("trials") or []
    if not trials:
        return {}
    best_params = bayes_bundle.get("best_params") or {}
    if best_params:
        for t in trials:
            if (t.get("params") or {}) == best_params:
                return dict(t.get("scores") or {})
    # Fallback: pick highest composite_score with finite value.
    best = max(
        trials,
        key=lambda t: t.get("composite_score")
        if isinstance(t.get("composite_score"), (int, float))
        else float("-inf"),
        default=None,
    )
    return dict((best or {}).get("scores") or {})


def _synth_eval_result(
    scores: dict[str, float],
    *,
    mode: str,
    extra_metadata: dict[str, Any] | None = None,
) -> Any:
    """Build a synthetic ``EvaluationResult`` from a flat ``{metric: score}``
    dict so we can run :class:`RAGDiagnosisEngine` against the BO
    winner config without a second evaluator pass.

    Bucketed via the canonical ``LAYER_OF`` map (same routing as the
    three-layer overview and ``workflows.evaluate``), so deepeval
    supplements (``bias`` / ``toxicity`` / ``g_eval``) land in their
    proper layers. Unknown metrics land in ``e2e`` so they aren't
    silently dropped.
    """
    from ragdx.core.metrics import LAYER_OF
    from ragdx.schemas.models import EvaluationResult

    buckets: dict[str, dict[str, float]] = {
        "retrieval": {}, "generation": {}, "e2e": {},
    }
    for k, v in scores.items():
        layer = LAYER_OF.get(k)
        buckets[layer if layer in buckets else "e2e"][k] = v
    md = {"synthesized_from": "bo_winner", "gt_mode": mode}
    if extra_metadata:
        md.update(extra_metadata)
    return EvaluationResult(
        retrieval=buckets["retrieval"],
        generation=buckets["generation"],
        e2e=buckets["e2e"],
        metadata=md,
    )


def _baseline_scores_for_mode(payload: dict) -> dict[str, float]:
    """Return the *pre-optimization* metric scores for one mode.

    Diagnosis is a decision-making input -- it's meant to answer
    "where are the bottlenecks in *this baseline* so we know what to
    optimize". That logically precedes the optimization, so we
    diagnose the baseline scores, not the BO winner.

    Sources, in priority order:

    1. ``dspy_a_b[mode].baseline_scores`` -- the GenerationOptimizer
       evaluates the user's seed config (post-retrieval-BO, but with
       the default unmodified prompt) before kicking off MIPROv2.
       This is the most actionable "baseline" we have: it's what the
       user would see if they ran ``ragdx evaluate`` on the BO winner
       without any DSPy prompt tuning.
    2. The first trial in ``bayes_search.trials`` -- a reasonable
       proxy for the untouched config when the experiment skipped the
       DSPy generation stage.
    """
    dspy_a_b = payload.get("dspy_a_b") or {}
    baseline = dspy_a_b.get("baseline_scores")
    if baseline:
        return dict(baseline)
    trials = (payload.get("bayes_search") or {}).get("trials") or []
    if trials:
        return dict(trials[0].get("scores") or {})
    return {}


def _hyp_key(h: dict) -> tuple[str, str]:
    """Stable identity for a hypothesis (component + root_cause)."""
    return (str(h.get("component") or ""), str(h.get("root_cause") or ""))


def _compare_diagnoses(
    baseline: dict | None,
    optimized: dict | None,
    *,
    baseline_scores: dict[str, float] | None = None,
    optimized_scores: dict[str, float] | None = None,
    posterior_shift_threshold: float = 0.05,
) -> dict:
    """Compute a baseline vs optimized delta summary.

    This is the "did the optimization actually fix what we set out to
    fix?" layer. Without it the reader gets two adjacent diagnoses and
    has to eyeball the difference; with it they get an explicit list
    of what was resolved / persisted / emerged plus posterior shifts.

    Returns a dict with:

    * ``resolved_hypotheses``  -- in baseline, gone in optimized.
    * ``persisted_hypotheses`` -- in both (still needs attention).
    * ``emerged_hypotheses``   -- only in optimized (regression or
       newly-exposed bottleneck, e.g. fixing recall exposes precision).
    * ``posterior_shifts`` -- per causal node, ``{node, baseline,
       optimized, delta, direction}``. ``direction`` is ``improved``
       when posterior drops (the defect is less likely) and
       ``regressed`` when it rises. Filtered to ``|delta| >= threshold``
       to suppress noise.
    * ``top_improvement`` / ``top_regression`` -- the single node with
       the largest absolute shift in each direction (or ``None``).
    * ``metric_deltas`` -- per-metric ``optimized - baseline`` (positive
       = improved for higher-is-better metrics; LOWER_IS_BETTER
       direction not flipped here -- consumers should know per metric).
    * ``summary`` -- single-sentence narrative ("Optimization resolved
       X, but Y regressed; head bottleneck shifted from A to B").
    """
    from ragdx.core.thresholds import LOWER_IS_BETTER

    if not baseline and not optimized:
        return {}

    base_hyp = (baseline or {}).get("hypotheses") or []
    opt_hyp = (optimized or {}).get("hypotheses") or []
    base_keys = {_hyp_key(h): h for h in base_hyp}
    opt_keys = {_hyp_key(h): h for h in opt_hyp}

    resolved = [base_keys[k] for k in base_keys.keys() - opt_keys.keys()]
    persisted = [opt_keys[k] for k in base_keys.keys() & opt_keys.keys()]
    emerged = [opt_keys[k] for k in opt_keys.keys() - base_keys.keys()]

    base_signals = {
        s["node"]: s for s in ((baseline or {}).get("causal_signals") or [])
    }
    opt_signals = {
        s["node"]: s for s in ((optimized or {}).get("causal_signals") or [])
    }
    posterior_shifts: list[dict] = []
    for node in base_signals.keys() | opt_signals.keys():
        bp = float((base_signals.get(node) or {}).get("posterior", 0.0))
        op = float((opt_signals.get(node) or {}).get("posterior", 0.0))
        delta = op - bp
        if abs(delta) < posterior_shift_threshold:
            continue
        posterior_shifts.append({
            "node": node,
            "baseline": round(bp, 4),
            "optimized": round(op, 4),
            "delta": round(delta, 4),
            # ``improved`` = the defect is less likely after optimization
            # (posterior went down). ``regressed`` = posterior went up.
            "direction": "improved" if delta < 0 else "regressed",
        })
    posterior_shifts.sort(key=lambda r: abs(r["delta"]), reverse=True)

    improvements = [r for r in posterior_shifts if r["direction"] == "improved"]
    regressions = [r for r in posterior_shifts if r["direction"] == "regressed"]
    top_improvement = improvements[0] if improvements else None
    top_regression = regressions[0] if regressions else None

    metric_deltas: dict[str, float] = {}
    if baseline_scores and optimized_scores:
        for m in sorted(set(baseline_scores) | set(optimized_scores)):
            bv = baseline_scores.get(m)
            ov = optimized_scores.get(m)
            if isinstance(bv, (int, float)) and isinstance(ov, (int, float)):
                metric_deltas[m] = round(float(ov) - float(bv), 4)

    # One-sentence story for the report header.
    parts: list[str] = []
    if resolved:
        parts.append(f"resolved {len(resolved)} baseline hypothesis(es)")
    if emerged:
        parts.append(f"{len(emerged)} new hypothesis(es) emerged")
    if persisted:
        parts.append(f"{len(persisted)} still persisting")
    if top_improvement:
        parts.append(
            f"top improvement: {top_improvement['node']} "
            f"({top_improvement['baseline']:.2f}→{top_improvement['optimized']:.2f})"
        )
    if top_regression:
        parts.append(
            f"top regression: {top_regression['node']} "
            f"({top_regression['baseline']:.2f}→{top_regression['optimized']:.2f})"
        )
    summary = "; ".join(parts) if parts else "no measurable diagnosis change."

    # Flag noise-level changes ("answer-correctness moved 0.02" isn't a
    # story). We surface raw deltas but a UI consumer can also use this
    # to filter rows. We keep the LOWER_IS_BETTER set here so future
    # consumers don't re-derive direction wrongly.
    return {
        "resolved_hypotheses": resolved,
        "persisted_hypotheses": persisted,
        "emerged_hypotheses": emerged,
        "posterior_shifts": posterior_shifts,
        "top_improvement": top_improvement,
        "top_regression": top_regression,
        "metric_deltas": metric_deltas,
        "lower_is_better_metrics": sorted(LOWER_IS_BETTER),
        "summary": summary,
    }


def _diagnose_per_mode(results_by_mode: dict[str, dict]) -> dict[str, dict]:
    """Diagnose the system BEFORE and AFTER the prompt optimization.

    ``ragdx experiment`` is the one-shot path: BO -> DSPy -> diagnose
    at the end. There is no upfront diagnose ceremony, but the final
    report still needs both sides of the optimization story, so per
    mode this emits the ``{baseline, optimized, comparison}`` triple:

    * ``baseline``   -- the system at the BO winner config with the
       seed prompt (``dspy_a_b.baseline_scores``), i.e. *before*
       prompt tuning.
    * ``optimized``  -- the final system after prompt tuning,
       including the deepeval supplements merged into
       ``optimized_scores``. Always emitted when scores exist, even
       if identical to the baseline -- an honest "no change" beats a
       silently missing section.
    * ``comparison`` -- resolved / persisted / emerged hypotheses,
       posterior shifts, metric deltas.

    Failures are caught and logged -- a missing diagnosis section is
    preferable to a failed bundle write.
    """
    from ragdx.core.diagnosis import RAGDiagnosisEngine

    engine = RAGDiagnosisEngine()
    out: dict[str, dict] = {}
    for mode, payload in results_by_mode.items():
        dspy_ab = payload.get("dspy_a_b") or {}
        baseline_scores = dict(dspy_ab.get("baseline_scores") or {})
        optimized_scores = dict(dspy_ab.get("optimized_scores") or {})
        if not baseline_scores and not optimized_scores:
            # No DSPy stage ran -- fall back to the BO winner as the
            # single post-optimization snapshot.
            optimized_scores = _winner_scores(
                payload.get("bayes_search") or {}
            )

        def _diag(scores: dict, phase: str, _mode: str = mode) -> dict | None:
            if not scores:
                return None
            try:
                res = _synth_eval_result(
                    scores, mode=_mode, extra_metadata={"phase": phase},
                )
                return engine.diagnose(res).model_dump()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "%s diagnosis for mode=%s failed: %s", phase, _mode, exc,
                )
                return None

        b_rep = _diag(baseline_scores, "baseline")
        o_rep = _diag(optimized_scores, "optimized")
        if not (b_rep or o_rep):
            continue
        entry: dict = {"baseline": b_rep, "optimized": o_rep}
        if b_rep and o_rep:
            entry["comparison"] = _compare_diagnoses(
                b_rep, o_rep,
                baseline_scores=baseline_scores,
                optimized_scores=optimized_scores,
            )
        out[mode] = entry
    return out


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
        # Rule-based diagnosis at the BO winner config. Surfaced in
        # the experiment-report HTML so the report tells the reader
        # both "what we tried" (bayes_search) and "what to attack next"
        # (diagnosis). LLM-refined diagnosis is still available via
        # the standalone ``ragdx diagnose`` command.
        "diagnosis": _diagnose_per_mode(results_by_mode),
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
# Experiment checkpointing
# =====================================================================
# One ``ragdx experiment`` run spans up to four long stages (2 modes x
# {joint BO, DSPy generation}), each resumable on its own. We persist
# one Checkpoint per (mode, stage), tied together by an
# ``experiment_group`` id in ``cli_args`` so ``--resume`` can pick the
# whole set back up. Completed stages replay from their checkpoint in
# seconds (zero LLM calls); the interrupted stage continues from its
# last completed trial / phase.
class _ExperimentCheckpoints:
    """Per-(mode, stage) checkpoint registry for one experiment run."""

    def __init__(self, resume: str = "", enabled: bool = True) -> None:
        self.enabled = enabled
        self.store = None
        self.group_id = ""
        self._by_key: dict[tuple[str, str], Any] = {}
        if not enabled:
            return
        from uuid import uuid4

        from ragdx.checkpoint import CheckpointStore
        self.store = CheckpointStore()
        if resume:
            # Group ALL experiment checkpoints (any status: completed
            # stages replay instantly, interrupted ones continue).
            groups: dict[str, list] = {}
            for c in self.store.list():
                if c.kind != "experiment":
                    continue
                g = str(c.cli_args.get("experiment_group") or "")
                if g:
                    groups.setdefault(g, []).append(c)
            resumable = {
                g: cs for g, cs in groups.items()
                if any(c.status != "completed" for c in cs)
            }
            if resume.lower() in {"auto", "latest", "true"}:
                if not resumable:
                    raise ValueError(
                        "No interrupted experiment checkpoint to resume. "
                        "Run `ragdx checkpoints` to see what's stored."
                    )
                self.group_id = max(
                    resumable,
                    key=lambda g: max(c.updated_at for c in resumable[g]),
                )
            else:
                if resume not in groups:
                    raise ValueError(
                        f"No experiment checkpoints for group {resume!r}. "
                        "Run `ragdx checkpoints` to see what's stored."
                    )
                self.group_id = resume
            for c in groups[self.group_id]:
                key = (
                    str(c.cli_args.get("mode") or ""),
                    str(c.cli_args.get("stage") or ""),
                )
                if c.status != "completed":
                    c.status = "running"
                    c.interrupted_reason = ""
                self._by_key[key] = c
            logger.info(
                "Resuming experiment group %s (%d stage checkpoint(s): %s)",
                self.group_id, len(self._by_key),
                ", ".join(f"{m}/{s}" for m, s in self._by_key),
            )
        else:
            self.group_id = "exp_" + uuid4().hex[:8]

    def get(self, mode: str, stage: str, *, name: str = "") -> Any:
        """Fetch (resume) or create the checkpoint for ``(mode, stage)``."""
        if not self.enabled or self.store is None:
            return None
        key = (mode, stage)
        ck = self._by_key.get(key)
        if ck is None:
            from ragdx.checkpoint import Checkpoint
            ck = Checkpoint(
                kind="experiment",
                name=name or f"experiment/{mode}/{stage}",
                cli_args={
                    "experiment_group": self.group_id,
                    "mode": mode,
                    "stage": stage,
                },
            )
            self.store.save(ck)
            self._by_key[key] = ck
        return ck

    def complete(self, ck: Any) -> None:
        """Best-effort completion marker (never sinks the run)."""
        if ck is None or self.store is None:
            return
        try:
            self.store.mark_completed(ck.checkpoint_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("checkpoint finalize skipped: %s", exc)


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
    resume: str = "",
    no_checkpoint: bool = False,
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

    # Checkpointing: one Checkpoint per (mode, stage), grouped by an
    # experiment id. ``resume="auto"`` (or a group id) picks the run
    # back up -- completed stages replay without LLM calls, the
    # interrupted stage continues from its last saved trial / phase.
    ckpts = _ExperimentCheckpoints(resume=resume, enabled=not no_checkpoint)
    if ckpts.enabled and not resume:
        logger.info(
            "Experiment checkpoints enabled (group %s). Resume an "
            "interrupted run with `ragdx experiment ... --resume %s` "
            "(or --resume auto).", ckpts.group_id, ckpts.group_id,
        )

    results_by_mode: dict[str, dict] = {}
    if cfg.mode in ("with_gt", "both"):
        records_gt = [
            DatasetRecord(
                question=r.question, ground_truth=r.ground_truth, contexts=list(r.contexts)
            )
            for r in base_records
        ]
        results_by_mode["with_gt"] = _run_one_mode(
            cfg, runtime, chunks_master, records_gt, "with_gt",
            ckpts=ckpts,
        )
    if cfg.mode in ("no_gt", "both"):
        records_no = [
            DatasetRecord(
                question=r.question, ground_truth=None, contexts=list(r.contexts)
            )
            for r in base_records
        ]
        results_by_mode["no_gt"] = _run_one_mode(
            cfg, runtime, chunks_master, records_no, "no_gt",
            ckpts=ckpts,
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
