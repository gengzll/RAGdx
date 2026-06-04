"""Embedding + LLM-rubric metric for DSPy's no-GT inner loop.

Replaces the ragas composite (``context_precision`` + ``faithfulness`` +
``answer_relevancy``) for prompt-tuning stages. Three problems with that
baseline:

* ``context_precision`` scores retrieval, not generation. The prompt
  optimizer never moves it (contexts are fixed at the BO winner config),
  so it's a constant 0.3-weighted term that wastes one LLM call per
  trial.
* ``faithfulness`` is two-step (claim extraction + per-claim NLI) and
  permissive judges (e.g. GLM-4-Flash) saturate it near 1.0 across
  trials -- DSPy then ties on the seed program and the BO degenerates.
* ``answer_relevancy`` is OK but alone produces only one continuous
  signal, and ragas's implementation calls the LLM to paraphrase the
  answer back into candidate questions -- expensive for what is, at
  heart, a cosine.

This metric produces a composite of:

1. ``embed(answer) · embed(joined_contexts)`` -- "answer-context
   grounding proxy". Deterministic, zero LLM calls, moves on every
   answer-wording change.
2. ``embed(answer) · embed(question)`` -- "on-topicness". Same idea as
   ``answer_relevancy`` but skips the LLM paraphrase loop.
3. A single multi-output rubric LLM call returning ``groundedness`` and
   ``completeness`` as scalars in [0, 1]. One call instead of ragas's
   three, and the LM grades two distinct axes so saturation on one
   doesn't collapse the signal.

Cost: ~1 LLM call per ``metric()`` invocation (the rubric); the ragas
composite was ~3 LLM calls. Embedding cost is negligible against a
local HuggingFace model.

The metric is callable as ``metric(example, pred, trace=None) -> float``
for direct use with MIPROv2 / COPRO / BootstrapFewShot. For GEPA the
:mod:`ragdx.optim.dspy_adapter` adapter wraps it into the 5-arg form
GEPA expects, same as for the ragas composite.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any

from ragdx.utils.logging import get_logger

logger = get_logger(__name__)


# Weights chosen so that the embedding-based, deterministic signals
# contribute meaningfully (so the metric never collapses if the rubric
# LLM saturates), and the rubric LLM still dominates when it produces
# discriminative output. Sum is 2.3, matching the order of magnitude
# of the ragas composite (2.8) so downstream baseline-vs-optimized
# comparisons stay readable.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "embed_answer_context": 0.5,
    "embed_answer_question": 0.3,
    "rubric_groundedness": 1.0,
    "rubric_completeness": 0.5,
}


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    # HuggingFaceEmbeddings with ``normalize_embeddings=True`` returns
    # unit vectors so ``dot`` already is the cosine, but we don't rely
    # on the caller setting that flag -- divide explicitly.
    cos = dot / (math.sqrt(na) * math.sqrt(nb))
    # Clip to [0, 1]: cosine of two text embeddings is theoretically
    # [-1, 1] but in practice for short, topically-related strings the
    # observed range is roughly [0.0, 1.0]. Negative cosines from random
    # texts collapse to 0 here rather than penalising into the negative.
    return max(0.0, min(1.0, cos))


def _hash_key(parts: tuple[str, ...]) -> str:
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()


def make_embed_rubric_metric(
    embeddings: Any,
    judge_lm: Any,
    *,
    weights: dict[str, float] | None = None,
) -> Callable[..., float]:
    """Build a DSPy metric callable that scores via embeddings + an LLM rubric.

    Parameters
    ----------
    embeddings:
        A LangChain-compatible embeddings object with ``embed_query(text)``.
        Typically ``runtime.embeddings`` (the same HuggingFace model used
        by FAISS at retrieval time -- intentional, so the metric measures
        "did the answer move closer to the context in the same space the
        retriever uses").
    judge_lm:
        A ``dspy.LM`` instance used by the rubric ``dspy.Predict`` call.
        Typically ``runtime.dspy_lm``. Override with a stronger judge LM
        for production tuning if you want.
    weights:
        Per-component weights. Defaults to :data:`_DEFAULT_WEIGHTS`.
        Unknown component names fall back to ``1.0``.

    Returns
    -------
    A callable ``metric(example, pred, trace=None) -> float``.

    Failure modes -- returns ``0.0``:

    * Empty answer or empty contexts.
    * ``dspy`` import fails (the caller should fall back to the legacy
      ragas / llm_judge metric in that case).

    Rubric calls that raise are logged and contribute ``0.0`` to that
    one component, so a single flaky judge call doesn't kill the trial.
    """
    try:
        import dspy  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise ImportError(
            "Building the embed-rubric DSPy metric requires DSPy. "
            "Install with `pip install ragdx[dspy]`."
        ) from exc

    if weights is None:
        weights = dict(_DEFAULT_WEIGHTS)

    class _Rubric(dspy.Signature):
        """Rate how well the answer is grounded in the context AND
        how completely it addresses the question.

        groundedness: 1.0 if every claim is directly supported by the
        context, 0.0 if the answer invents facts. Partial support = 0.5.

        completeness: 1.0 if the answer fully addresses the question
        given the context, 0.0 if it dodges or only answers a fragment.
        An answer can be fully grounded but incomplete (e.g. "I cannot
        find the answer in the context") -- score those low on
        completeness, high on groundedness.
        """

        question: str = dspy.InputField()
        context: str = dspy.InputField()
        answer: str = dspy.InputField()
        groundedness: float = dspy.OutputField(desc="0.0 to 1.0")
        completeness: float = dspy.OutputField(desc="0.0 to 1.0")

    rubric = dspy.Predict(_Rubric)

    # Cache embeddings of constant inputs (question + contexts) across
    # MIPROv2's threadpool. Same trainset is reused every trial, so the
    # only thing that changes between calls is ``pred.answer`` -- which
    # we don't cache.
    _embed_cache: dict[str, list[float]] = {}

    def _embed(text: str) -> list[float]:
        key = _hash_key((text,))
        cached = _embed_cache.get(key)
        if cached is not None:
            return cached
        try:
            vec = list(embeddings.embed_query(text))
        except Exception as exc:  # pragma: no cover - depends on backend
            logger.debug("embed_rubric: embed_query failed (returning 0): %s", exc)
            return []
        _embed_cache[key] = vec
        return vec

    def _clip01(v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(f):
            return 0.0
        return max(0.0, min(1.0, f))

    def metric(example: Any, pred: Any, trace: Any | None = None) -> float:
        question = str(getattr(example, "question", "") or "")
        ctx = getattr(example, "context", None) or getattr(example, "contexts", "")
        if isinstance(ctx, list):
            contexts = [str(c) for c in ctx if c]
        else:
            contexts = [str(ctx)] if ctx else []
        answer = str(getattr(pred, "answer", "") or "")

        if not answer or not contexts:
            return 0.0

        joined_ctx = "\n\n".join(contexts)

        # ---- Component 1 + 2: deterministic embedding cosines ----
        ans_vec = _embed(answer)
        ctx_vec = _embed(joined_ctx)
        q_vec = _embed(question)
        cos_ac = _cosine(ans_vec, ctx_vec)
        cos_aq = _cosine(ans_vec, q_vec)

        # ---- Component 3 + 4: single rubric LLM call ----
        grounded = 0.0
        complete = 0.0
        try:
            with dspy.context(lm=judge_lm):
                out = rubric(question=question, context=joined_ctx, answer=answer)
            grounded = _clip01(getattr(out, "groundedness", 0.0))
            complete = _clip01(getattr(out, "completeness", 0.0))
        except Exception as exc:  # pragma: no cover - depends on judge LM
            logger.debug(
                "embed_rubric: rubric LLM call failed (components -> 0): %s", exc
            )

        score = (
            cos_ac * weights.get("embed_answer_context", 0.0)
            + cos_aq * weights.get("embed_answer_question", 0.0)
            + grounded * weights.get("rubric_groundedness", 0.0)
            + complete * weights.get("rubric_completeness", 0.0)
        )
        return float(score)

    metric.__ragdx_kind__ = "embed_rubric"  # type: ignore[attr-defined]
    metric.__ragdx_weights__ = dict(weights)  # type: ignore[attr-defined]
    return metric


__all__ = ["make_embed_rubric_metric"]
