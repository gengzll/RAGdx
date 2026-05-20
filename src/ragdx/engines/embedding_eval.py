"""Local, dependency-free embedding evaluator.

This evaluator produces a deterministic :class:`EvaluationResult` from a
list of :class:`DatasetRecord` instances using a pure-Python TF-IDF +
cosine-similarity model. It is intended for:

* CI / quickstart smoke tests that should run without any LLM credentials.
* Offline regression checks where a coarse but stable signal is preferable
  to a faked metric.
* Local triage of dataset-quality regressions in a fully air-gapped env.

The metrics produced are *proxies* — they are correlated with real RAG
evaluation metrics but are not interchangeable with them. The output's
``metadata["mode"]`` is set to ``"embedding-proxy"`` and the evaluator
name is recorded so downstream consumers can decide how much weight to
assign.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

from ragdx.schemas.models import DatasetRecord, EvaluationResult

# ``\w`` with the default UNICODE flag matches letters from any script
# (Latin, CJK, Cyrillic, ...) plus digits and underscore. CJK characters
# have no whitespace boundary so they are emitted as a single token; that
# is coarse but still produces a non-zero overlap signal for multilingual
# corpora, which the previous ASCII-only regex did not.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _tf(tokens: Sequence[str]) -> Counter:
    return Counter(tokens)


def _cosine(a: Counter, b: Counter, idf: dict) -> float:
    if not a or not b:
        return 0.0
    # Weighted vectors (tf * idf)
    def _w(counter: Counter) -> dict:
        return {tok: count * idf.get(tok, 0.0) for tok, count in counter.items()}

    wa = _w(a)
    wb = _w(b)
    dot = sum(wa.get(t, 0.0) * wb.get(t, 0.0) for t in wa.keys() & wb.keys())
    if dot == 0.0:
        return 0.0
    na = math.sqrt(sum(v * v for v in wa.values()))
    nb = math.sqrt(sum(v * v for v in wb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _build_idf(documents: Sequence[Sequence[str]]) -> dict:
    """Compute smoothed inverse document frequency over a corpus of token lists."""

    n_docs = len(documents) or 1
    df: Counter = Counter()
    for doc in documents:
        for tok in set(doc):
            df[tok] += 1
    return {tok: math.log((n_docs + 1) / (count + 1)) + 1.0 for tok, count in df.items()}


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


class EmbeddingEvaluator:
    """Compute RAG metric proxies from token-overlap cosine similarity.

    Parameters
    ----------
    min_tokens:
        Records whose question yields fewer than ``min_tokens`` tokens are
        silently skipped (returning 0 contribution). Defaults to 1.
    """

    name = "embedding-proxy"

    def __init__(self, *, min_tokens: int = 1) -> None:
        self.min_tokens = max(1, int(min_tokens))

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def evaluate(self, records: Iterable[DatasetRecord]) -> EvaluationResult:
        records = list(records)
        if not records:
            return EvaluationResult(
                metadata={
                    "tool": self.name,
                    "mode": "embedding-proxy",
                    "record_count": 0,
                    "note": "Empty dataset — no metrics computed.",
                }
            )

        # Build a shared IDF table over every text the evaluator sees.
        corpus: list[list[str]] = []
        for r in records:
            corpus.append(_tokenize(r.question))
            if r.answer:
                corpus.append(_tokenize(r.answer))
            if r.ground_truth:
                corpus.append(_tokenize(r.ground_truth))
            for c in r.contexts:
                corpus.append(_tokenize(c))
            for c in r.reference_contexts:
                corpus.append(_tokenize(c))
        idf = _build_idf(corpus)

        retrieval_ctx_precision: list[float] = []
        retrieval_ctx_recall: list[float] = []
        generation_faithfulness: list[float] = []
        generation_relevancy: list[float] = []
        e2e_correctness: list[float] = []

        for r in records:
            q_tokens = _tokenize(r.question)
            if len(q_tokens) < self.min_tokens:
                continue
            q_tf = _tf(q_tokens)
            ans_tf = _tf(_tokenize(r.answer or ""))
            gt_tf = _tf(_tokenize(r.ground_truth or ""))
            ctx_tfs = [_tf(_tokenize(c)) for c in r.contexts]
            ref_tfs = [_tf(_tokenize(c)) for c in r.reference_contexts]

            # Retrieval — precision is mean(ctx<>question), recall is
            # max(ctx<>ground_truth) when ground truth is available, else
            # mean(ctx<>reference_contexts) when those are available.
            if ctx_tfs:
                retrieval_ctx_precision.append(
                    _mean(_cosine(c, q_tf, idf) for c in ctx_tfs)
                )
                if gt_tf:
                    retrieval_ctx_recall.append(
                        max((_cosine(c, gt_tf, idf) for c in ctx_tfs), default=0.0)
                    )
                elif ref_tfs:
                    # mean over reference contexts of best-matching retrieved ctx
                    retrieval_ctx_recall.append(
                        _mean(
                            max((_cosine(c, ref, idf) for c in ctx_tfs), default=0.0)
                            for ref in ref_tfs
                        )
                    )

            # Generation — faithfulness ~ how well answer is supported by
            # retrieved context (max similarity), relevancy ~ answer<>question.
            if ans_tf and ctx_tfs:
                generation_faithfulness.append(
                    max((_cosine(ans_tf, c, idf) for c in ctx_tfs), default=0.0)
                )
            if ans_tf and q_tf:
                generation_relevancy.append(_cosine(ans_tf, q_tf, idf))

            # End-to-end — answer<>ground_truth similarity.
            if ans_tf and gt_tf:
                e2e_correctness.append(_cosine(ans_tf, gt_tf, idf))

        retrieval: dict = {}
        if retrieval_ctx_precision:
            retrieval["context_precision"] = _mean(retrieval_ctx_precision)
        if retrieval_ctx_recall:
            retrieval["context_recall"] = _mean(retrieval_ctx_recall)

        generation: dict = {}
        if generation_faithfulness:
            generation["faithfulness"] = _mean(generation_faithfulness)
        if generation_relevancy:
            generation["response_relevancy"] = _mean(generation_relevancy)

        e2e: dict = {}
        if e2e_correctness:
            e2e["answer_correctness"] = _mean(e2e_correctness)

        return EvaluationResult(
            retrieval=retrieval,
            generation=generation,
            e2e=e2e,
            metadata={
                "tool": self.name,
                "mode": "embedding-proxy",
                "record_count": len(records),
                "note": (
                    "Metrics computed via TF-IDF cosine similarity. They are "
                    "proxies for the conventional ragas/RAGChecker metrics — "
                    "treat as relative-quality signal, not absolute ground truth."
                ),
            },
            raw_tool_outputs={
                "embedding-proxy": {
                    "retrieval": retrieval,
                    "generation": generation,
                    "e2e": e2e,
                }
            },
        )


__all__ = ["EmbeddingEvaluator"]
