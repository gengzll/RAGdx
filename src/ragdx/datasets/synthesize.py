"""LLM-driven synthetic question generation for cold-start RAG eval.

The classic "I just have a corpus, no labelled QA pairs" problem. Given
some source chunks, ask an LLM to generate diverse, answerable questions
grounded in the text. Useful as the seed set for a no-GT optimization
pipeline.

We *don't* generate ground-truth answers here because the demo's
philosophy is "no-GT means no GT" — the resulting questions get fed into
your RAG + scored with reference-free metrics. If you want pseudo-GT
answers, run the synthesized questions through your strongest LLM as a
separate step.

Example::

    from ragdx.datasets import synthesize_questions

    qs = synthesize_questions(
        corpus_chunks=corpus,
        n=8,
        llm_callable=my_glm_callable,
        chunks_per_question=2,
    )
    for q in qs:
        print(q.question, "  ⟵ from", q.source_chunk_ids)
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SynthesizedQuestion:
    """One generated question + the chunk IDs it was drawn from."""

    question: str
    source_chunk_ids: list[int]
    source_text_preview: str


# Prompt template kept simple + readable. The LLM is instructed to emit
# exactly one factual question per call, grounded in the supplied text.
_PROMPT_TEMPLATE = """\
You are creating diverse, factual questions for a retrieval-augmented
generation test set.

Read the following passages carefully. Then write ONE clear, specific
question whose answer is fully contained in the passages. The question
must:
  * be a single sentence ending in '?'
  * ask for a fact, definition, list, or comparison stated in the text
  * NOT require outside knowledge
  * NOT begin with 'Based on the text' or 'According to the passage'

Passages:
---
{passages}
---

Question:"""


_QUESTION_RE = re.compile(r"^[\s]*(?P<q>.+?\?)", re.DOTALL)


def _extract_question(raw: str) -> str | None:
    """Pull a single question out of the LLM response, tolerant of fluff."""
    if not raw:
        return None
    # Drop common prefaces the LLM sometimes adds.
    cleaned = raw.strip()
    for prefix in ("Question:", "Q:", "Q.", "**Question**:"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    match = _QUESTION_RE.search(cleaned)
    if not match:
        return None
    q = match.group("q").strip()
    if len(q) < 8 or "?" not in q:
        return None
    return q


def synthesize_questions(
    corpus_chunks: Sequence[str],
    *,
    n: int,
    llm_callable: Callable[[str], str],
    chunks_per_question: int = 2,
    max_passages_chars: int = 1500,
    rng: random.Random | None = None,
    max_retries_per_question: int = 2,
) -> list[SynthesizedQuestion]:
    """Generate ``n`` synthetic questions from random corpus chunks.

    Parameters
    ----------
    corpus_chunks:
        Source chunks (typically from a PDF / text loader). Must be
        non-empty.
    n:
        How many questions to generate.
    llm_callable:
        ``prompt -> response_text``. Whatever LLM you want (GLM, GPT, etc.).
    chunks_per_question:
        How many chunks to combine for each prompt. 1 produces narrow
        questions; 2-3 produces multi-context questions that exercise
        retrieval more thoroughly.
    max_passages_chars:
        Cap total chars sent to the LLM per call (prevents giant prompts
        for long-chunk corpora). Truncates from the end.
    rng:
        Optional Random for deterministic chunk selection in tests.
    max_retries_per_question:
        If the LLM returns garbage, retry up to this many times with a new
        chunk sample before skipping the slot.
    """
    if not corpus_chunks:
        raise ValueError("corpus_chunks must be non-empty")
    if n <= 0:
        raise ValueError("n must be positive")
    if chunks_per_question < 1:
        raise ValueError("chunks_per_question must be >= 1")

    rng = rng or random.Random(7)
    k = min(chunks_per_question, len(corpus_chunks))

    out: list[SynthesizedQuestion] = []
    for _ in range(n):
        for _attempt in range(max_retries_per_question + 1):
            chunk_ids = rng.sample(range(len(corpus_chunks)), k)
            passages = "\n\n---\n\n".join(corpus_chunks[i] for i in chunk_ids)
            if len(passages) > max_passages_chars:
                passages = passages[:max_passages_chars]
            prompt = _PROMPT_TEMPLATE.format(passages=passages)

            try:
                raw = llm_callable(prompt)
            except Exception as exc:  # pragma: no cover - depends on live LLM
                raw = f"<error: {exc}>"
            question = _extract_question(raw)
            if question:
                out.append(
                    SynthesizedQuestion(
                        question=question,
                        source_chunk_ids=chunk_ids,
                        source_text_preview=passages[:200],
                    )
                )
                break
            # else: retry with a fresh sample
    return out


__all__ = ["SynthesizedQuestion", "synthesize_questions"]
