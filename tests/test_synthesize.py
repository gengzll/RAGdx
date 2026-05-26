"""Tests for ragdx.datasets.synthesize.

Uses a stub LLM callable so no real LLM is required.
"""

from __future__ import annotations

import random

import pytest

from ragdx.datasets.synthesize import (
    SynthesizedQuestion,
    _extract_question,
    synthesize_questions,
)


# ---------------------------------------------------------------- regex
def test_extract_question_handles_clean_response():
    assert _extract_question("What is RAG?") == "What is RAG?"


def test_extract_question_strips_common_prefixes():
    assert _extract_question("Question: What is RAG?") == "What is RAG?"
    assert _extract_question("Q: What is RAG?") == "What is RAG?"
    assert _extract_question("**Question**: What is RAG?") == "What is RAG?"


def test_extract_question_returns_none_for_garbage():
    assert _extract_question("") is None
    assert _extract_question("no question mark here") is None
    assert _extract_question("?") is None  # too short
    assert _extract_question(None) is None  # type: ignore[arg-type]


def test_extract_question_takes_first_question_when_multiple():
    raw = "What is RAG? Also, what is DSPy?"
    assert _extract_question(raw) == "What is RAG?"


# --------------------------------------------------------- synthesize core
def test_synthesize_questions_happy_path():
    corpus = [
        "RAG uses a retriever and a generator.",
        "Chunking splits documents into smaller pieces.",
        "Embeddings encode text into vectors.",
    ]
    counter = {"n": 0}

    def fake_llm(prompt: str) -> str:
        counter["n"] += 1
        return f"What does passage {counter['n']} describe?"

    qs = synthesize_questions(corpus, n=3, llm_callable=fake_llm, rng=random.Random(0))
    assert len(qs) == 3
    assert all(isinstance(q, SynthesizedQuestion) for q in qs)
    assert all(q.question.endswith("?") for q in qs)


def test_synthesize_questions_retries_on_garbage_llm():
    corpus = ["doc a", "doc b", "doc c", "doc d"]
    calls = {"n": 0}

    def flaky_llm(prompt: str) -> str:
        calls["n"] += 1
        # Fail the first attempt, succeed on retry.
        if calls["n"] == 1:
            return "no question here"
        return "What does this corpus discuss?"

    qs = synthesize_questions(
        corpus, n=1, llm_callable=flaky_llm,
        max_retries_per_question=3, rng=random.Random(0),
    )
    assert len(qs) == 1
    assert qs[0].question == "What does this corpus discuss?"


def test_synthesize_questions_skips_when_llm_always_fails():
    corpus = ["doc a", "doc b"]

    def bad_llm(prompt: str) -> str:
        return "no question mark"

    qs = synthesize_questions(
        corpus, n=2, llm_callable=bad_llm,
        max_retries_per_question=2, rng=random.Random(0),
    )
    assert qs == []


def test_synthesize_questions_passes_chunk_ids_back():
    corpus = ["alpha", "beta", "gamma", "delta", "epsilon"]

    def echo_llm(prompt: str) -> str:
        return "What is in the passages?"

    qs = synthesize_questions(
        corpus, n=2, llm_callable=echo_llm,
        chunks_per_question=3, rng=random.Random(0),
    )
    assert all(len(q.source_chunk_ids) == 3 for q in qs)
    # all ids in range
    for q in qs:
        for cid in q.source_chunk_ids:
            assert 0 <= cid < len(corpus)


# --------------------------------------------------------- validation
def test_synthesize_questions_rejects_empty_corpus():
    with pytest.raises(ValueError, match="non-empty"):
        synthesize_questions([], n=1, llm_callable=lambda _: "Q?")


def test_synthesize_questions_rejects_bad_n():
    with pytest.raises(ValueError, match="positive"):
        synthesize_questions(["doc"], n=0, llm_callable=lambda _: "Q?")


def test_synthesize_questions_rejects_bad_chunks_per_question():
    with pytest.raises(ValueError, match="chunks_per_question"):
        synthesize_questions(["doc"], n=1, llm_callable=lambda _: "Q?", chunks_per_question=0)


def test_synthesize_questions_caps_chunks_per_question_to_corpus_size():
    corpus = ["only one chunk"]

    def llm(prompt: str) -> str:
        return "What is in the passage?"

    qs = synthesize_questions(corpus, n=1, llm_callable=llm, chunks_per_question=10)
    assert len(qs) == 1
    assert qs[0].source_chunk_ids == [0]
