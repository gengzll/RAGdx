"""Real LangChain RAG pipeline backed by FAISS + sentence-transformers.

Unlike ``examples.langchain_pipeline`` (which uses 2 hardcoded docs and a
template answer), this pipeline:

- Loads a corpus from a JSONL file (one ``{"text": ..., "source": ...}``
  per line) pointed to by ``cfg["search_parameters"]["corpus_path"]``.
- Builds a FAISS vector index using ``sentence-transformers/all-MiniLM-L6-v2``
  (small, ~80MB, runs on CPU, no API key).
- Retrieves top ``cfg["runtime"]["retriever_k"]`` chunks.
- Synthesizes a "structured extractive" answer by concatenating the
  retrieved chunks with bracketed citations. No LLM API key required.

This means varying ``top_k`` produces genuinely different retrieval
sets *and* different answers, so the optimization metrics carry signal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def create_pipeline(config: Dict[str, Any]):
    try:
        from langchain_core.documents import Document
        from langchain_core.runnables import RunnableLambda
        from langchain.chains.retrieval import create_retrieval_chain
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception:
        # Fall back to the older community embeddings module if the
        # langchain-huggingface package is not installed.
        from langchain_core.documents import Document
        from langchain_core.runnables import RunnableLambda
        from langchain.chains.retrieval import create_retrieval_chain
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

    search_params = config.get("search_parameters", {}) or {}
    runtime = config.get("runtime", {}) or {}

    corpus_path = search_params.get("corpus_path")
    if not corpus_path:
        raise RuntimeError(
            "langchain_pipeline_real requires search_parameters.corpus_path "
            "to point at a JSONL corpus (one {text, source} per line)."
        )
    k = int(runtime.get("retriever_k", 3))
    model_name = search_params.get(
        "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load corpus.
    docs = []
    for line in Path(corpus_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        docs.append(
            Document(
                page_content=row["text"],
                metadata={"source": row.get("source", "unknown")},
            )
        )
    if not docs:
        raise RuntimeError(f"Corpus {corpus_path} is empty.")

    # Build FAISS index. Note: don't pass show_progress_bar here — both
    # langchain_community and langchain_huggingface inject it internally,
    # which causes a "multiple values" TypeError on sentence-transformers 5+.
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = FAISS.from_documents(docs, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": k})

    def _combine(inputs: Dict[str, Any]):
        question = inputs.get("input", "")
        contexts = inputs.get("context", [])
        # Extractive synthesis: concatenate chunks with [n] tags.
        pieces = []
        citations = []
        for i, d in enumerate(contexts, start=1):
            pieces.append(f"[{i}] {d.page_content.strip()}")
            citations.append(d.metadata.get("source", f"chunk-{i}"))
        synth = (
            f"Question: {question}\nAnswer (extracted from {len(contexts)} "
            f"retrieved passages):\n" + "\n".join(pieces)
        )
        return {
            "answer": synth,
            "citations": citations,
            "contexts": [d.page_content for d in contexts],
        }

    return create_retrieval_chain(retriever, RunnableLambda(_combine))
