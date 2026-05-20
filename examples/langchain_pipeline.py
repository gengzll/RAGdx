"""
Example LangChain RAG Pipeline

Main Idea:
This example demonstrates how to create a simple RAG (Retrieval-Augmented Generation) pipeline using LangChain. It shows the basic structure for integrating retrieval and generation components in a chain.

Functionalities:
- Document storage: Creates sample documents with metadata
- Retrieval component: Simple demo retriever that returns top-k documents
- Answer generation: Combines question and retrieved contexts into an answer
- Citation tracking: Includes source citations in the response
- Configurable parameters: Supports runtime configuration for retriever_k

Key components:
- DemoRetriever: Mock retriever returning sample documents
- Answer combiner: Generates responses based on retrieved contexts
- Citation extraction: Tracks document sources for verifiability

Usage:
Create and use the pipeline:

    from examples.langchain_pipeline import create_pipeline

    config = {"runtime": {"retriever_k": 2}}
    pipeline = create_pipeline(config)

    result = pipeline.invoke({"input": "What is RAG?"})
    print(result["answer"])  # Demo LangChain answer...

This example is used for testing the LangChain adapter and demonstrating RAG pipeline integration.
"""

from __future__ import annotations

from typing import Any, Dict


def create_pipeline(config: Dict[str, Any]):
    try:
        from langchain_core.documents import Document
        from langchain_core.runnables import RunnableLambda
        from langchain.chains.retrieval import create_retrieval_chain
    except Exception as exc:
        raise RuntimeError("LangChain packages are required for this pipeline. Install the langchain extras.") from exc

    docs = [
        Document(page_content="RAG combines retrieval with generation.", metadata={"source": "demo-1"}),
        Document(page_content="Citations improve verifiability.", metadata={"source": "demo-2"}),
    ]

    # ``create_retrieval_chain`` expects the retriever to be a Runnable
    # (with ``.with_config``), not a plain class with ``invoke``. Wrap a
    # closure in ``RunnableLambda`` so the demo works on langchain >=0.3.
    retriever_k = int(config.get("runtime", {}).get("retriever_k", 2))

    def _retrieve(query):
        # ``create_retrieval_chain`` passes the full inputs dict here; pull
        # the ``input`` field if so, otherwise treat as a plain string.
        if isinstance(query, dict):
            query = query.get("input", "")
        return docs[:retriever_k]

    retriever = RunnableLambda(_retrieve)

    def _combine(inputs: Dict[str, Any]):
        question = inputs.get("input", "")
        contexts = inputs.get("context", [])
        return {
            "answer": f"Demo LangChain answer for: {question} using {len(contexts)} contexts.",
            "citations": [d.metadata.get("source", "unknown") for d in contexts],
            "contexts": [d.page_content for d in contexts],
        }

    return create_retrieval_chain(retriever, RunnableLambda(_combine))
