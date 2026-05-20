"""
Example LlamaIndex RAG Pipeline

Main Idea:
This example demonstrates how to create a simple RAG (Retrieval-Augmented Generation) pipeline using LlamaIndex. It shows the basic setup for document indexing and query processing.

Functionalities:
- Document indexing: Creates vector store index from sample documents
- Query engine: Configures similarity-based retrieval and generation
- Configurable retrieval: Supports top_k parameter for retrieval count
- Vector search: Uses embedding-based similarity for document retrieval

Key components:
- Document creation: Converts text content into LlamaIndex documents
- VectorStoreIndex: Builds searchable vector index from documents
- Query engine: Provides retrieval-augmented question answering

Usage:
Create and use the query engine:

    from examples.llamaindex_pipeline import create_query_engine

    config = {"runtime": {"retriever_top_k": 2}}
    engine = create_query_engine(config)

    response = engine.query("What is RAG?")
    print(response.response)  # Answer based on retrieved documents

This example is used for testing the LlamaIndex adapter and demonstrating vector-based RAG pipelines.
"""

from __future__ import annotations

from typing import Any, Dict


def create_query_engine(config: Dict[str, Any]):
    try:
        from llama_index.core import Document, VectorStoreIndex
    except Exception as exc:
        raise RuntimeError("LlamaIndex packages are required for this pipeline. Install the llamaindex extras.") from exc

    docs = [Document(text="RAG combines retrieval with generation."), Document(text="Citations improve verifiability.")]
    index = VectorStoreIndex.from_documents(docs)
    top_k = int(config.get("runtime", {}).get("retriever_top_k", 2))
    return index.as_query_engine(similarity_top_k=top_k)
