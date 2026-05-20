"""
Evaluation Engines and Adapters

Main Idea:
This package contains adapters and engines for various RAG evaluation tools and diagnosis methods. It provides interfaces to external evaluation libraries and implements specialized diagnosis algorithms.

Functionalities:
- ragas_adapter: Integration with RAGAS evaluation framework
- ragchecker_adapter: Integration with RAGChecker evaluation tool
- llm_diagnosis: LLM-powered diagnosis and explanation engine
- root_cause: Rule-based root cause analysis for RAG issues

Usage:
Adapters are typically used through the UnifiedEvaluator:

    from ragdx.engines.ragas_adapter import RagasAdapter
    from ragdx.engines.ragchecker_adapter import RAGCheckerAdapter
"""