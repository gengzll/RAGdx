"""
Optimization Components

Main Idea:
This package contains optimization algorithms and adapters for RAG pipeline tuning. It provides Bayesian optimization, LLM-based planning, and integration with popular RAG frameworks for automated parameter optimization.

Functionalities:
- planner: Optimization planning with rule-based and LLM-enhanced strategies
- executor: Execution engine for optimization experiments
- Adapters for RAG frameworks: LangChain, LlamaIndex, DSPy, AutoRAG
- Bayesian optimization: Heavy-duty Bayesian optimization for complex parameter spaces

Usage:
Import optimization components:

    from ragdx.optim.planner import OptimizationPlanner
    from ragdx.optim.executor import OptimizationExecutor
"""