"""
Optimization Components

This package contains the optimization pieces used by the end-to-end
``experiment`` pipeline:

- ``planner``: rule-based / LLM-enhanced optimization planning
- ``bayes_search``: Bayesian search over RAG-config axes
- ``dspy_adapter``: DSPy prompt optimization (MIPROv2 / GEPA)
- ``objectives``: the composite scoring objective
- ``stages``: chunking / retrieval / generation / joint stage optimizers

Usage::

    from ragdx.optim.planner import OptimizationPlanner
    from ragdx.optim.bayes_search import BayesianSearch
"""
