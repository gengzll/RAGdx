"""
RAG Diagnosis Engine

Main Idea:
This module provides the main diagnosis engine for RAG pipelines. It orchestrates the analysis of evaluation results to identify performance issues, generate hypotheses about root causes, and provide actionable recommendations.

Functionalities:
- Rule-based diagnosis: Uses predefined rules and thresholds to identify common RAG issues
- LLM-powered diagnosis: Leverages large language models for more nuanced analysis and explanations
- Hybrid diagnosis: Combines rule-based and LLM approaches for comprehensive insights
- Diagnosis report generation: Creates structured reports with hypotheses, priority actions, and confidence scores

The engine supports different diagnosis modes:
- Rule-based only (default): Fast, deterministic analysis
- LLM-only: AI-powered analysis with natural language explanations
- Both: Runs both methods and summarizes with LLM

Usage:
Basic usage with rule-based diagnosis:

    from ragdx.core.diagnosis import RAGDiagnosisEngine

    engine = RAGDiagnosisEngine()
    report = engine.diagnose(evaluation_result)

With LLM support:

    from ragdx.engines.llm_diagnosis import LLMDiagnosisExplainer

    llm_explainer = LLMDiagnosisExplainer(llm_callable=my_llm_function)
    engine = RAGDiagnosisEngine(llm_explainer=llm_explainer)
    report = engine.diagnose(evaluation_result, use_llm=True)

The diagnosis report includes identified issues, confidence levels, and recommended actions.
"""

from __future__ import annotations

from ragdx.engines.llm_diagnosis import LLMDiagnosisExplainer
from ragdx.engines.root_cause import RuleBasedRootCauseAnalyzer
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import DiagnosisReport, EvaluationResult


class RAGDiagnosisEngine:
    def __init__(self, analyzer: RuleBasedRootCauseAnalyzer | None = None, llm_explainer: LLMDiagnosisExplainer | None = None):
        self.analyzer = analyzer or RuleBasedRootCauseAnalyzer()
        self.llm_explainer = llm_explainer
        self.planner = OptimizationPlanner()

    def diagnose(
        self,
        result: EvaluationResult,
        use_llm: bool = False,
        use_both: bool = False,
        optimization_history: list[str] | None = None,
        *,
        learn: bool = True,
    ) -> DiagnosisReport:
        # ``optimization_history`` lets the rule analyzer escalate its
        # recommendations when a defect persists despite the targeting
        # optimization already having run (history-aware diagnosis).
        # ``learn=False`` = report-style diagnosis: don't write the
        # posteriors back to the causal prior store (repeated bundle
        # renders would otherwise saturate the priors).
        rule_report = self.analyzer.analyze(
            result, optimization_history=optimization_history, learn=learn,
        )
        if use_both:
            if self.llm_explainer is None:
                raise ValueError("LLM diagnosis requested but no llm_explainer is configured.")
            llm_report = self.llm_explainer.explain(result, rule_report)
            return self.llm_explainer.summarize_both(result, rule_report, llm_report)
        if use_llm:
            if self.llm_explainer is None:
                raise ValueError("LLM diagnosis requested but no llm_explainer is configured.")
            return self.llm_explainer.explain(result, rule_report)
        return rule_report
