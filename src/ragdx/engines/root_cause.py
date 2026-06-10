"""
Rule-Based Root Cause Analysis Engine

Main Idea:
This module implements a rule-based root cause analysis system for RAG pipeline diagnosis. It uses predefined causal graphs, metric thresholds, and probabilistic reasoning to identify the most likely causes of performance issues.

Functionalities:
- Threshold-based gap analysis: Identifies metrics falling below acceptable thresholds
- Causal graph reasoning: Uses NetworkX-based causal graphs to propagate probabilities
- Hypothesis generation: Creates structured hypotheses with severity, confidence, and evidence
- Bayesian updating: Updates prior probabilities based on observed metric gaps
- Action prioritization: Recommends remediation actions in order of expected leverage

Key components analyzed:
- Corpus chunking defects
- Retrieval recall/precision issues
- Context packing problems
- Grounding failures
- Citation binding issues
- Evaluator instability
- Distribution shifts

Usage:
Basic analysis:

    from ragdx.engines.root_cause import RuleBasedRootCauseAnalyzer

    analyzer = RuleBasedRootCauseAnalyzer()
    report = analyzer.analyze(evaluation_result)

With custom thresholds:

    custom_thresholds = {"faithfulness": 0.95, "context_precision": 0.85}
    analyzer = RuleBasedRootCauseAnalyzer(thresholds=custom_thresholds)
    report = analyzer.analyze(evaluation_result)

The analyzer produces diagnosis reports with prioritized hypotheses and recommended actions.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean

import networkx as nx

from ragdx.core.thresholds import DEFAULT_THRESHOLDS, LOWER_IS_BETTER
from ragdx.schemas.models import (
    CausalEdge,
    CausalGraph,
    CausalSignal,
    DiagnosisHypothesis,
    DiagnosisReport,
    EvaluationResult,
)
from ragdx.storage.run_store import RunStore


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(0.999, max(0.001, p))
    return math.log(p / (1.0 - p))


# =====================================================================
# Escalation ladders (history-aware diagnosis)
# =====================================================================
# When a defect is STILL below threshold *after* the optimization that
# targets it was already tried, repeating the same advice ("add a
# reranker") is useless. Each ladder escalates: level 0 is the first
# attempt, level 1 assumes the basic fix was tried and didn't land,
# level 2 is the "the obvious lever is exhausted, change the approach"
# tier. The level is chosen by how many times the defect's targeting
# optimization already appears in the run history.
#
# Each entry: {"actions": [...], "candidates": [...], "note": str}.
_ESCALATION_LADDERS: dict[str, list[dict]] = {
    'retrieval_precision_defect': [
        {
            'actions': ['Add or tune a reranker.', 'Reduce top-k, or separate the recall stage from the final evidence stage.', 'Use metadata or section-aware filters.'],
            'candidates': ['autorag_pipeline_search'],
            'note': '',
        },
        {
            'actions': ['Basic precision tuning already ran and precision is still low — escalate: add a cross-encoder reranker (not just a score cutoff).', 'Switch to hybrid retrieval (BM25 + dense) so lexical precision complements semantic recall.', 'Add query rewriting / decomposition to sharpen the retrieval target.'],
            'candidates': ['autorag_pipeline_search', 'corpus_chunking_search'],
            'note': 'Retrieval BO already attempted; raising retrieval complexity.',
        },
        {
            'actions': ['Precision is stuck after repeated retrieval tuning — move to the corpus level: re-chunk with section-aware / semantic splitting so chunks carry cleaner single-topic evidence.', 'Add metadata filters to constrain the candidate pool before ranking.', 'Audit chunk boundaries against the actual question set — the corpus may lack the granularity this query distribution needs.'],
            'candidates': ['corpus_chunking_search', 'joint_ablation_eval'],
            'note': 'Retrieval levers exhausted; escalating to corpus chunking + ablation.',
        },
    ],
    'retrieval_recall_defect': [
        {
            'actions': ['Increase recall with hybrid retrieval or a larger candidate pool before reranking.', 'Tune chunk size, overlap, and document segmentation.', 'Inspect query rewriting and metadata filters.'],
            'candidates': ['autorag_pipeline_search', 'corpus_chunking_search'],
            'note': '',
        },
        {
            'actions': ['Recall tuning already ran and recall is still low — escalate: multi-query expansion or HyDE to cast a wider net.', 'Increase top_k substantially and let a reranker prune the noise.', 'Re-chunk smaller so relevant spans are not buried inside large chunks.'],
            'candidates': ['autorag_pipeline_search', 'corpus_chunking_search'],
            'note': 'Retrieval BO already attempted; widening the retrieval net.',
        },
        {
            'actions': ['Recall is stuck after repeated tuning — the evidence may simply not be in the corpus, or chunking is destroying it. Audit a sample of failing questions against the raw document.', 'Try parent-document or sentence-window retrieval to preserve context around hits.'],
            'candidates': ['corpus_chunking_search', 'joint_ablation_eval'],
            'note': 'Retrieval levers exhausted; suspect corpus coverage.',
        },
    ],
    'grounding_defect': [
        {
            'actions': ['Optimize the synthesis prompt to require evidence-backed answers.', 'Use structured answer templates with citation requirements.', 'Add answer compression or a claim-verification stage.'],
            'candidates': ['dspy_prompt_optimization'],
            'note': '',
        },
        {
            'actions': ['Prompt tuning already ran and grounding is still weak — escalate: add a claim-level self-check / verification pass after generation.', 'Constrain the answer to quoted evidence only.', 'Try a stronger generator LM; prompt optimization alone has plateaued.'],
            'candidates': ['dspy_prompt_optimization', 'joint_ablation_eval'],
            'note': 'Prompt optimization already attempted; adding verification / stronger LM.',
        },
        {
            'actions': ['Grounding is stuck after repeated prompt tuning — the upstream retrieval is likely feeding noise. Re-check precision first, then revisit generation.', 'Consider a retrieve-then-verify architecture where each claim is checked against a retrieved span.'],
            'candidates': ['joint_ablation_eval'],
            'note': 'Generation levers exhausted; suspect upstream retrieval noise.',
        },
    ],
}

# Which optimization candidate "targets" each defect. Used to count how
# many times the relevant lever was already pulled when picking the
# escalation level.
_DEFECT_TARGETING_CANDIDATES: dict[str, set[str]] = {
    'retrieval_precision_defect': {'autorag_pipeline_search'},
    'retrieval_recall_defect': {'autorag_pipeline_search', 'corpus_chunking_search'},
    'grounding_defect': {'dspy_prompt_optimization'},
}


def _escalation_level(defect: str, applied_candidates: list[str]) -> int:
    """How far up the ladder to climb for ``defect``.

    Counts how many of the applied optimization candidates target this
    defect; that count (clamped to the ladder length) is the level. A
    defect with no prior attempts stays at level 0 (the base advice).
    """
    ladder = _ESCALATION_LADDERS.get(defect)
    if not ladder:
        return 0
    targeting = _DEFECT_TARGETING_CANDIDATES.get(defect, set())
    tries = sum(1 for c in applied_candidates if c in targeting)
    return max(0, min(tries, len(ladder) - 1))


class RuleBasedRootCauseAnalyzer:
    def __init__(self, thresholds: dict[str, float] | None = None, root: str | None = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS.copy()
        # Resolve the storage root via the same settings the rest of
        # ragdx uses so ``--project`` / ``RAGDX_PROJECT`` / ``RAGDX_ROOT``
        # actually isolate causal priors per project. Without this, every
        # diagnose call writes back to the global ``.ragdx/causal/priors.json``
        # regardless of --project, so posteriors saturate to the 0.95
        # clamp across all projects (observed in demo_diagnosis run on
        # 2026-06-04: all 8 nodes prior=0.95 in a "fresh" project).
        if root is None:
            try:
                from ragdx.config import get_settings
                root = str(get_settings().storage.root)
            except Exception:  # pragma: no cover - defensive
                root = '.ragdx'
        self.store = RunStore(root)
        self.base_priors = {
            'corpus_chunking_defect': 0.10,
            'retrieval_recall_defect': 0.20,
            'retrieval_precision_defect': 0.18,
            'context_packing_defect': 0.10,
            'grounding_defect': 0.18,
            'citation_binding_defect': 0.10,
            'judge_or_metric_instability': 0.06,
            'distribution_shift': 0.08,
        }
        self.node_components = {
            'corpus_chunking_defect': 'retrieval',
            'retrieval_recall_defect': 'retrieval',
            'retrieval_precision_defect': 'retrieval',
            'context_packing_defect': 'generation',
            'grounding_defect': 'generation',
            'citation_binding_defect': 'e2e',
            'judge_or_metric_instability': 'pipeline',
            'distribution_shift': 'pipeline',
        }
        self.node_actions = {
            'corpus_chunking_defect': 'Run parser and chunking search before retriever tuning.',
            'retrieval_recall_defect': 'Expand recall with hybrid retrieval, query rewriting, and larger candidate pools.',
            'retrieval_precision_defect': 'Strengthen reranking, filtering, and precision-oriented retrieval settings.',
            'context_packing_defect': 'Tune context packing order, context budget, and section-aware packing.',
            'grounding_defect': 'Use grounded answer templates, verification, and citation-first prompting.',
            'citation_binding_defect': 'Bind citations at sentence or claim level and return passage ids.',
            'judge_or_metric_instability': 'Audit evaluator prompts, compare against human labels, and calibrate judges.',
            'distribution_shift': 'Cluster failing traffic, review domain shift, and expand the benchmark.',
        }
        self.causal_edges = [
            CausalEdge(source='corpus_chunking_defect', target='retrieval_recall_defect', weight=0.55, rationale='Poor parsing and chunk boundaries directly suppress recall.'),
            CausalEdge(source='retrieval_recall_defect', target='context_packing_defect', weight=0.22, rationale='Weak recall increases packing pressure on partial evidence.'),
            CausalEdge(source='retrieval_precision_defect', target='context_packing_defect', weight=0.18, rationale='Noisy retrieval degrades packed context quality.'),
            CausalEdge(source='retrieval_precision_defect', target='grounding_defect', weight=0.45, rationale='Noisy retrieval raises unsupported reasoning risk.'),
            CausalEdge(source='context_packing_defect', target='grounding_defect', weight=0.40, rationale='Poor packing directly weakens grounding.'),
            CausalEdge(source='grounding_defect', target='citation_binding_defect', weight=0.30, rationale='Unsupported claims often come with weak citation mapping.'),
            CausalEdge(source='judge_or_metric_instability', target='distribution_shift', weight=0.10, rationale='Evaluator disagreement can indicate shift or judge mismatch.'),
            CausalEdge(source='distribution_shift', target='retrieval_recall_defect', weight=0.30, rationale='Shifted traffic often first breaks evidence coverage.'),
            CausalEdge(source='distribution_shift', target='grounding_defect', weight=0.18, rationale='Shifted domains also weaken grounding reliability.'),
        ]
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.base_priors.keys())
        for edge in self.causal_edges:
            self.graph.add_edge(edge.source, edge.target, weight=edge.weight, rationale=edge.rationale)

    def _gap(self, metric: str, value: float) -> float:
        target = self.thresholds.get(metric)
        if target is None:
            return 0.0
        if metric in LOWER_IS_BETTER:
            return round(max(0.0, value - target), 4)
        return round(max(0.0, target - value), 4)

    def _metric_gaps(self, result: EvaluationResult) -> dict[str, float]:
        gaps: dict[str, float] = {}
        for bucket in (result.retrieval, result.generation, result.e2e):
            for metric, value in bucket.items():
                gap = self._gap(metric, value)
                if gap > 0:
                    gaps[metric] = gap
        return dict(sorted(gaps.items(), key=lambda kv: kv[1], reverse=True))

    def _agreement_map(self, result: EvaluationResult) -> dict[str, float]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for item in result.evaluator_scores:
            grouped[item.metric].append(item.score)
        agreement = {c.metric: c.agreement_score for c in result.calibrations}
        for metric, vals in grouped.items():
            if len(vals) <= 1:
                agreement.setdefault(metric, 1.0)
            else:
                spread = max(vals) - min(vals)
                agreement.setdefault(metric, round(max(0.0, 1.0 - spread), 4))
        return agreement

    def _trace_summary(self, result: EvaluationResult) -> dict[str, float]:
        if not result.traces:
            return {}
        chunk_counts = [len(t.retrieved_chunks) for t in result.traces if t.retrieved_chunks is not None]
        cited_counts = [len(t.citations) for t in result.traces if t.citations is not None]
        latency = [t.latency_ms for t in result.traces if t.latency_ms is not None]
        costs = [t.cost_usd for t in result.traces if t.cost_usd is not None]
        answerless = sum(1 for t in result.traces if not (t.answer or '').strip())
        return {
            'avg_chunks': round(mean(chunk_counts), 4) if chunk_counts else 0.0,
            'avg_citations': round(mean(cited_counts), 4) if cited_counts else 0.0,
            'avg_latency_ms': round(mean(latency), 4) if latency else 0.0,
            'avg_cost_usd': round(mean(costs), 4) if costs else 0.0,
            'answerless_rate': round(answerless / max(len(result.traces), 1), 4),
            'retrieve_span_rate': round(sum(1 for t in result.traces if any(s.kind == 'retrieve' for s in t.spans)) / max(len(result.traces), 1), 4),
            'rerank_span_rate': round(sum(1 for t in result.traces if any(s.kind == 'rerank' for s in t.spans)) / max(len(result.traces), 1), 4),
            'verify_span_rate': round(sum(1 for t in result.traces if any(s.kind == 'verify' for s in t.spans)) / max(len(result.traces), 1), 4),
        }

    def _feedback_summary(self, result: EvaluationResult) -> dict[str, float]:
        if not result.feedback_events:
            return {}
        total = len(result.feedback_events)
        neg_kinds = {'thumbs_down', 'user_correction', 'escalation', 'hallucination', 'policy'}
        negative = sum(1 for e in result.feedback_events if e.kind in neg_kinds)
        return {
            'negative_rate': round(negative / total, 4),
            'hallucination_feedback_rate': round(sum(1 for e in result.feedback_events if e.kind == 'hallucination') / total, 4),
            'escalation_rate': round(sum(1 for e in result.feedback_events if e.kind == 'escalation') / total, 4),
            'policy_rate': round(sum(1 for e in result.feedback_events if e.kind == 'policy') / total, 4),
        }

    def _historical_priors(self) -> dict[str, float]:
        priors = self.store.load_causal_priors(self.base_priors)
        return {k: round(min(0.95, max(0.01, float(v))), 4) for k, v in priors.items()}

    def _adaptive_priors(self, result: EvaluationResult) -> dict[str, float]:
        priors = dict(self._historical_priors())
        feedback = self._feedback_summary(result)
        trace = self._trace_summary(result)
        metadata_updates = result.metadata.get('causal_prior_updates', {}) or {}
        for node, value in metadata_updates.items():
            if node in priors:
                priors[node] = round(min(0.95, max(0.01, float(value))), 4)
        if feedback.get('hallucination_feedback_rate', 0.0) > 0.15:
            priors['grounding_defect'] = round(min(0.95, priors['grounding_defect'] + 0.06), 4)
        if feedback.get('negative_rate', 0.0) > 0.35:
            priors['distribution_shift'] = round(min(0.95, priors['distribution_shift'] + 0.05), 4)
        if trace.get('avg_chunks', 0.0) < 3.5:
            priors['corpus_chunking_defect'] = round(min(0.95, priors['corpus_chunking_defect'] + 0.04), 4)
        return priors

    def _trace_node_deltas(self, result: EvaluationResult) -> dict[str, float]:
        deltas = {node: 0.0 for node in self.base_priors}
        if not result.traces:
            return deltas
        for tr in result.traces:
            chunks = len(tr.retrieved_chunks)
            citations = len(tr.citations)
            has_rerank = any(s.kind == 'rerank' for s in tr.spans)
            has_verify = any(s.kind == 'verify' for s in tr.spans)
            if chunks <= 2:
                deltas['corpus_chunking_defect'] += 0.06
                deltas['retrieval_recall_defect'] += 0.08
            if chunks >= 9 and not has_rerank:
                deltas['retrieval_precision_defect'] += 0.06
            if chunks >= 9:
                deltas['context_packing_defect'] += 0.05
            if citations == 0 and (tr.answer or '').strip():
                deltas['citation_binding_defect'] += 0.05
            if tr.latency_ms is not None and tr.latency_ms > 6500:
                deltas['distribution_shift'] += 0.02
            if not has_verify:
                deltas['grounding_defect'] += 0.01
        n = max(len(result.traces), 1)
        return {k: round(v / n, 4) for k, v in deltas.items()}

    def _node_evidence(self, node: str, result: EvaluationResult, gaps: dict[str, float], priors: dict[str, float], trace_deltas: dict[str, float]) -> tuple[float, list[str]]:
        cp = result.score('context_precision', 1.0) or 1.0
        cr = result.score('context_recall', 1.0) or 1.0
        faith = result.score('faithfulness', 1.0) or 1.0
        util = result.score('context_utilization', 1.0) or 1.0
        cite = result.score('citation_accuracy', 1.0) or 1.0
        hall = result.score('hallucination', 0.0) or 0.0
        noise = result.score('noise_sensitivity', 0.0) or 0.0
        ans = result.score('answer_correctness', 1.0) or 1.0
        feedback = self._feedback_summary(result)
        agreement = self._agreement_map(result)
        avg_agreement = mean(agreement.values()) if agreement else 1.0
        trace = self._trace_summary(result)

        contributions: list[tuple[float, str]] = []
        def add(delta: float, reason: str) -> None:
            if abs(delta) > 1e-9:
                contributions.append((delta, reason))

        if node == 'corpus_chunking_defect':
            add(1.0 * gaps.get('context_recall', 0.0), f"Recall gap is {gaps.get('context_recall', 0.0):.2f}.")
            if result.metadata.get('document_structure_preserved') is False:
                add(0.45, 'Metadata shows document structure was not preserved during ingestion.')
            if trace.get('avg_chunks', 0.0) < 4.0:
                add(0.22, 'Traces show a small number of retrieved chunks per query.')
        elif node == 'retrieval_recall_defect':
            add(1.2 * gaps.get('context_recall', 0.0), f"context_recall={cr:.2f} is below target.")
            if cp >= self.thresholds['context_precision']:
                add(0.18, 'Precision is acceptable, so evidence miss is more likely than noise.')
        elif node == 'retrieval_precision_defect':
            add(1.15 * gaps.get('context_precision', 0.0), f"context_precision={cp:.2f} is below target.")
            if hall > self.thresholds['hallucination']:
                add(0.12, 'Hallucination rises when noisy passages enter the context window.')
            if trace.get('rerank_span_rate', 0.0) < 0.5:
                add(0.10, 'Sparse rerank spans suggest weak post-retrieval filtering.')
        elif node == 'context_packing_defect':
            add(0.9 * gaps.get('context_utilization', 0.0), f"context_utilization={util:.2f} is below target.")
            if trace.get('avg_chunks', 0.0) >= 8.0:
                add(0.14, 'Many retrieved chunks raise packing pressure.')
            if cp < self.thresholds['context_precision'] and cr >= 0.7:
                add(0.08, 'Enough evidence exists but packing may be letting noise dominate.')
        elif node == 'grounding_defect':
            add(1.15 * gaps.get('faithfulness', 0.0), f"faithfulness={faith:.2f} is below target.")
            add(0.95 * gaps.get('answer_correctness', 0.0), f"answer_correctness={ans:.2f} is below target.")
            if hall > self.thresholds['hallucination']:
                add(0.24, 'Hallucination exceeds threshold.')
            if feedback.get('hallucination_feedback_rate', 0.0) > 0.10:
                add(0.18, 'Feedback contains repeated hallucination complaints.')
            if trace.get('verify_span_rate', 0.0) < 0.3:
                add(0.05, 'Verification spans are sparse.')
        elif node == 'citation_binding_defect':
            add(1.1 * gaps.get('citation_accuracy', 0.0), f"citation_accuracy={cite:.2f} is below target.")
            if faith >= self.thresholds['faithfulness'] and cite < self.thresholds['citation_accuracy']:
                add(0.14, 'Answer quality is acceptable but citation mapping is still weak.')
        elif node == 'judge_or_metric_instability':
            if avg_agreement < 0.75:
                add(0.55 * (0.75 - avg_agreement + 0.01), f"Evaluator agreement={avg_agreement:.2f} is low.")
            if result.metadata.get('judge_prompt_changed'):
                add(0.12, 'Metadata indicates the judge prompt changed.')
        elif node == 'distribution_shift':
            if result.metadata.get('dataset_shift') or result.metadata.get('domain_shift'):
                add(0.32, 'Metadata indicates dataset or domain shift.')
            if feedback.get('negative_rate', 0.0) > 0.30:
                add(0.18, 'Negative production feedback is elevated.')
            if trace.get('answerless_rate', 0.0) > 0.10:
                add(0.10, 'Answerless responses suggest shifted or harder queries.')

        if trace_deltas.get(node, 0.0) > 0:
            add(trace_deltas[node], f"Trace-level attribution adds {trace_deltas[node]:.2f}.")
        if noise > self.thresholds.get('noise_sensitivity', 1.0) and node in {'retrieval_precision_defect', 'grounding_defect'}:
            add(0.10, f"noise_sensitivity={noise:.2f} exceeds threshold.")

        score = _logit(priors[node]) + sum(delta for delta, _ in contributions)
        evidence = [reason for _, reason in sorted(contributions, key=lambda x: abs(x[0]), reverse=True)[:6]]
        return score, evidence

    def _build_causal_graph(self, result: EvaluationResult, gaps: dict[str, float]) -> CausalGraph:
        priors = self._adaptive_priors(result)
        trace_deltas = self._trace_node_deltas(result)
        logit_scores: dict[str, float] = {}
        evidence_map: dict[str, list[str]] = {}
        for node in self.graph.nodes:
            logit_scores[node], evidence_map[node] = self._node_evidence(node, result, gaps, priors, trace_deltas)

        topo = list(nx.topological_sort(self.graph))
        propagated = dict(logit_scores)
        for _ in range(3):
            for node in topo:
                incoming = 0.0
                for parent in self.graph.predecessors(node):
                    parent_posterior = _sigmoid(propagated[parent])
                    delta = self.graph[parent][node]['weight'] * max(0.0, parent_posterior - priors[parent])
                    incoming += delta
                    if delta > 0.01:
                        msg = f"Upstream propagation from {parent} (+{delta:.2f}) via causal edge."
                        if msg not in evidence_map[node]:
                            evidence_map[node].append(msg)
                propagated[node] = logit_scores[node] + incoming
        nodes = []
        for node in topo:
            posterior = round(min(0.995, max(0.001, _sigmoid(propagated[node]))), 4)
            nodes.append(CausalSignal(
                node=node,
                component=self.node_components[node],
                prior=priors[node],
                posterior=posterior,
                evidence=evidence_map[node],
                recommended_experiment=self.node_actions[node],
            ))
        nodes = sorted(nodes, key=lambda s: s.posterior, reverse=True)
        edges = [CausalEdge(source=u, target=v, weight=float(d['weight']), rationale=d['rationale']) for u, v, d in self.graph.edges(data=True)]
        return CausalGraph(nodes=nodes, edges=edges)

    def analyze(
        self,
        result: EvaluationResult,
        optimization_history: list[str] | None = None,
    ) -> DiagnosisReport:
        # ``optimization_history`` is the list of optimization candidate
        # names already applied to this RAG (e.g. ["autorag_pipeline_search"]
        # after one retrieval tune). Used to escalate recommendations
        # when a defect persists despite the obvious fix having run.
        applied = list(optimization_history or [])
        gaps = self._metric_gaps(result)
        cp = result.score('context_precision', 1.0) or 1.0
        cr = result.score('context_recall', 1.0) or 1.0
        cer = result.score('context_entities_recall', cr) or cr
        faith = result.score('faithfulness', 1.0) or 1.0
        util = result.score('context_utilization', 1.0) or 1.0
        noise = result.score('noise_sensitivity', 0.0) or 0.0
        hall = result.score('hallucination', 0.0) or 0.0
        ans = result.score('answer_correctness', result.score('answer_accuracy', 1.0) or 1.0) or 1.0
        cite = result.score('citation_accuracy', 1.0) or 1.0
        agreement = self._agreement_map(result)
        causal_graph = self._build_causal_graph(result, gaps)
        causal_signals = causal_graph.nodes

        hypotheses: list[DiagnosisHypothesis] = []
        candidates: list[str] = []
        actions: list[str] = []
        disambiguation: list[str] = []

        if cr < self.thresholds['context_recall'] and cp >= self.thresholds['context_precision']:
            _lvl = _escalation_level('retrieval_recall_defect', applied)
            _rung = _ESCALATION_LADDERS['retrieval_recall_defect'][_lvl]
            _ev = [f"context_recall={cr:.2f} is below target {self.thresholds['context_recall']:.2f}", f"context_precision={cp:.2f} is not the primary bottleneck", f"entity recall proxy={cer:.2f} suggests missing supporting facts"]
            if _lvl > 0:
                _ev.append(f"retrieval tuning was already applied {applied.count('autorag_pipeline_search')}x but recall is still below target (escalation level {_lvl}).")
            hypotheses.append(DiagnosisHypothesis(component='retrieval', root_cause=('evidence miss despite acceptable retrieval precision' if _lvl == 0 else 'recall still failing after retrieval tuning — escalate retrieval strategy'), severity='high', confidence=0.86, evidence=_ev, recommended_actions=list(_rung['actions'])))
            candidates.extend(_rung['candidates'])
            actions.append('Prioritize retrieval recall experiments before generator prompt tuning.' if _lvl == 0 else _rung['note'])
            disambiguation.append('Hold the generator fixed and compare recall with and without chunking changes.')

        if cp < self.thresholds['context_precision']:
            _lvl = _escalation_level('retrieval_precision_defect', applied)
            _rung = _ESCALATION_LADDERS['retrieval_precision_defect'][_lvl]
            _ev = [f"context_precision={cp:.2f} is below target {self.thresholds['context_precision']:.2f}", 'Noisy contexts usually propagate to hallucination and citation failures.']
            if _lvl > 0:
                _ev.append(f"retrieval tuning was already applied {applied.count('autorag_pipeline_search')}x but precision is still below target (escalation level {_lvl}).")
            hypotheses.append(DiagnosisHypothesis(component='retrieval', root_cause=('retrieval noise or weak ranking quality' if _lvl == 0 else 'precision still failing after retrieval tuning — escalate retrieval complexity'), severity='high', confidence=0.84, evidence=_ev, recommended_actions=list(_rung['actions'])))
            candidates.extend(_rung['candidates'])
            actions.append('Improve ranking precision and evidence filtering.' if _lvl == 0 else _rung['note'])

        if faith < self.thresholds['faithfulness'] and cr >= 0.7:
            _lvl = _escalation_level('grounding_defect', applied)
            _rung = _ESCALATION_LADDERS['grounding_defect'][_lvl]
            _ev = [f"faithfulness={faith:.2f} is below target {self.thresholds['faithfulness']:.2f}", f"context_recall={cr:.2f} indicates at least some relevant evidence is available", f"context_utilization={util:.2f} suggests evidence may not be used effectively"]
            if _lvl > 0:
                _ev.append(f"prompt tuning was already applied {applied.count('dspy_prompt_optimization')}x but grounding is still weak (escalation level {_lvl}).")
            hypotheses.append(DiagnosisHypothesis(component='generation', root_cause=('generator is not grounding sufficiently on retrieved evidence' if _lvl == 0 else 'grounding still weak after prompt tuning — escalate generation strategy'), severity='high', confidence=0.83, evidence=_ev, recommended_actions=list(_rung['actions'])))
            candidates.extend(_rung['candidates'])
            actions.append('Tune generator behavior after retrieval quality is acceptable.' if _lvl == 0 else _rung['note'])
            disambiguation.append('Compare the same retrieved context under a citation-first prompt and a claim-then-evidence prompt.')

        if noise > self.thresholds['noise_sensitivity'] or hall > self.thresholds['hallucination']:
            hypotheses.append(DiagnosisHypothesis(component='generation', root_cause='answer is fragile under distractors or unsupported reasoning', severity='high', confidence=0.81, evidence=[f"noise_sensitivity={noise:.2f}", f"hallucination={hall:.2f}", 'The generator likely overweights spurious or weakly relevant text.'], recommended_actions=['Constrain the answer to quoted or cited evidence.', 'Use context packing with stronger ordering and section labels.', 'Add a verifier or claim-level grounding pass.']))
            candidates.extend(['dspy_prompt_optimization', 'joint_ablation_eval'])
            actions.append('Run ablations to separate noisy retrieval from generator overreach.')

        if ans < self.thresholds['answer_correctness'] and not hypotheses:
            hypotheses.append(DiagnosisHypothesis(component='pipeline', root_cause='end-to-end quality is weak but component-level signals are inconclusive', severity='medium', confidence=0.62, evidence=[f"answer_correctness={ans:.2f} is below target"], recommended_actions=['Run controlled ablations over retriever, reranker, and answer prompt.', 'Inspect difficult examples and metric-label alignment.']))
            candidates.append('joint_ablation_eval')
            actions.append('Review evaluation set quality and run component ablations.')

        if cite < self.thresholds['citation_accuracy']:
            hypotheses.append(DiagnosisHypothesis(component='e2e', root_cause='citation mapping is weaker than answer generation', severity='medium', confidence=0.72, evidence=[f"citation_accuracy={cite:.2f} is below target {self.thresholds['citation_accuracy']:.2f}"], recommended_actions=['Enforce sentence-level citation formatting.', 'Return passage ids with the answer synthesis step.']))
            actions.append('Add explicit citation scaffolding in the generation prompt or response schema.')

        # Per-layer aggregate scores (retrieval / generation / e2e) so
        # the summary can lead with "which layer is weakest" -- a
        # coarser, more actionable signal than any single metric gap.
        from ragdx.core.metrics import compute_layer_scores, weakest_layer
        layer_scores = compute_layer_scores(
            {**result.retrieval, **result.generation, **result.e2e}
        )
        weak = weakest_layer(layer_scores)

        summary = 'Metrics are close to the configured thresholds. No dominant bottleneck is detected.' if not hypotheses else f"Primary bottleneck: {hypotheses[0].root_cause}. {len(hypotheses)} diagnosis hypotheses were generated."
        # Lead with the weakest layer when we have layer scores -- this
        # is the "attack this layer first" prioritization the layer
        # aggregation enables.
        if weak is not None and layer_scores[weak]["score"] is not None:
            _ls_bits = ", ".join(
                f"{lyr}={layer_scores[lyr]['score']:.2f}"
                for lyr in ("retrieval", "generation", "e2e")
                if layer_scores[lyr]["score"] is not None
            )
            summary = (
                f"Weakest layer: {weak} ({layer_scores[weak]['score']:.2f}). "
                f"Layer scores: {_ls_bits}. " + summary
            )
        if causal_signals:
            lead_signal = causal_signals[0]
            summary += f" Lead causal node: {lead_signal.node} with posterior {lead_signal.posterior:.2f}."
            if lead_signal.recommended_experiment and lead_signal.recommended_experiment not in actions:
                actions.append(lead_signal.recommended_experiment)

        confidence = round(min(0.985, mean([h.confidence for h in hypotheses]) if hypotheses else 0.75), 4)
        if agreement:
            confidence = round(confidence * (0.80 + 0.20 * mean(agreement.values())), 4)

        # Build the rule-based layer separately so callers (notably the
        # LLM explainer wrapping us) can preserve our lineage in
        # ``report.rule_based`` while overwriting the top-level fields
        # with their own analysis. Phase 2: rule vs LLM split.
        from ragdx.schemas.models import DiagnosisLayer
        rule_layer = DiagnosisLayer(
            source="rule",
            summary=summary,
            hypotheses=list(hypotheses),
            causal_signals=list(causal_signals),
            metric_gaps=dict(gaps),
            optimization_candidates=sorted(set(candidates)),
            priority_actions=list(dict.fromkeys(actions)),
            disambiguation_actions=list(dict.fromkeys(disambiguation)),
            diagnosis_confidence=confidence,
        )
        report = DiagnosisReport(
            summary=summary,
            expected_thresholds=self.thresholds,
            metric_gaps=gaps,
            hypotheses=hypotheses,
            optimization_candidates=sorted(set(candidates)),
            priority_actions=list(dict.fromkeys(actions)),
            causal_signals=causal_signals,
            causal_graph=causal_graph,
            evaluator_agreement=agreement,
            diagnosis_confidence=confidence,
            disambiguation_actions=list(dict.fromkeys(disambiguation)),
            # New: per-source lineage.
            rule_based=rule_layer,
            active_source="rule",
            # New: per-layer aggregate scores for the three-layer view.
            layer_scores=layer_scores,
        )
        self.store.update_causal_priors_from_report(report, result.feedback_events)
        return report
