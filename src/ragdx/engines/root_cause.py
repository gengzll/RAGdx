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
from typing import Dict, List, Tuple

import networkx as nx

from ragdx.core.thresholds import DEFAULT_THRESHOLDS, LOWER_IS_BETTER
from ragdx.storage.run_store import RunStore
from ragdx.schemas.models import CausalEdge, CausalGraph, CausalSignal, DiagnosisHypothesis, DiagnosisReport, EvaluationResult


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(0.999, max(0.001, p))
    return math.log(p / (1.0 - p))


class RuleBasedRootCauseAnalyzer:
    def __init__(self, thresholds: Dict[str, float] | None = None, root: str = '.ragdx'):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS.copy()
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

    def _metric_gaps(self, result: EvaluationResult) -> Dict[str, float]:
        gaps: Dict[str, float] = {}
        for bucket in (result.retrieval, result.generation, result.e2e):
            for metric, value in bucket.items():
                gap = self._gap(metric, value)
                if gap > 0:
                    gaps[metric] = gap
        return dict(sorted(gaps.items(), key=lambda kv: kv[1], reverse=True))

    def _agreement_map(self, result: EvaluationResult) -> Dict[str, float]:
        grouped: Dict[str, List[float]] = defaultdict(list)
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

    def _trace_summary(self, result: EvaluationResult) -> Dict[str, float]:
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

    def _feedback_summary(self, result: EvaluationResult) -> Dict[str, float]:
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

    def _historical_priors(self) -> Dict[str, float]:
        priors = self.store.load_causal_priors(self.base_priors)
        return {k: round(min(0.95, max(0.01, float(v))), 4) for k, v in priors.items()}

    def _adaptive_priors(self, result: EvaluationResult) -> Dict[str, float]:
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

    def _trace_node_deltas(self, result: EvaluationResult) -> Dict[str, float]:
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

    def _node_evidence(self, node: str, result: EvaluationResult, gaps: Dict[str, float], priors: Dict[str, float], trace_deltas: Dict[str, float]) -> Tuple[float, List[str]]:
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

        contributions: List[Tuple[float, str]] = []
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

    def _build_causal_graph(self, result: EvaluationResult, gaps: Dict[str, float]) -> CausalGraph:
        priors = self._adaptive_priors(result)
        trace_deltas = self._trace_node_deltas(result)
        logit_scores: Dict[str, float] = {}
        evidence_map: Dict[str, List[str]] = {}
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

    def analyze(self, result: EvaluationResult) -> DiagnosisReport:
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

        hypotheses: List[DiagnosisHypothesis] = []
        candidates: List[str] = []
        actions: List[str] = []
        disambiguation: List[str] = []

        if cr < self.thresholds['context_recall'] and cp >= self.thresholds['context_precision']:
            hypotheses.append(DiagnosisHypothesis(component='retrieval', root_cause='evidence miss despite acceptable retrieval precision', severity='high', confidence=0.86, evidence=[f"context_recall={cr:.2f} is below target {self.thresholds['context_recall']:.2f}", f"context_precision={cp:.2f} is not the primary bottleneck", f"entity recall proxy={cer:.2f} suggests missing supporting facts"], recommended_actions=['Increase recall with hybrid retrieval or larger candidate pool before reranking.', 'Tune chunk size, overlap, and document segmentation.', 'Inspect query rewriting and metadata filters.']))
            candidates.extend(['autorag_pipeline_search', 'corpus_chunking_search'])
            actions.append('Prioritize retrieval recall experiments before generator prompt tuning.')
            disambiguation.append('Hold the generator fixed and compare recall with and without chunking changes.')

        if cp < self.thresholds['context_precision']:
            hypotheses.append(DiagnosisHypothesis(component='retrieval', root_cause='retrieval noise or weak ranking quality', severity='high', confidence=0.84, evidence=[f"context_precision={cp:.2f} is below target {self.thresholds['context_precision']:.2f}", 'Noisy contexts usually propagate to hallucination and citation failures.'], recommended_actions=['Add or tune reranker.', 'Reduce top-k or separate recall stage from final evidence stage.', 'Use metadata or section-aware filters.']))
            candidates.append('autorag_pipeline_search')
            actions.append('Improve ranking precision and evidence filtering.')

        if faith < self.thresholds['faithfulness'] and cr >= 0.7:
            hypotheses.append(DiagnosisHypothesis(component='generation', root_cause='generator is not grounding sufficiently on retrieved evidence', severity='high', confidence=0.83, evidence=[f"faithfulness={faith:.2f} is below target {self.thresholds['faithfulness']:.2f}", f"context_recall={cr:.2f} indicates at least some relevant evidence is available", f"context_utilization={util:.2f} suggests evidence may not be used effectively"], recommended_actions=['Optimize synthesis prompt to require evidence-backed answers.', 'Use structured answer templates with citation requirements.', 'Add answer compression or claim verification stage.']))
            candidates.append('dspy_prompt_optimization')
            actions.append('Tune generator behavior after retrieval quality is acceptable.')
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

        summary = 'Metrics are close to the configured thresholds. No dominant bottleneck is detected.' if not hypotheses else f"Primary bottleneck: {hypotheses[0].root_cause}. {len(hypotheses)} diagnosis hypotheses were generated."
        if causal_signals:
            lead_signal = causal_signals[0]
            summary += f" Lead causal node: {lead_signal.node} with posterior {lead_signal.posterior:.2f}."
            if lead_signal.recommended_experiment and lead_signal.recommended_experiment not in actions:
                actions.append(lead_signal.recommended_experiment)

        confidence = round(min(0.985, mean([h.confidence for h in hypotheses]) if hypotheses else 0.75), 4)
        if agreement:
            confidence = round(confidence * (0.80 + 0.20 * mean(agreement.values())), 4)

        report = DiagnosisReport(summary=summary, expected_thresholds=self.thresholds, metric_gaps=gaps, hypotheses=hypotheses, optimization_candidates=sorted(set(candidates)), priority_actions=list(dict.fromkeys(actions)), causal_signals=causal_signals, causal_graph=causal_graph, evaluator_agreement=agreement, diagnosis_confidence=confidence, disambiguation_actions=list(dict.fromkeys(disambiguation)))
        self.store.update_causal_priors_from_report(report, result.feedback_events)
        return report
