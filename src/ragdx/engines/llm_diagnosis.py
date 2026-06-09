"""LLM-powered diagnosis and explanation engine.

The explainer wraps a :data:`ragdx.llm.LLMCallable` (or any provider
instance) and produces refined / synthesised :class:`DiagnosisReport`
objects on top of the rule-based output.

LLM responses are coerced to JSON and validated against the schema; any
free-form additions to constrained fields (notably the ``component``
literal) are normalised so the result remains a valid Pydantic model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from ragdx.errors import LLMError
from ragdx.schemas.models import DiagnosisReport, EvaluationResult
from ragdx.utils.logging import get_logger

logger = get_logger(__name__)


DEFAULT_REFINE_PROMPT = """
You are a senior RAG evaluation diagnostician.
Your task is to refine a deterministic diagnosis of a retrieval-augmented generation system.

You will receive:
1. target thresholds
2. measured metrics across retrieval, generation, and end-to-end layers
3. an initial rule-based diagnosis
4. metadata about the run
5. optional evaluator agreement, traces, and feedback signals if available

Reasoning requirements:
- Start from the metrics, not from stylistic preference.
- Distinguish retrieval recall failure, retrieval precision or ranking noise, generator grounding failure, citation failure, and broader pipeline failure.
- Separate primary bottlenecks from secondary symptoms.
- Prefer causal reasoning over rephrasing the rule-based diagnosis.
- Be conservative when evidence is weak or mixed.
- Convert the analysis into remediation actions in execution order.

Output requirements:
- Return strict JSON only.
- Use exactly these top-level keys:
  summary, expected_thresholds, metric_gaps, hypotheses, optimization_candidates, priority_actions, causal_signals, evaluator_agreement, diagnosis_confidence, disambiguation_actions
- summary must be concise and specific.
- hypotheses must be a list of objects with keys:
  component, root_cause, severity, confidence, evidence, recommended_actions
- causal_signals must be a list of objects with keys: node, component, posterior, prior, evidence, recommended_experiment
- confidence must be numeric in [0,1].
- severity must be one of: low, medium, high, critical.
- component must be one of: retrieval, generation, e2e, pipeline.
- expected_thresholds should preserve the input threshold mapping unless there is a clear reason to restate it.
- metric_gaps should preserve the gap mapping unless the input appears inconsistent.
- priority_actions should be ordered from highest to lowest leverage.
""".strip()


DEFAULT_SUMMARIZE_BOTH_PROMPT = """
You are a senior RAG diagnosis reviewer.
You are given two diagnosis reports for the same RAG evaluation result:
1. a deterministic rule-based diagnosis
2. an LLM-refined diagnosis

Your task is to synthesize them into one final diagnosis report.

Instructions:
- Preserve agreements between the two reports.
- Resolve conflicts carefully using the actual metrics and thresholds.
- Prefer the more evidence-backed explanation, not the more verbose one.
- Avoid duplicate hypotheses.
- Keep the final report practical for remediation planning.
- Priority actions should reflect the real execution order.
- Retrieval fixes should usually precede generator tuning when retrieval is the main bottleneck.

Output requirements:
- Return strict JSON only.
- Use exactly these top-level keys:
  summary, expected_thresholds, metric_gaps, hypotheses, optimization_candidates, priority_actions, causal_signals, evaluator_agreement, diagnosis_confidence, disambiguation_actions
- hypotheses must be deduplicated and ranked by likely leverage.
- summary should explicitly mention the dominant bottleneck and any secondary issue.
""".strip()


_VALID_COMPONENTS = {"retrieval", "generation", "e2e", "pipeline"}
_VALID_SEVERITIES = {"low", "medium", "high", "critical"}


def _normalize_component(value: Any) -> str:
    """Coerce LLM-emitted component labels to the canonical literal set.

    LLMs occasionally emit values like ``"e2e / citation mapping"`` or
    ``"Pipeline observability"``; we map by prefix or substring match
    rather than reject outright.
    """

    if not isinstance(value, str):
        return "pipeline"
    text = value.strip().lower()
    if text in _VALID_COMPONENTS:
        return text
    head = re.split(r"[\s/|,;:.]", text, maxsplit=1)[0]
    if head in _VALID_COMPONENTS:
        return head
    for candidate in _VALID_COMPONENTS:
        if candidate in text:
            return candidate
    return "pipeline"


def _normalize_severity(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() in _VALID_SEVERITIES:
        return value.strip().lower()
    return "medium"


def _clip_confidence(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, f))


def _normalize_payload(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    hypotheses: list[dict[str, Any]] = []
    for raw in data.get("hypotheses", []) or []:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item["component"] = _normalize_component(item.get("component"))
        item["severity"] = _normalize_severity(item.get("severity"))
        item["confidence"] = _clip_confidence(item.get("confidence", 0.5))
        hypotheses.append(item)
    out["hypotheses"] = hypotheses
    causal: list[dict[str, Any]] = []
    for raw in data.get("causal_signals", []) or []:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item["component"] = _normalize_component(item.get("component"))
        causal.append(item)
    out["causal_signals"] = causal
    return out


class LLMDiagnosisExplainer:
    def __init__(
        self,
        llm_callable: Callable[[str], str | dict[str, Any]],
        prompt_template: str = DEFAULT_REFINE_PROMPT,
        summary_prompt_template: str = DEFAULT_SUMMARIZE_BOTH_PROMPT,
    ):
        if llm_callable is None:
            raise ValueError("llm_callable is required for LLMDiagnosisExplainer.")
        self.llm_callable = llm_callable
        self.prompt_template = prompt_template
        self.summary_prompt_template = summary_prompt_template

    @staticmethod
    def _coerce_json(raw: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            raise LLMError(f"LLM call returned unsupported type {type(raw).__name__}")
        text = raw.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError as exc:
                    raise LLMError(f"LLM response was not valid JSON: {exc}") from exc
            raise LLMError("LLM response did not contain a JSON object.") from None

    def _call(self, prompt: str) -> dict[str, Any]:
        try:
            raw = self.llm_callable(prompt)
        except Exception as exc:
            raise LLMError(f"LLM call failed: {exc}") from exc
        return self._coerce_json(raw)

    def explain(self, result: EvaluationResult, base_report: DiagnosisReport) -> DiagnosisReport:
        payload = {
            "thresholds": base_report.expected_thresholds,
            "metrics": {
                "retrieval": result.retrieval,
                "generation": result.generation,
                "e2e": result.e2e,
            },
            "initial_diagnosis": base_report.model_dump(),
            "metadata": result.metadata,
            "traces": [t.model_dump() for t in result.traces[:10]],
            "feedback_events": [f.model_dump() for f in result.feedback_events[:20]],
            "evaluator_scores": [e.model_dump() for e in result.evaluator_scores[:50]],
            "calibrations": [c.model_dump() for c in result.calibrations],
        }
        prompt = f"{self.prompt_template}\n\nINPUT:\n{json.dumps(payload, indent=2, ensure_ascii=False)}"
        data = _normalize_payload(self._call(prompt))
        try:
            llm_report = DiagnosisReport(**data)
        except Exception as exc:
            raise LLMError(f"LLM diagnosis output failed validation: {exc}") from exc

        # Phase 2: preserve lineage. Top-level fields reflect the LLM
        # output (active), but we keep both ``rule_based`` (from the
        # base report) and ``llm_based`` (just produced) so the
        # renderer can show "what changed from rule to LLM".
        from ragdx.schemas.models import DiagnosisLayer
        llm_layer = DiagnosisLayer(
            source="llm",
            summary=llm_report.summary,
            hypotheses=list(llm_report.hypotheses),
            causal_signals=list(llm_report.causal_signals),
            metric_gaps=dict(llm_report.metric_gaps),
            optimization_candidates=list(llm_report.optimization_candidates),
            priority_actions=list(llm_report.priority_actions),
            disambiguation_actions=list(llm_report.disambiguation_actions),
            diagnosis_confidence=llm_report.diagnosis_confidence,
        )
        # ``base_report.rule_based`` may be None for old callers; fall
        # back to a freshly-built layer from base_report's fields.
        rule_layer = base_report.rule_based or DiagnosisLayer(
            source="rule",
            summary=base_report.summary,
            hypotheses=list(base_report.hypotheses),
            causal_signals=list(base_report.causal_signals),
            metric_gaps=dict(base_report.metric_gaps),
            optimization_candidates=list(base_report.optimization_candidates),
            priority_actions=list(base_report.priority_actions),
            disambiguation_actions=list(base_report.disambiguation_actions),
            diagnosis_confidence=base_report.diagnosis_confidence,
        )
        llm_report.rule_based = rule_layer
        llm_report.llm_based = llm_layer
        llm_report.active_source = "llm"
        return llm_report

    def summarize_both(
        self,
        result: EvaluationResult,
        rule_report: DiagnosisReport,
        llm_report: DiagnosisReport,
    ) -> DiagnosisReport:
        payload = {
            "metrics": {
                "retrieval": result.retrieval,
                "generation": result.generation,
                "e2e": result.e2e,
            },
            "metadata": result.metadata,
            "traces": [t.model_dump() for t in result.traces[:10]],
            "feedback_events": [f.model_dump() for f in result.feedback_events[:20]],
            "evaluator_scores": [e.model_dump() for e in result.evaluator_scores[:50]],
            "calibrations": [c.model_dump() for c in result.calibrations],
            "rule_based_diagnosis": rule_report.model_dump(),
            "llm_diagnosis": llm_report.model_dump(),
        }
        prompt = f"{self.summary_prompt_template}\n\nINPUT:\n{json.dumps(payload, indent=2, ensure_ascii=False)}"
        data = _normalize_payload(self._call(prompt))
        try:
            synth = DiagnosisReport(**data)
        except Exception as exc:
            raise LLMError(f"LLM synthesis output failed validation: {exc}") from exc

        # Phase 2: keep all three layers so the renderer can show
        # rule / LLM / synthesis side-by-side.
        from ragdx.schemas.models import DiagnosisLayer
        synthesis_layer = DiagnosisLayer(
            source="synthesis",
            summary=synth.summary,
            hypotheses=list(synth.hypotheses),
            causal_signals=list(synth.causal_signals),
            metric_gaps=dict(synth.metric_gaps),
            optimization_candidates=list(synth.optimization_candidates),
            priority_actions=list(synth.priority_actions),
            disambiguation_actions=list(synth.disambiguation_actions),
            diagnosis_confidence=synth.diagnosis_confidence,
        )
        synth.rule_based = rule_report.rule_based or DiagnosisLayer(
            source="rule",
            summary=rule_report.summary,
            hypotheses=list(rule_report.hypotheses),
            causal_signals=list(rule_report.causal_signals),
            metric_gaps=dict(rule_report.metric_gaps),
            optimization_candidates=list(rule_report.optimization_candidates),
            priority_actions=list(rule_report.priority_actions),
            disambiguation_actions=list(rule_report.disambiguation_actions),
            diagnosis_confidence=rule_report.diagnosis_confidence,
        )
        synth.llm_based = llm_report.llm_based or DiagnosisLayer(
            source="llm",
            summary=llm_report.summary,
            hypotheses=list(llm_report.hypotheses),
            causal_signals=list(llm_report.causal_signals),
            metric_gaps=dict(llm_report.metric_gaps),
            optimization_candidates=list(llm_report.optimization_candidates),
            priority_actions=list(llm_report.priority_actions),
            disambiguation_actions=list(llm_report.disambiguation_actions),
            diagnosis_confidence=llm_report.diagnosis_confidence,
        )
        synth.synthesis = synthesis_layer
        synth.active_source = "synthesis"
        return synth


__all__ = ["DEFAULT_REFINE_PROMPT", "DEFAULT_SUMMARIZE_BOTH_PROMPT", "LLMDiagnosisExplainer"]
