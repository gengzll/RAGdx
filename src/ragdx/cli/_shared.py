"""Internal helpers shared across the cli/ subpackage.

These were private helpers (``_store`` / ``_build_llm_callable`` /
``_load_eval`` / ``_build_engine`` / ``_diagnose_and_plan``) inlined
at the top of the old ``cli.py`` monolith. Centralising them here
lets every command subgroup reach the same set without each having
to re-import the world.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ragdx.config import get_settings
from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.engines.llm_diagnosis import LLMDiagnosisExplainer
from ragdx.errors import LLMConfigError
from ragdx.llm import LLMCallable, get_llm_callable
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import EvaluationResult
from ragdx.storage.run_store import RunStore


def _store() -> RunStore:
    """Build a :class:`RunStore` rooted at the configured storage path."""
    return RunStore(root=str(get_settings().storage.root))


def _build_llm_callable() -> LLMCallable:
    """Instantiate the configured LLM provider, surfacing config errors clearly."""
    try:
        return get_llm_callable()
    except LLMConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _load_eval(path: str | Path) -> EvaluationResult:
    """Load an EvaluationResult from a JSON file."""
    with open(path, encoding="utf-8") as f:
        return EvaluationResult(**json.load(f))


def _load_eval_or_latest(path: str | Path) -> tuple[EvaluationResult, str]:
    """Load an eval from ``path``; fall back to ``RunStore.latest()``
    when ``path`` is empty.

    Returns ``(result, hint)`` where ``hint`` is either ``""`` (loaded
    from an explicit path; no log needed) or a short string the caller
    can show the user to make the implicit default visible
    (``f"latest run {id} ({name})"``).

    Used by ``diagnose`` / ``plan`` / ``optimize`` / ``save`` /
    ``compare`` so the common "rerun on the thing I just saved" case
    needs zero arguments. Empty RunStore raises a clear error.
    """
    if path:
        return _load_eval(path), ""
    latest = _store().latest()
    if latest is None:
        raise typer.BadParameter(
            "No evaluation JSON given and the RunStore is empty. Pass "
            "an EvaluationResult JSON path, or run `ragdx evaluate "
            "--save` first to populate the RunStore."
        )
    return latest.evaluation, f"latest run {latest.run_id} ({latest.name})"


def _resolve_run_id_or_latest(run_id: str) -> tuple[str, str]:
    """Resolve a run-id argument, falling back to the latest saved
    run when ``run_id`` is empty.

    Returns ``(run_id, hint)`` matching :func:`_load_eval_or_latest`'s
    contract. Empty RunStore raises ``typer.BadParameter``.

    Used by commands that accept a run id positional (e.g.
    ``export-report``, ``attach-feedback``).
    """
    if run_id:
        return run_id, ""
    latest = _store().latest()
    if latest is None:
        raise typer.BadParameter(
            "No run id given and the RunStore is empty. Pass a run "
            "id, or run `ragdx evaluate --save` first to populate the "
            "RunStore."
        )
    return latest.run_id, f"latest run ({latest.name})"


def _build_engine(use_llm: bool = False, use_both: bool = False) -> RAGDiagnosisEngine:
    """Build a diagnosis engine, optionally with an LLM explainer."""
    if not use_llm and not use_both:
        return RAGDiagnosisEngine()
    llm_callable = _build_llm_callable()
    return RAGDiagnosisEngine(llm_explainer=LLMDiagnosisExplainer(llm_callable=llm_callable))


def _diagnose_and_plan(
    result: EvaluationResult,
    use_llm: bool = False,
    use_both: bool = False,
    use_llm_planner: bool = False,
    strategy: str = "bayesian",
    budget: int = 12,
):
    """Run diagnose + plan in one pass. Shared between ``diagnose``,
    ``plan``, ``optimize``, and ``save`` commands."""
    engine = _build_engine(use_llm=use_llm, use_both=use_both)
    report = engine.diagnose(result, use_llm=use_llm, use_both=use_both)
    planner_llm = _build_llm_callable() if (use_llm_planner or use_llm or use_both) else None
    plan = OptimizationPlanner(llm_callable=planner_llm).build_plan(
        report, result=result, strategy=strategy, budget=budget
    )
    return report, plan


__all__ = [
    "_build_engine",
    "_build_llm_callable",
    "_diagnose_and_plan",
    "_load_eval",
    "_load_eval_or_latest",
    "_resolve_run_id_or_latest",
    "_store",
]
