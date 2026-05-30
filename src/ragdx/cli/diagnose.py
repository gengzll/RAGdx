"""``ragdx diagnose`` / ``plan`` / ``explain-plan`` commands.

Three commands that consume a normalized ``EvaluationResult`` JSON and
produce diagnostic / planning artifacts. They share the
:func:`_diagnose_and_plan` helper so that ``diagnose`` and ``plan``
return consistent reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print

from ragdx.cli._app import app
from ragdx.cli._shared import _diagnose_and_plan, _load_eval, _store
from ragdx.utils.reporting import summarize_plan


@app.command()
def diagnose(
    eval_json: str = typer.Argument(..., help="Path to an evaluation results JSON file."),
    save: bool = typer.Option(False, help="Persist the run, diagnosis and plan to the RunStore."),
    name: str = typer.Option("", help="Optional human-readable name to attach to the saved run."),
    baseline_run_id: str = typer.Option("", help="Run id to attach as the baseline for this run."),
    use_llm: bool = typer.Option(False, help="Use LLM diagnosis instead of rule-based diagnosis."),
    use_both: bool = typer.Option(False, help="Rule-based + LLM diagnosis, then LLM synthesis."),
    use_llm_planner: bool = typer.Option(False, help="Refine the optimization plan with an LLM."),
):
    """Diagnose an evaluation result and emit a DiagnosisReport JSON."""
    if use_llm and use_both:
        raise typer.BadParameter("Use either --use-llm or --use-both, not both.")
    result = _load_eval(eval_json)
    report, plan = _diagnose_and_plan(
        result, use_llm=use_llm, use_both=use_both, use_llm_planner=use_llm_planner
    )
    if save:
        run = _store().save_run(
            result, report, plan, name=name or None, baseline_run_id=baseline_run_id or None
        )
        print(f"Saved run: {run.run_id}")
    print(report.model_dump_json(indent=2))


@app.command()
def plan(
    eval_json: str = typer.Argument(..., help="Path to an evaluation results JSON file."),
    strategy: str = typer.Option("bayesian", help="Search strategy: bayesian or pareto_evolutionary."),
    budget: int = typer.Option(12, help="Trial budget to distribute across experiments."),
    use_llm_planner: bool = typer.Option(False, help="Refine the optimization plan with an LLM."),
    human_readable: bool = typer.Option(
        False,
        "--human-readable",
        help="Print a human-readable baseline-relative explanation of the generated plan.",
    ),
):
    """Generate an OptimizationPlan from an evaluation result."""
    result = _load_eval(eval_json)
    _report, opt_plan = _diagnose_and_plan(
        result, strategy=strategy, budget=budget, use_llm_planner=use_llm_planner
    )
    if human_readable:
        print(summarize_plan(opt_plan.model_dump()))
    else:
        print(opt_plan.model_dump_json(indent=2))


@app.command("explain-plan")
def explain_plan(
    plan_json: str = typer.Argument(..., help="Path to an OptimizationPlan JSON file."),
):
    """Pretty-print an existing OptimizationPlan."""
    payload = json.loads(Path(plan_json).read_text(encoding="utf-8"))
    print(summarize_plan(payload))


__all__ = ["diagnose", "explain_plan", "plan"]
