"""``ragdx optimize`` / ``monitor-session`` / ``show-runner-templates``.

These commands drive the executor (production-path optimization
sessions backed by user-written runner scripts), inspect saved
sessions, and reveal which environment variables the executor reads.
"""

from __future__ import annotations

import json

import typer
from rich import print

from ragdx.cli._app import app
from ragdx.cli._shared import _diagnose_and_plan, _load_eval, _store
from ragdx.optim.executor import OptimizationExecutor
from ragdx.utils.logging import get_logger

logger = get_logger(__name__)


@app.command()
def optimize(
    eval_json: str = typer.Argument(..., help="Path to an evaluation results JSON file."),
    strategy: str = typer.Option("bayesian", help="Search strategy: bayesian or pareto_evolutionary."),
    budget: int = typer.Option(12, help="Trial budget to distribute across experiments."),
    mode: str = typer.Option("simulate", help="Execution mode: simulate, prepare_only, or execute."),
    save_run: bool = typer.Option(True, help="Save the run, diagnosis, plan, and optimization session."),
    name: str = typer.Option("", help="Optional name for the saved run."),
    use_llm: bool = typer.Option(False, help="Use LLM diagnosis instead of rule-based diagnosis."),
    use_both: bool = typer.Option(False, help="Rule-based + LLM diagnosis, then LLM synthesis."),
    use_llm_planner: bool = typer.Option(False, help="Refine the optimization plan with an LLM."),
    strict_execute: bool = typer.Option(
        True,
        "--strict-execute/--lenient-execute",
        help="In execute mode, fail loudly when no runner is configured. "
        "Use --lenient-execute to allow the legacy silent fallback to simulated scoring.",
    ),
):
    """Diagnose → plan → execute an optimization session."""
    if use_llm and use_both:
        raise typer.BadParameter("Use either --use-llm or --use-both, not both.")
    if strategy not in {"bayesian", "pareto_evolutionary"}:
        raise typer.BadParameter("strategy must be bayesian or pareto_evolutionary")
    if mode not in {"simulate", "prepare_only", "execute"}:
        raise typer.BadParameter("mode must be simulate, prepare_only, or execute")

    if mode == "simulate":
        # Loud, visible warning — easy to miss when this scrolls past in a
        # busy terminal. The executor also writes this to session.notes.
        print(
            "[bold yellow]⚠️  SIMULATED MODE[/bold yellow]: "
            "trial scores come from a deterministic hash-based stub, "
            "not a real runner. Use `--mode execute` (with a configured "
            "RAGDX_<tool>_RUNNER_CMD) for real metrics."
        )

    result = _load_eval(eval_json)
    report, opt_plan = _diagnose_and_plan(
        result,
        use_llm=use_llm,
        use_both=use_both,
        use_llm_planner=use_llm_planner,
        strategy=strategy,
        budget=budget,
    )
    store = _store()
    run = None
    if save_run:
        run = store.save_run(result, report, opt_plan, name=name or None)
    executor = OptimizationExecutor(strict_execute=strict_execute)
    session = executor.execute_plan(
        opt_plan,
        baseline=result,
        strategy=strategy,
        mode=mode,
        run_id=run.run_id if run else None,
    )
    store.save_session(session)
    if run is not None:
        store.update_run_latest_session(run.run_id, session.session_id)
    if mode == "execute" and session.status == "failed":
        logger.error(
            "Optimize session %s ended in 'failed' state — every trial errored. "
            "Common causes: missing RAGDX_*_RUNNER_CMD, or runner script crashed. "
            "See per-trial logs in .ragdx/optimization/<session_id>/outputs/.",
            session.session_id,
        )
    print(session.model_dump_json(indent=2))


@app.command("monitor-session")
def monitor_session(
    session_id: str = typer.Argument(..., help="Optimization session id."),
    show_logs: bool = typer.Option(False, help="Show per-trial logs."),
):
    """Inspect the status of an optimization session."""
    from ragdx.storage.run_store import SessionNotFoundError

    try:
        session = _store().load_session(session_id)
    except SessionNotFoundError as exc:
        raise typer.BadParameter(f"Session not found: {session_id}") from exc
    if show_logs:
        print(session.model_dump_json(indent=2))
        return
    print(
        json.dumps(
            {
                "session_id": session.session_id,
                "status": session.status,
                "strategy": session.strategy,
                "mode": session.mode,
                "completed_trials": session.completed_trials,
                "total_trials": session.total_trials,
                "best_trial_id": session.best_trial_id,
                "pareto_front_ids": session.pareto_front_ids,
            },
            indent=2,
        )
    )


@app.command("show-runner-templates")
def show_runner_templates():
    """Print example RAGDX_*_RUNNER_CMD environment values."""
    templates = {
        "RAGDX_DSPY_RUNNER_CMD": "python examples/run_external_trial_example.py --config {config} --output {output}",
        "RAGDX_AUTORAG_RUNNER_CMD": "python examples/run_external_trial_example.py --config {config} --output {output}",
        "RAGDX_LANGCHAIN_RUNNER_CMD": "python examples/run_langchain_trial.py --config {config} --output {output}",
        "RAGDX_LLAMAINDEX_RUNNER_CMD": "python examples/run_llamaindex_trial.py --config {config} --output {output}",
    }
    print(json.dumps(templates, indent=2))


__all__ = ["monitor_session", "optimize", "show_runner_templates"]
