"""``ragdx save`` / ``runs`` / ``sessions`` / ``compare`` /
``attach-feedback`` / ``feedback-summary`` / ``export-report``.

The RunStore-facing commands -- persistence, listing, comparison,
markdown export. Keep these grouped because they share the same
``_store()`` and ``_load_eval()`` helpers and naturally interleave
in user workflows ("save then compare then export").
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print
from rich.table import Table

from ragdx.cli._app import app
from ragdx.cli._shared import _diagnose_and_plan, _load_eval, _store
from ragdx.core.compare import compare_results
from ragdx.schemas.models import FeedbackEvent


@app.command()
def save(
    eval_json: str = typer.Argument(..., help="Path to an evaluation results JSON file."),
    name: str = typer.Option("", help="Optional human-readable name."),
    tags: str = typer.Option("", help="Comma-separated tags to attach to the saved run."),
    notes: str = typer.Option("", help="Free-form notes attached to the saved run."),
    baseline_run_id: str = typer.Option("", help="Run id to attach as the baseline."),
    use_llm: bool = typer.Option(False, help="Use LLM diagnosis instead of rule-based diagnosis."),
    use_both: bool = typer.Option(False, help="Rule-based + LLM diagnosis, then LLM synthesis."),
    use_llm_planner: bool = typer.Option(False, help="Refine the optimization plan with an LLM."),
):
    """Run diagnose+plan and persist the run."""
    if use_llm and use_both:
        raise typer.BadParameter("Use either --use-llm or --use-both, not both.")
    result = _load_eval(eval_json)
    report, opt_plan = _diagnose_and_plan(
        result, use_llm=use_llm, use_both=use_both, use_llm_planner=use_llm_planner
    )
    run = _store().save_run(
        result,
        report,
        opt_plan,
        name=name or None,
        tags=[x.strip() for x in tags.split(",") if x.strip()],
        notes=notes,
        baseline_run_id=baseline_run_id or None,
    )
    print(run.model_dump_json(indent=2))


@app.command()
def compare(
    current_eval_json: str = typer.Argument(..., help="Path to the current EvaluationResult JSON."),
    baseline_eval_json: str = typer.Argument(..., help="Path to the baseline EvaluationResult JSON."),
):
    """Compare two evaluation results metric-by-metric."""
    current = _load_eval(current_eval_json)
    baseline = _load_eval(baseline_eval_json)
    comparisons = compare_results(current, baseline)
    table = Table(title="Metric comparison")
    table.add_column("Metric")
    table.add_column("Current")
    table.add_column("Baseline")
    table.add_column("Delta")
    table.add_column("Direction")
    for c in comparisons:
        table.add_row(
            c.metric, f"{c.current:.4f}", f"{c.baseline:.4f}", f"{c.delta:+.4f}", c.direction
        )
    print(table)


@app.command("attach-feedback")
def attach_feedback(
    run_id: str = typer.Argument(..., help="Target run id."),
    feedback_json: str = typer.Argument(..., help="Path to a single FeedbackEvent or list of FeedbackEvent JSON."),
):
    """Attach FeedbackEvent payloads to a previously saved run."""
    payload = json.loads(Path(feedback_json).read_text(encoding="utf-8"))
    events = payload if isinstance(payload, list) else [payload]
    typed = [FeedbackEvent(**item) for item in events]
    run = _store().attach_feedback_to_run(run_id, typed)
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "feedback_attached": len(typed),
                "total_feedback": len(run.evaluation.feedback_events),
            },
            indent=2,
        )
    )


@app.command("feedback-summary")
def feedback_summary():
    """Print an aggregate summary of feedback events across stored runs."""
    print(json.dumps(_store().feedback_summary(), indent=2))


@app.command()
def runs():
    """List all saved runs."""
    rows = _store().list_runs()
    table = Table(title="Saved ragdx runs")
    table.add_column("Run ID")
    table.add_column("Created")
    table.add_column("Name")
    table.add_column("Baseline")
    table.add_column("Latest session")
    table.add_column("Feedback")
    table.add_column("Tags")
    for r in rows:
        table.add_row(
            r.run_id,
            r.created_at,
            r.name,
            r.baseline_run_id or "",
            r.latest_session_id or "",
            str(len(r.evaluation.feedback_events)),
            ", ".join(r.tags),
        )
    print(table)


@app.command()
def sessions():
    """List all saved optimization sessions."""
    rows = _store().list_sessions()
    table = Table(title="Saved ragdx optimization sessions")
    table.add_column("Session ID")
    table.add_column("Created")
    table.add_column("Run ID")
    table.add_column("Strategy")
    table.add_column("Mode")
    table.add_column("Status")
    table.add_column("Progress")
    for s in rows:
        table.add_row(
            s.session_id,
            s.created_at,
            s.run_id or "",
            s.strategy,
            s.mode,
            s.status,
            f"{s.completed_trials}/{s.total_trials}",
        )
    print(table)


@app.command("export-report")
def export_report(
    run_id: str = typer.Argument(..., help="Saved run id."),
    output_md: str = typer.Argument(..., help="Output Markdown file path."),
):
    """Export a Markdown report for a saved run."""
    path = _store().export_markdown(run_id, output_md)
    print(f"Wrote {path}")


__all__ = [
    "attach_feedback",
    "compare",
    "export_report",
    "feedback_summary",
    "runs",
    "save",
    "sessions",
]
