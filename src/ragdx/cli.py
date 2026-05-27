"""Command-line interface for the ragdx library.

Subcommands cover the full diagnose → plan → optimize → save → inspect
workflow plus configuration helpers. LLM-backed features delegate to
:mod:`ragdx.llm`, which selects a provider via :class:`ragdx.config.LLMSettings`.

Run ``ragdx --help`` for a full command listing. Use ``ragdx show-config``
to print the effective configuration resolved from environment variables.

"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import typer
from rich import print
from rich.table import Table

from ragdx.config import get_settings
from ragdx.core.compare import compare_results
from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.core.evaluator import UnifiedEvaluator
from ragdx.engines.llm_diagnosis import LLMDiagnosisExplainer
from ragdx.errors import LLMConfigError, RagdxError
from ragdx.llm import LLMCallable, get_llm_callable
from ragdx.optim.executor import OptimizationExecutor
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import EvaluationResult, FeedbackEvent
from ragdx.storage.run_store import RunStore
from ragdx.utils.logging import configure_logging, get_logger
from ragdx.utils.reporting import summarize_plan

logger = get_logger(__name__)

app = typer.Typer(add_completion=False, help="ragdx — RAG diagnosis & optimization toolkit.")


@app.callback()
def _root_options(
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        help="Logger level for the ragdx package (DEBUG/INFO/WARNING/ERROR). "
        "Overrides RAGDX_LOG_LEVEL for this invocation.",
    ),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        help="Optional log file path. Overrides RAGDX_LOG_FILE for this invocation.",
    ),
) -> None:
    """Global flags applied to every subcommand."""
    if log_level:
        os.environ["RAGDX_LOG_LEVEL"] = log_level.upper()
    if log_file:
        os.environ["RAGDX_LOG_FILE"] = log_file
    configure_logging(force=True)


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
    with open(path, encoding="utf-8") as f:
        return EvaluationResult(**json.load(f))


def _build_engine(use_llm: bool = False, use_both: bool = False) -> RAGDiagnosisEngine:
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
    engine = _build_engine(use_llm=use_llm, use_both=use_both)
    report = engine.diagnose(result, use_llm=use_llm, use_both=use_both)
    planner_llm = _build_llm_callable() if (use_llm_planner or use_llm or use_both) else None
    plan = OptimizationPlanner(llm_callable=planner_llm).build_plan(
        report, result=result, strategy=strategy, budget=budget
    )
    return report, plan


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


@app.command("normalize-tools")
def normalize_tools(
    ragas_json: str = typer.Option("", help="Path to a Ragas score dump JSON."),
    ragchecker_json: str = typer.Option("", help="Path to a RAGChecker score dump JSON."),
    output_json: str = typer.Option(
        "normalized_evaluation.json", help="Destination for the unified EvaluationResult."
    ),
):
    """Normalize external evaluator outputs into a unified EvaluationResult.

    Only the adapters whose input file is supplied are exercised. This
    avoids tripping the (unrelated) optional-dependency import inside the
    other adapter when the user only has one tool installed.
    """
    if not ragas_json and not ragchecker_json:
        raise typer.BadParameter(
            "Pass at least one of --ragas-json / --ragchecker-json."
        )
    evaluator = UnifiedEvaluator()
    ragas_scores = (
        json.loads(Path(ragas_json).read_text(encoding="utf-8")) if ragas_json else None
    )
    ragchecker_scores = (
        json.loads(Path(ragchecker_json).read_text(encoding="utf-8")) if ragchecker_json else None
    )
    result = evaluator.evaluate(
        [],
        ragas_scores=ragas_scores,
        ragchecker_scores=ragchecker_scores,
        use_ragas=ragas_scores is not None,
        use_ragchecker=ragchecker_scores is not None,
    )
    Path(output_json).write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote {output_json}")


@app.command()
def dashboard():
    """Launch the Streamlit dashboard for the local RunStore."""
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__).parent / "ui" / "dashboard.py")],
        check=False,
    )


@app.command("experiment-dashboard")
def experiment_dashboard(
    bundle: str = typer.Option(
        "",
        "--bundle",
        help="Path to a ragdx experiment bundle (result.json). Falls back to "
        ".ragdx_experiment/result.json, then docs/examples/pdf_no_gt_result.json.",
    ),
    port: int = typer.Option(
        8501, "--port", help="Streamlit server port.",
    ),
):
    """Launch the generic Streamlit dashboard for ``ragdx experiment`` bundles.

    Renders any ``schema_version: 1`` bundle. Legacy bundles are
    auto-upgraded via ``migrate_legacy_bundle``.
    """
    script = Path(__file__).parent / "ui" / "experiment_dashboard.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(script),
        "--server.port", str(port),
    ]
    if bundle:
        cmd += ["--", "--bundle", bundle]
    subprocess.run(cmd, check=False)


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


@app.command("experiment")
def experiment(
    corpus: str = typer.Argument(
        ...,
        help="Corpus source. One of: "
        "(a) HuggingFace dataset name like 'explodinggradients/amnesty_qa', "
        "(b) path to a .pdf file, "
        "(c) path to a .jsonl corpus file ({text, source?} per line).",
    ),
    has_gt: bool = typer.Option(
        False,
        "--has-gt/--no-gt",
        help="Whether the input carries ground-truth answers.",
    ),
    mode: str = typer.Option(
        "auto",
        "--mode",
        help="Experiment mode: 'with_gt', 'no_gt', 'both', or 'auto'. "
        "'with_gt' / 'both' require --has-gt. 'auto' picks 'both' when --has-gt, else 'no_gt'.",
    ),
    questions_path: str | None = typer.Option(
        None,
        "--questions",
        help="JSONL with {question, ground_truth, contexts?} per line. "
        "Required when --has-gt and the corpus is a PDF or plain JSONL.",
    ),
    n_questions: int = typer.Option(5, "--n-questions", help="How many records to use."),
    n_bo_trials: int = typer.Option(8, "--bo-trials", help="Bayesian search trial budget."),
    n_bo_init: int = typer.Option(3, "--bo-init", help="Random initialisation rounds for BO."),
    output_dir: str = typer.Option(
        ".ragdx_experiment",
        "--output-dir",
        help="Directory for result.json (also picked up by the matching Streamlit dashboard).",
    ),
    api_key: str | None = typer.Option(
        None, "--api-key",
        help="LLM API key. Falls back to ZHIPU_API_KEY / OPENAI_API_KEY env vars.",
    ),
    api_base: str = typer.Option(
        "https://open.bigmodel.cn/api/paas/v4", "--api-base",
        help="OpenAI-compatible base URL (default: Zhipu).",
    ),
    model: str = typer.Option(
        "openai/glm-4-flash", "--model",
        help="LiteLLM model identifier.",
    ),
    seed: int = typer.Option(7, "--seed"),
    save_run: bool = typer.Option(
        False, "--save-run",
        help="Also persist the experiment as a Run in the local RunStore "
        "so it shows up in `ragdx runs`.",
    ),
    name: str = typer.Option("", "--name", help="Optional name for the saved run."),
    no_save: bool = typer.Option(
        False, "--no-save",
        help="Skip writing result.json (only return in-memory).",
    ),
):
    """Run the complete end-to-end RAG optimization experiment.

    Pipeline: load corpus -> resolve / synthesize questions -> Bayesian
    RAG-config search -> DSPy before/after at the winner config -> evaluate
    everything with the production composite objective. Writes a
    ``schema_version: 1`` bundle that ``ragdx experiment-dashboard``
    renders directly.

    Examples::

        # With-GT, HuggingFace dataset, run both modes side-by-side
        ragdx experiment explodinggradients/amnesty_qa --has-gt --mode both

        # No-GT PDF: questions synthesised from the corpus
        ragdx experiment <path/to/your.pdf> --no-gt

        # With-GT PDF: external labelled questions file
        ragdx experiment <path/to/report.pdf> --has-gt \\
            --questions data/labelled_qa.jsonl --mode with_gt
    """
    if mode not in {"with_gt", "no_gt", "both", "auto"}:
        raise typer.BadParameter("mode must be one of: with_gt, no_gt, both, auto")

    # Import lazily so `ragdx --help` doesn't pull in dspy / langchain.
    from ragdx.experiments import run_experiment

    try:
        result = run_experiment(
            corpus=corpus,
            has_gt=has_gt,
            mode=mode,  # type: ignore[arg-type]
            questions_path=questions_path,
            n_questions=n_questions,
            n_bo_trials=n_bo_trials,
            n_bo_init=n_bo_init,
            output_dir=output_dir,
            api_key=api_key,
            api_base=api_base,
            model=model,
            seed=seed,
            save=not no_save,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    # Render a concise summary table (full bundle is in result.json).
    table = Table(title="Experiment summary", show_lines=False)
    table.add_column("mode", style="cyan")
    table.add_column("best top_k", justify="right")
    table.add_column("chunk_size", justify="right")
    table.add_column("composite (BO winner)", justify="right")
    table.add_column("DSPy composite Δ", justify="right")

    meta = result.bundle.get("meta") or {}
    modes_run = meta.get("modes_run") or []
    bayes_search = result.bundle.get("bayes_search") or {}
    dspy_a_b = result.bundle.get("dspy_a_b") or {}

    def _row_for_mode(mode_key: str) -> tuple[str, str, str, str, str]:
        bo = bayes_search.get(mode_key, {})
        ba = dspy_a_b.get(mode_key, {})
        best_params = (bo or {}).get("best_params") or {}
        best_comp = (bo or {}).get("best_composite")
        comp = (ba or {}).get("composite") or {}
        delta = comp.get("delta")
        return (
            mode_key,
            str(best_params.get("top_k", "—")),
            str(best_params.get("chunk_size", "—")),
            f"{best_comp:.3f}" if isinstance(best_comp, (int, float)) else "—",
            f"{delta:+.3f}" if isinstance(delta, (int, float)) else "—",
        )

    for m in modes_run:
        table.add_row(*_row_for_mode(m))
    print(table)
    print(f"[green]Bundle written to[/green] {result.output_path}")
    if save_run:
        # Best-effort: synthesise an EvaluationResult from the winner scores
        # so the run shows up alongside the legacy ragdx save/optimize ones.
        from ragdx.schemas.models import EvaluationResult as _ER

        first_mode = modes_run[0] if modes_run else None
        bo = bayes_search.get(first_mode, {}) if first_mode else {}
        winner_scores = (bo or {}).get("best_scores") or {}
        baseline_result = _ER(
            retrieval={k: v for k, v in winner_scores.items() if "context" in k},
            generation={k: v for k, v in winner_scores.items() if k in {"faithfulness", "answer_relevancy", "response_relevancy"}},
            e2e={k: v for k, v in winner_scores.items() if k in {"answer_correctness", "answer_accuracy"}},
            metadata={
                "experiment": True,
                "corpus": corpus,
                "mode": mode,
                "bundle_path": str(result.output_path),
            },
        )
        report = RAGDiagnosisEngine().diagnose(baseline_result)
        opt_plan = OptimizationPlanner().build_plan(
            report, result=baseline_result, strategy="bayesian", budget=n_bo_trials,
        )
        store = _store()
        run = store.save_run(baseline_result, report, opt_plan, name=name or None)
        print(f"[green]Saved run[/green] {run.run_id}")


@app.command("show-config")
def show_config():
    """Print the effective ragdx configuration resolved from the environment."""
    settings = get_settings()
    payload = {
        "storage": {
            "root": str(settings.storage.root),
        },
        "llm": {
            "provider": settings.llm.provider,
            "model": settings.llm.model,
            "has_api_key": bool(settings.llm.api_key),
            "base_url": settings.llm.base_url,
            "api_version": settings.llm.api_version,
            "temperature": settings.llm.temperature,
            "max_tokens": settings.llm.max_tokens,
            "timeout": settings.llm.timeout,
        },
        "execution": {
            "strict_execute": settings.execution.strict_execute,
            "runner_timeout_sec": settings.execution.runner_timeout_sec,
            "bo_backend": settings.execution.bo_backend,
            "fallback_simulate_on_missing_runner": settings.execution.fallback_simulate_on_missing_runner,
        },
    }
    print(json.dumps(payload, indent=2))


def main() -> None:
    """Entry point used by the ``ragdx`` console script."""
    try:
        app()
    except LLMConfigError as exc:
        print(f"[red]LLM configuration error:[/red] {exc}", file=sys.stderr)
        sys.exit(2)
    except RagdxError as exc:
        print(f"[red]ragdx error:[/red] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
