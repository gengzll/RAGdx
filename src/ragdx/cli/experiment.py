"""``ragdx experiment`` / ``experiment-dashboard`` / ``experiment-report``.

The end-to-end experiment workflow's CLI surface. Kept together
because ``experiment`` produces a bundle that the other two render.
``experiment`` is the heavy command (drives BO + DSPy + ragas);
``experiment-dashboard`` and ``experiment-report`` are thin wrappers
around streamlit / the HTML renderer.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer
from rich import print
from rich.table import Table

from ragdx.cli._app import app
from ragdx.cli._shared import _store
from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.optim.planner import OptimizationPlanner


@app.command("experiment")
def experiment(
    corpus: str = typer.Argument(
        ...,
        help="Corpus source(s). One of: "
        "(a) HuggingFace dataset name like 'explodinggradients/amnesty_qa', "
        "(b) path to a .pdf file, "
        "(c) path to a .jsonl corpus file ({text, source?} per line), "
        "(d) **comma-separated list** of any of the above for "
        "multi-corpus runs (e.g. 'a.pdf,b.pdf' — chunks pooled, "
        "questions from --questions or synthesized).",
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
    llm_max_concurrent: int = typer.Option(
        2, "--llm-max-concurrent",
        help="Max in-flight LLM calls per evaluation batch. Default 2 for "
        "strict rate-limited endpoints (GLM-4-Flash); bump to 8-16 for "
        "OpenAI / Anthropic. Propagates to ragas + DSPy worker pools.",
    ),
    llm_max_retries: int = typer.Option(
        5, "--llm-max-retries",
        help="Per-call transport-layer retry budget. Propagates to the "
        "openai client (ragas judge) and litellm (BO generation, DSPy).",
    ),
    system_instruction: str = typer.Option(
        "", "--system-instruction",
        help="RAG system prompt shared by BO generation and the DSPy "
        "baseline. Defaults to ragdx's generic 'use only context, be "
        "concise' prompt. Override for domain-specific guidance "
        "(legal / medical / etc.). Mutually exclusive with "
        "--system-instruction-file.",
    ),
    system_instruction_file: str = typer.Option(
        "", "--system-instruction-file",
        help="Path to a text file whose contents are used as the system "
        "instruction. Useful for long / multi-line prompts. Mutually "
        "exclusive with --system-instruction.",
    ),
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
    resume: str = typer.Option(
        "", "--resume",
        help="Resume an interrupted experiment. Pass the experiment "
        "group id printed at the original run's start, or ``auto`` to "
        "pick the most recently interrupted one. Completed stages "
        "(BO / DSPy, per mode) replay from their checkpoints without "
        "LLM calls; the interrupted stage continues from its last "
        "saved trial / phase.",
    ),
    no_checkpoint: bool = typer.Option(
        False, "--no-checkpoint",
        help="Disable per-trial / per-phase checkpointing for this run.",
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

    # Resolve --system-instruction / --system-instruction-file (mutually
    # exclusive, both optional; None falls back to ragdx's default).
    if system_instruction and system_instruction_file:
        raise typer.BadParameter(
            "Use either --system-instruction OR --system-instruction-file, not both."
        )
    resolved_instruction: str | None = None
    if system_instruction:
        resolved_instruction = system_instruction
    elif system_instruction_file:
        sip = Path(system_instruction_file)
        if not sip.exists():
            raise typer.BadParameter(
                f"--system-instruction-file path not found: {sip}"
            )
        resolved_instruction = sip.read_text(encoding="utf-8").strip()
        if not resolved_instruction:
            raise typer.BadParameter(
                f"--system-instruction-file {sip} is empty after stripping whitespace."
            )

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
            llm_max_concurrent=llm_max_concurrent,
            llm_max_retries=llm_max_retries,
            system_instruction=resolved_instruction,
            save=not no_save,
            resume=resume,
            no_checkpoint=no_checkpoint,
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
    # ``Path(__file__).parents[1]`` reaches ``ragdx/``; the dashboard
    # script lives at ``ragdx/ui/experiment_dashboard.py``.
    script = Path(__file__).resolve().parents[1] / "ui" / "experiment_dashboard.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(script),
        "--server.port", str(port),
    ]
    if bundle:
        cmd += ["--", "--bundle", bundle]
    subprocess.run(cmd, check=False)


@app.command("ui")
def ui(
    port: int = typer.Option(
        8501, "--port", help="Streamlit server port.",
    ),
):
    """Launch the ragdx studio: upload a PDF (+ optional Excel/CSV
    ground-truth), run the end-to-end experiment with live progress,
    then view and download the report.

    This is the recommended entry point for interactive use::

        ragdx ui
    """
    script = Path(__file__).resolve().parents[1] / "ui" / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(script),
        "--server.port", str(port),
    ]
    subprocess.run(cmd, check=False)


@app.command("experiment-report")
def experiment_report(
    bundle: str = typer.Argument(
        ..., help="Path to a ragdx experiment bundle (result.json).",
    ),
    output: str = typer.Option(
        "experiment_report.html", "--output", "-o",
        help="HTML output path. Open in a browser; Ctrl-P → Save as PDF "
        "for a PDF copy.",
    ),
    title: str = typer.Option(
        "", "--title",
        help="Optional custom report title (default: derived from "
        "model + modes_run).",
    ),
):
    """Render an experiment bundle as a self-contained HTML report.

    Produces a single static HTML file with embedded Plotly charts --
    no streamlit runtime required to view it. Mirrors the content of
    ``ragdx experiment-dashboard`` but as a deliverable artifact
    (shareable, version-controllable, PDF-exportable via the browser).
    """
    from ragdx.ui.experiment_report import render_report

    bundle_path = Path(bundle)
    if not bundle_path.exists():
        raise typer.BadParameter(f"Bundle not found: {bundle_path}")
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    html = render_report(data, title=title or None)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"[green]Wrote report to[/green] {out_path}")
    print(f"  Open in a browser: file:///{out_path.resolve().as_posix()}")
    print("  PDF export: Ctrl-P in the browser, choose 'Save as PDF'.")


__all__ = ["experiment", "experiment_dashboard", "experiment_report", "ui"]
