"""``ragdx workspace ...`` -- per-experiment workspace commands.

A *workspace* is a folder under ``workspaces/<name>/`` that owns one
experiment: its RAG config, its questions, every artifact every
command produces, plus a small ``workspace.yaml`` manifest that
records what's been run. The verbs are short because the workspace
already knows the answers to ``--config``, ``--questions``,
``--corpus``, and ``--baseline-run-id``:

* ``ragdx workspace init <name>``           -- create the folder
* ``ragdx workspace eval <name>``           -- baseline evaluation
* ``ragdx workspace diagnose <name>``       -- diagnose latest eval
* ``ragdx workspace tune-rag <name>``       -- chunker / retriever / joint
* ``ragdx workspace tune-prompt <name>``    -- DSPy prompt + demos
* ``ragdx workspace report <name>``         -- render HTML
* ``ragdx workspace list``                  -- show every workspace
* ``ragdx workspace show <name>``           -- show one's history
* ``ragdx workspace compare <a> <b> [c..]`` -- cross-experiment delta

The legacy ``ragdx evaluate / diagnose / tune`` commands keep working
identically -- workspaces are pure convenience. Under the hood every
verb here delegates to the same workflow functions the legacy
commands use, so behaviour stays consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich import print
from rich.table import Table

from ragdx.cli._app import app
from ragdx.workspace import (
    Workspace,
    WorkspaceError,
    init_workspace,
    list_workspaces,
    load_workspace,
    workspaces_root,
)

_ws_app = typer.Typer(
    help="Per-experiment workspaces (config + questions + artifacts in one folder).",
    no_args_is_help=True,
)


# =====================================================================
# init / list / show
# =====================================================================
@_ws_app.command("init")
def init(
    name: str = typer.Argument(..., help="Workspace name (becomes the folder name)."),
    config: str = typer.Option(
        "", "--config", "-c",
        help="Path to an existing RAGConfig YAML. Copied into the "
        "workspace as rag_config.yaml. Omit to start with a stub.",
    ),
    questions: str = typer.Option(
        "", "--questions", "-q",
        help="Path to a JSONL eval suite. Copied into the workspace as "
        "questions.jsonl. Omit to start with a stub.",
    ),
    corpus: str = typer.Option(
        "", "--corpus",
        help="Optional corpus path override (otherwise resolved from "
        "the YAML's corpus.path field).",
    ),
    evaluator: str = typer.Option(
        "ragas", "--evaluator",
        help="Which evaluation library this workspace uses by default: "
        "ragas or deepeval.",
    ),
    mode: str = typer.Option(
        "auto", "--mode",
        help="GT mode: auto / with_gt / no_gt.",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite",
        help="If the workspace already exists, recreate its manifest.",
    ),
):
    """Create a new workspace folder."""
    import shutil

    if evaluator not in {"ragas", "deepeval"}:
        raise typer.BadParameter(
            f"--evaluator must be 'ragas' or 'deepeval', got {evaluator!r}."
        )
    if mode not in {"auto", "with_gt", "no_gt"}:
        raise typer.BadParameter(
            f"--mode must be one of auto/with_gt/no_gt, got {mode!r}."
        )

    ws = init_workspace(
        name,
        evaluator=evaluator,
        mode=mode,
        corpus=corpus or None,
        overwrite=overwrite,
    )

    # Best-effort: copy user-supplied config / questions into the
    # workspace so the workspace is self-contained.
    if config:
        src = Path(config)
        if not src.exists():
            raise typer.BadParameter(f"--config not found: {src}")
        shutil.copy2(src, ws.rag_config_path())
        print(f"[dim]Copied[/dim] {src} -> {ws.rag_config_path().name}")
    if questions:
        src = Path(questions)
        if not src.exists():
            raise typer.BadParameter(f"--questions not found: {src}")
        shutil.copy2(src, ws.questions_path())
        print(f"[dim]Copied[/dim] {src} -> {ws.questions_path().name}")

    print(f"[green]Initialized workspace[/green] {ws.name} -> {ws.root}")
    if not config:
        print(
            f"[yellow]Note:[/yellow] no --config given. Place "
            f"{ws.rag_config_path().name} in the workspace before "
            f"running `ragdx workspace eval {name}`."
        )
    if not questions:
        print(
            f"[yellow]Note:[/yellow] no --questions given. Place "
            f"{ws.questions_path().name} in the workspace too."
        )


@_ws_app.command("list")
def list_cmd() -> None:
    """List every workspace under ``workspaces/`` (or ``$RAGDX_WORKSPACES``)."""
    names = list_workspaces()
    if not names:
        print(f"[dim]No workspaces found under {workspaces_root()}.[/dim]")
        return
    table = Table(title="ragdx workspaces", show_lines=False)
    table.add_column("name", style="cyan")
    table.add_column("evaluator")
    table.add_column("history", justify="right")
    table.add_column("latest")
    for n in names:
        try:
            ws = load_workspace(n)
            last = ws.history[-1].command if ws.history else "-"
            table.add_row(n, ws.evaluator, str(len(ws.history)), last)
        except WorkspaceError:
            table.add_row(n, "?", "?", "?")
    print(table)
    print(f"[dim]Root: {workspaces_root()}[/dim]")


@_ws_app.command("show")
def show(
    name: str = typer.Argument(..., help="Workspace name."),
) -> None:
    """Show one workspace's config + history."""
    ws = load_workspace(name)
    print(f"[bold]{ws.name}[/bold]  [dim]{ws.root}[/dim]")
    print(f"  created_at: {ws.created_at}")
    print(f"  evaluator:  {ws.evaluator}")
    print(f"  mode:       {ws.mode}")
    print(f"  rag_config: {ws.rag_config}")
    print(f"  questions:  {ws.questions}")
    if ws.corpus:
        print(f"  corpus:     {ws.corpus}")
    if ws.current:
        print("[bold]Current pointers:[/bold]")
        for k, v in ws.current.items():
            print(f"  {k}: {v}")
    if ws.notes:
        print(f"[bold]Notes:[/bold] {ws.notes}")
    if ws.history:
        print("[bold]History:[/bold]")
        for h in ws.history:
            out = f" -> {h.output}" if h.output else ""
            run = f" (run_id={h.run_id})" if h.run_id else ""
            print(f"  {h.ts}  {h.command}{out}{run}")
    else:
        print("[dim]No commands run yet.[/dim]")


# =====================================================================
# Shared resolver: API key from env (same fallback chain CLI uses)
# =====================================================================
def _resolve_api_key(rag_config: Any) -> None:
    """Mutate ``rag_config.generator.api_key`` from env if unset."""
    import os
    if rag_config.generator.api_key:
        return
    rag_config.generator.api_key = (
        os.environ.get("ZHIPU_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not rag_config.generator.api_key:
        raise typer.BadParameter(
            "No api_key resolved. Set generator.api_key in the YAML "
            "or the ZHIPU_API_KEY / OPENAI_API_KEY env var."
        )


def _require_workspace_files(ws: Workspace) -> None:
    """Verify the workspace's rag_config + questions actually exist."""
    if not ws.rag_config_path().exists():
        raise typer.BadParameter(
            f"Workspace {ws.name!r} has no {ws.rag_config}. Drop the "
            f"YAML at {ws.rag_config_path()} or re-run "
            f"`ragdx workspace init {ws.name} --config ...`."
        )
    if not ws.questions_path().exists():
        raise typer.BadParameter(
            f"Workspace {ws.name!r} has no {ws.questions}. Drop the "
            f"JSONL at {ws.questions_path()} or re-run "
            f"`ragdx workspace init {ws.name} --questions ...`."
        )


def _scope_storage_to_workspace(ws: Workspace) -> None:
    """Root the RunStore + causal-prior store inside the workspace.

    Without this, every workspace's ``diagnose`` writes its learned
    causal posteriors back to the *global* ``.ragdx/causal/priors.json``.
    Because the prior-update is monotonic toward the 0.95 clamp, a few
    cross-experiment diagnose calls saturate every node -- so baseline
    and optimized causal graphs end up visually identical (observed in
    the first demo pass: all nodes pinned at ~0.96 regardless of the
    actual metrics).

    Scoping ``RAGDX_ROOT`` to ``<workspace>/.ragdx`` gives each
    workspace its own causal-prior history (and RunStore), matching the
    "one workspace = one experiment" model: the priors learn from *this*
    experiment's diagnose calls only, starting from clean
    ``base_priors``. ``get_settings()`` reads the env fresh each call,
    so setting it here -- before any store is built -- takes effect.
    Honours an explicit ``RAGDX_ROOT`` the user already set (we don't
    override a deliberate override).
    """
    import os
    if os.environ.get("RAGDX_ROOT"):
        return
    os.environ["RAGDX_ROOT"] = str((ws.root / ".ragdx").resolve())


# =====================================================================
# eval
# =====================================================================
@_ws_app.command("eval")
def eval_cmd(
    name: str = typer.Argument(..., help="Workspace name."),
    evaluator: str = typer.Option(
        "", "--evaluator",
        help="Override the workspace's default evaluator for this run.",
    ),
    output: str = typer.Option(
        "", "--output", "-o",
        help="Output filename (relative to the workspace). "
        "Default: ``baseline.eval.json`` when no eval exists yet, "
        "else ``post_<n>.eval.json``.",
    ),
    save: bool = typer.Option(
        True, "--save/--no-save",
        help="Persist to the project RunStore (default: yes).",
    ),
) -> None:
    """Evaluate the workspace's RAG config against its questions."""
    from ragdx.cli._shared import _diagnose_and_plan, _store
    from ragdx.experiments import (
        ExperimentConfig,
        _load_corpus_and_records,
        _load_jsonl_questions,
        _normalize_corpus,
    )
    from ragdx.runtime.factories import build_runtime
    from ragdx.schemas.rag_config import RAGConfig
    from ragdx.workflows.evaluate import _scores_to_evaluation_result  # noqa: F401
    from ragdx.workflows.evaluate import evaluate as workflow_evaluate

    ws = load_workspace(name)
    _require_workspace_files(ws)
    _scope_storage_to_workspace(ws)

    chosen_evaluator = evaluator or ws.evaluator
    if chosen_evaluator not in {"ragas", "deepeval"}:
        raise typer.BadParameter(
            f"--evaluator must be 'ragas' or 'deepeval', got {chosen_evaluator!r}."
        )

    out_name = output or (
        "baseline.eval.json" if not ws.current.get("latest_eval")
        else f"post_{len([h for h in ws.history if h.command == 'eval'])}.eval.json"
    )
    out_path = ws.path(out_name)

    rag_config = RAGConfig.from_yaml(ws.rag_config_path())
    _resolve_api_key(rag_config)

    corpus_path = ws.corpus_path()
    if corpus_path is None:
        raise typer.BadParameter(
            f"Workspace {ws.name!r} has no corpus path. Set --corpus in "
            f"`init`, set ``corpus.path`` in the YAML, or pass it through."
        )

    records = _load_jsonl_questions(ws.questions_path())
    if not records:
        raise typer.BadParameter(f"{ws.questions_path()} contained zero records.")

    # Build a stub ExperimentConfig just for the corpus loader.
    has_gt = any((r.ground_truth or "").strip() for r in records)
    stub_cfg = ExperimentConfig(
        corpus=_normalize_corpus(str(corpus_path))
        if "," in str(corpus_path)
        else str(corpus_path),
        has_gt=has_gt, mode="auto",
        questions_path=str(ws.questions_path()),
        api_key=rag_config.generator.api_key,
        api_base=rag_config.generator.api_base,
        model=rag_config.generator.model,
    )
    runtime = build_runtime(rag_config)
    chunks, _records_unused, source_meta = _load_corpus_and_records(
        stub_cfg, runtime,
    )

    metadata: dict[str, Any] = {
        "workspace": ws.name,
        "config_path": str(ws.rag_config_path()),
        "questions_path": str(ws.questions_path()),
        "corpus": str(corpus_path),
        "corpus_source": source_meta,
    }

    result = workflow_evaluate(
        rag_config,
        chunks=chunks, records=records,
        runtime=runtime, metadata=metadata,
        evaluator=chosen_evaluator,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        result.model_dump_json(indent=2), encoding="utf-8",
    )

    run_id: str | None = None
    if save:
        rep, opt_plan = _diagnose_and_plan(result)
        run = _store().save_run(
            result, rep, opt_plan,
            name=f"{ws.name}/{out_name}",
            rag_config=rag_config.scrubbed_for_commit(),
        )
        run_id = run.run_id

    ws.log("eval", output=out_path, run_id=run_id,
           extra={"evaluator": chosen_evaluator})
    print(f"[green]Wrote[/green] {out_path}")
    if run_id:
        print(f"[green]Saved run[/green] {run_id}")


# =====================================================================
# diagnose
# =====================================================================
@_ws_app.command("diagnose")
def diagnose_cmd(
    name: str = typer.Argument(..., help="Workspace name."),
    use_llm: bool = typer.Option(
        False, "--use-llm",
        help="Use LLM diagnosis instead of pure rule-based.",
    ),
    use_both: bool = typer.Option(
        False, "--use-both",
        help="Rule-based + LLM, then LLM-synthesised combined report.",
    ),
    output: str = typer.Option(
        "", "--output", "-o",
        help="Output filename (relative to workspace). "
        "Default: ``<source>.diagnose.json``.",
    ),
    eval_file: str = typer.Option(
        "", "--eval-file",
        help="Specific evaluation JSON to diagnose. Default: the "
        "latest eval logged in this workspace.",
    ),
) -> None:
    """Diagnose the workspace's latest (or specified) evaluation."""
    if use_llm and use_both:
        raise typer.BadParameter("Use either --use-llm or --use-both, not both.")

    from ragdx.cli._shared import _diagnose_and_plan, _load_eval_or_latest

    ws = load_workspace(name)
    _scope_storage_to_workspace(ws)

    if eval_file:
        eval_path = ws.path(eval_file)
    else:
        latest = ws.latest("eval")
        if latest is None:
            raise typer.BadParameter(
                f"Workspace {ws.name!r} has no evaluation yet. Run "
                f"`ragdx workspace eval {ws.name}` first."
            )
        eval_path = latest

    if not eval_path.exists():
        raise typer.BadParameter(f"Evaluation file not found: {eval_path}")

    result, _ = _load_eval_or_latest(str(eval_path))

    # History-aware escalation: translate the workspace's tune history
    # into the list of optimization candidates already applied, so the
    # analyzer can escalate its advice when a defect persists. A
    # ``tune-rag`` run counts as ``autorag_pipeline_search`` (the
    # retrieval-side candidate); a ``tune-prompt`` run counts as
    # ``dspy_prompt_optimization``. Only diagnose calls that come AFTER
    # at least one tune see a non-empty history -- so the very first
    # baseline diagnose gets base-level advice, and a re-diagnose after
    # an unsuccessful tune gets escalated advice.
    _hist_map = {
        "tune-rag": "autorag_pipeline_search",
        "tune-prompt": "dspy_prompt_optimization",
    }
    optimization_history = [
        _hist_map[h.command]
        for h in ws.history
        if h.command in _hist_map
    ]

    report, _plan = _diagnose_and_plan(
        result, use_llm=use_llm, use_both=use_both,
        optimization_history=optimization_history,
    )

    out_name = output or f"{eval_path.stem}.diagnose.json"
    out_path = ws.path(out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        report.model_dump_json(indent=2), encoding="utf-8",
    )

    mode = "both" if use_both else "llm" if use_llm else "rule"
    ws.log("diagnose", output=out_path,
           extra={"mode": mode, "source_eval": str(eval_path.name)})
    print(f"[green]Wrote[/green] {out_path} [dim](mode={mode})[/dim]")

    # Quick stdout summary so the user sees the headline without
    # having to grep the JSON.
    print(f"\n[bold]Summary:[/bold] {report.summary}")
    if report.optimization_candidates:
        print(f"[bold]Candidates:[/bold] {', '.join(report.optimization_candidates)}")


# =====================================================================
# tune-rag
# =====================================================================
@_ws_app.command("tune-rag")
def tune_rag_cmd(
    name: str = typer.Argument(..., help="Workspace name."),
    stage: str = typer.Option(
        "joint", "--stage",
        help="Which RAG-side stage to tune: chunking / retrieval / joint. "
        "``joint`` (default) varies all three of chunk_size / "
        "chunk_overlap / top_k via Bayesian search.",
    ),
    budget: int = typer.Option(
        8, "--budget", "-b",
        help="BO trial budget. Higher = better-found winner but more LLM calls.",
    ),
    output: str = typer.Option(
        "", "--output", "-o",
        help="Output filename (relative to workspace). "
        "Default: ``tune_rag_<stage>.json``.",
    ),
    resume: str = typer.Option(
        "", "--resume",
        help="Resume an interrupted run. Pass a checkpoint id, or "
        "``auto`` to pick the latest interrupted tune-rag checkpoint "
        "in this workspace.",
    ),
    no_checkpoint: bool = typer.Option(
        False, "--no-checkpoint",
        help="Disable per-trial checkpointing for this run.",
    ),
) -> None:
    """Tune the retrieval side (chunker / retriever / joint) of the workspace's RAG."""
    if stage not in {"chunking", "retrieval", "joint"}:
        raise typer.BadParameter(
            f"--stage for tune-rag must be chunking / retrieval / joint, "
            f"got {stage!r}. For prompt tuning use `tune-prompt`."
        )
    _delegate_to_tune(
        name=name, stage=stage, budget=budget,
        output=output or f"tune_rag_{stage}.json",
        verb="tune-rag",
        resume=resume, no_checkpoint=no_checkpoint,
    )


# =====================================================================
# tune-prompt
# =====================================================================
@_ws_app.command("tune-prompt")
def tune_prompt_cmd(
    name: str = typer.Argument(..., help="Workspace name."),
    dspy_optimizer: str = typer.Option(
        "gepa", "--dspy-optimizer",
        help="DSPy teleprompter: gepa (default) / mipro / copro / "
        "bootstrap_fewshot.",
    ),
    dspy_metric: str = typer.Option(
        "auto", "--dspy-metric",
        help="Inner-loop metric: auto / embed_rubric / geval / ragas / "
        "llm_judge / token_f1.",
    ),
    mipro_auto: str = typer.Option(
        "light", "--mipro-auto",
        help="MIPROv2 / GEPA budget: light / medium / heavy.",
    ),
    output: str = typer.Option(
        "", "--output", "-o",
        help="Output filename (relative to workspace). "
        "Default: ``tune_prompt_<optimizer>.json``.",
    ),
    resume: str = typer.Option(
        "", "--resume",
        help="Resume an interrupted run. Pass a checkpoint id, or "
        "``auto`` to pick the latest interrupted tune-generation "
        "checkpoint in this workspace.",
    ),
    no_checkpoint: bool = typer.Option(
        False, "--no-checkpoint",
        help="Disable per-phase checkpointing for this run.",
    ),
) -> None:
    """Tune the generator prompt (DSPy) for the workspace's RAG."""
    _delegate_to_tune(
        name=name, stage="generation", budget=8,
        output=output or f"tune_prompt_{dspy_optimizer}.json",
        verb="tune-prompt",
        dspy_optimizer=dspy_optimizer,
        dspy_metric=dspy_metric,
        mipro_auto=mipro_auto,
        resume=resume, no_checkpoint=no_checkpoint,
    )


def _delegate_to_tune(
    *,
    name: str,
    stage: str,
    budget: int,
    output: str,
    verb: str,
    dspy_optimizer: str = "gepa",
    dspy_metric: str = "auto",
    mipro_auto: str = "light",
    resume: str = "",
    no_checkpoint: bool = False,
) -> None:
    """Shared body for tune-rag / tune-prompt.

    Composes the same building blocks ``cli/tune.py::tune`` uses but
    sources every input (config / questions / corpus / baseline link)
    from the workspace, so the user types only the verb + name.

    Checkpointing: because ``_scope_storage_to_workspace`` already
    pointed ``RAGDX_ROOT`` at ``<workspace>/.ragdx``, the
    ``CheckpointStore`` (which resolves its root via settings) writes to
    ``<workspace>/.ragdx/checkpoints/``. A crash mid-BO loses at most
    one trial; resume with ``--resume auto`` (or an explicit id).
    """
    import json
    from typing import Any as _Any

    from ragdx.checkpoint import Checkpoint, CheckpointStore
    from ragdx.cli._shared import _diagnose_and_plan, _store
    from ragdx.core.diagnosis import RAGDiagnosisEngine
    from ragdx.experiments import (
        ExperimentConfig,
        _build_ragas_metrics_for_mode,
        _compare_diagnoses,
        _load_corpus_and_records,
        _load_jsonl_questions,
        _normalize_corpus,
        _synth_eval_result,
    )
    from ragdx.optim._gt_helpers import gt_mode as detect_gt_mode
    from ragdx.optim.objectives import default_objective
    from ragdx.optim.stages import (
        ChunkingOptimizer,
        GenerationOptimizer,
        JointOptimizer,
        RetrievalOptimizer,
        StageContext,
    )
    from ragdx.runtime.factories import build_runtime
    from ragdx.schemas.rag_config import RAGConfig
    from ragdx.workflows.evaluate import _scores_to_evaluation_result

    ws = load_workspace(name)
    _require_workspace_files(ws)
    _scope_storage_to_workspace(ws)
    corpus_path = ws.corpus_path()
    if corpus_path is None:
        raise typer.BadParameter(
            f"Workspace {ws.name!r} has no corpus path."
        )

    rag_config = RAGConfig.from_yaml(ws.rag_config_path())
    _resolve_api_key(rag_config)
    records = _load_jsonl_questions(ws.questions_path())
    if not records:
        raise typer.BadParameter(f"{ws.questions_path()} contained zero records.")

    stub_cfg = ExperimentConfig(
        corpus=_normalize_corpus(str(corpus_path))
        if "," in str(corpus_path)
        else str(corpus_path),
        has_gt=any((r.ground_truth or "").strip() for r in records),
        mode="auto", questions_path=str(ws.questions_path()),
        api_key=rag_config.generator.api_key,
        api_base=rag_config.generator.api_base,
        model=rag_config.generator.model,
    )
    runtime = build_runtime(rag_config)
    chunks, _records_unused, _source_meta = _load_corpus_and_records(
        stub_cfg, runtime,
    )

    eff_gt_mode = detect_gt_mode(records)
    objective = default_objective(eff_gt_mode)
    # ``ctx.metrics`` must be ragas Metric *instances* (not strings)
    # because ``_evaluate_with_ragas`` forwards them straight to
    # ``ragas.evaluate(metrics=...)``. Re-use the same helper the
    # experiment workflow uses so the metric set is consistent.
    metrics = _build_ragas_metrics_for_mode(eff_gt_mode)

    # Stage dispatch.
    stage_map = {
        "chunking": ChunkingOptimizer,
        "retrieval": RetrievalOptimizer,
        "joint": JointOptimizer,
        "generation": GenerationOptimizer,
    }
    StageClass = stage_map[stage]
    optimizer = StageClass()
    label = eff_gt_mode

    # --- Checkpoint: load (--resume) or create -----------------------
    # CheckpointStore resolves its root via settings, which we've
    # scoped to the workspace, so checkpoints live in
    # ``<workspace>/.ragdx/checkpoints/``.
    checkpoint_obj = None
    checkpoint_store = None
    if not no_checkpoint:
        checkpoint_store = CheckpointStore()
        ckpt_kind = f"tune.{stage}"
        if resume:
            target = resume
            if target.lower() in {"auto", "latest", "true"}:
                incomplete = [
                    c for c in checkpoint_store.list_incomplete()
                    if c.kind == ckpt_kind
                ]
                if not incomplete:
                    raise typer.BadParameter(
                        f"No interrupted {ckpt_kind} checkpoint to resume in "
                        f"workspace {ws.name!r}. Run `ragdx checkpoints` to see "
                        "what's available."
                    )
                target = incomplete[0].checkpoint_id
            try:
                checkpoint_obj = checkpoint_store.load(target)
            except FileNotFoundError as exc:
                raise typer.BadParameter(str(exc)) from exc
            checkpoint_obj.status = "running"
            checkpoint_obj.interrupted_reason = ""
            print(
                f"[bold]Resuming checkpoint[/bold] [cyan]{checkpoint_obj.checkpoint_id}[/cyan] "
                f"(trials_done=[cyan]{len(checkpoint_obj.trials_completed)}[/cyan], "
                f"phase=[cyan]{checkpoint_obj.generation_phase or '—'}[/cyan])"
            )
        else:
            checkpoint_obj = Checkpoint(
                kind=ckpt_kind,
                cli_args={
                    "workspace": ws.name, "stage": stage, "budget": budget,
                    "dspy_optimizer": dspy_optimizer, "dspy_metric": dspy_metric,
                    "mipro_auto": mipro_auto,
                },
                name=f"{ws.name}/{verb}",
                rag_config_yaml=rag_config.scrubbed_for_commit().to_yaml_string(),
            )
            checkpoint_store.save(checkpoint_obj)
            print(
                f"[dim]Checkpoint:[/dim] [cyan]{checkpoint_obj.checkpoint_id}[/cyan] "
                f"[dim](resume via `ragdx workspace {verb} {ws.name} --resume "
                f"{checkpoint_obj.checkpoint_id}`)[/dim]"
            )

    ctx_kwargs: dict[str, _Any] = dict(
        base_config=rag_config,
        chunks_master=chunks,
        records=records,
        objective=objective,
        metrics=metrics,
        runtime=runtime,
        label=label,
        n_bo_init=2 if stage != "generation" else 0,
        n_bo_trials=budget,
        checkpoint=checkpoint_obj,
        checkpoint_store=checkpoint_store,
    )
    # Extra knobs only the generation stage uses.
    if stage == "generation":
        ctx_kwargs["dspy_optimizer"] = dspy_optimizer
        ctx_kwargs["dspy_metric"] = dspy_metric
        ctx_kwargs["mipro_auto"] = mipro_auto

    ctx = StageContext(**ctx_kwargs)
    print(f"[dim]Running[/dim] {stage} stage on workspace {ws.name!r}...")
    result = optimizer.optimize(ctx)

    # Mark the checkpoint completed so it doesn't show up as resumable.
    if checkpoint_obj is not None and checkpoint_store is not None:
        try:
            checkpoint_store.mark_completed(checkpoint_obj.checkpoint_id)
        except Exception as _exc:  # pragma: no cover - defensive
            print(f"[yellow]Checkpoint finalize skipped:[/yellow] {_exc}")

    # Build a bundle shape compatible with ``ragdx experiment-report``
    # so the same renderer works on workspace tune output.
    has_gt = eff_gt_mode == "with_gt"
    mode_label = "with_gt" if has_gt else "no_gt"
    bundle: dict[str, _Any] = {
        "schema_version": 1,
        "stage": stage,
        "gt_mode": mode_label,
        "base_config": rag_config.scrubbed_for_commit().model_dump(mode="json"),
        "best_params": result.best_params,
        "best_composite": result.best_composite,
        "objective_spec": result.objective_spec,
        "meta": {
            "workspace": ws.name,
            "model": rag_config.generator.model,
            "model_endpoint": rag_config.generator.api_base,
            "experiment_mode": stage,
            "modes_run": [mode_label],
            "has_gt": has_gt,
            "detected_gt_mode": mode_label,
            "source": f"workspace {ws.name} {verb}",
        },
    }
    if stage == "generation":
        bundle["dspy_a_b"] = {mode_label: result.extras}
    else:
        bundle["bayes_search"] = {mode_label: result.to_bayes_search_bundle()}

    # Baseline + optimized + comparison diagnoses (same shape that
    # ``experiments._diagnose_per_mode`` writes for full experiments).
    try:
        if stage == "generation":
            _b_scores = dict(result.extras.get("baseline_scores", {}) or {})
            _o_scores = dict(result.extras.get("optimized_scores", {}) or {})
        else:
            _b_scores = dict(result.trials[0].scores or {}) if result.trials else {}
            _matching = [t for t in result.trials if t.params == result.best_params]
            _o_scores = dict(_matching[0].scores or {}) if _matching else {}
        eng = RAGDiagnosisEngine()
        b_rep = (
            eng.diagnose(_synth_eval_result(_b_scores, mode=mode_label)).model_dump()
            if _b_scores else None
        )
        o_rep = (
            eng.diagnose(_synth_eval_result(_o_scores, mode=mode_label)).model_dump()
            if _o_scores and _o_scores != _b_scores else None
        )
        if b_rep or o_rep:
            entry: dict = {"baseline": b_rep, "optimized": o_rep}
            if b_rep and o_rep:
                entry["comparison"] = _compare_diagnoses(
                    b_rep, o_rep,
                    baseline_scores=_b_scores, optimized_scores=_o_scores,
                )
            bundle["diagnosis"] = {mode_label: entry}
    except Exception as exc:
        print(f"[yellow]Diagnosis skipped:[/yellow] {exc}")

    out_path = ws.path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"[green]Wrote[/green] {out_path}")

    # Also write the optimized config so the next workspace eval can
    # consume it without the user threading paths manually.
    opt_yaml = ws.path(f"rag_optimized.{verb.replace('-', '_')}.yaml")
    if result.best_config is not None:
        result.best_config.scrubbed_for_commit().to_yaml(opt_yaml)
        print(f"[green]Wrote optimized config[/green] {opt_yaml}")

    # Save the run to RunStore + workspace history.
    run_id: str | None = None
    try:
        if stage == "generation":
            _save_scores = dict(result.extras.get("optimized_scores", {}) or {})
        else:
            _matching = [t for t in result.trials if t.params == result.best_params]
            _save_scores = dict(_matching[0].scores or {}) if _matching else {}
        if _save_scores:
            eval_result = _scores_to_evaluation_result(
                _save_scores,
                metadata={
                    "source": f"workspace {ws.name} {verb}",
                    "stage": stage, "best_params": result.best_params,
                    "best_composite": result.best_composite,
                    "bundle_path": str(out_path.resolve()),
                },
            )
            rep, opt_plan = _diagnose_and_plan(eval_result)
            run = _store().save_run(
                eval_result, rep, opt_plan,
                name=f"{ws.name}/{output}",
                baseline_run_id=ws.current.get("baseline_run_id"),
                rag_config=(
                    result.best_config.scrubbed_for_commit()
                    if result.best_config is not None else None
                ),
            )
            run_id = run.run_id
    except Exception as exc:
        print(f"[yellow]Save-run skipped:[/yellow] {exc}")

    ws.log(verb, output=out_path, run_id=run_id, extra={"stage": stage})
    if run_id:
        print(f"[green]Saved run[/green] {run_id}")


# =====================================================================
# report
# =====================================================================
@_ws_app.command("report")
def report_cmd(
    name: str = typer.Argument(..., help="Workspace name."),
    source: str = typer.Option(
        "", "--source",
        help="Bundle JSON to render. Default: the most recent tune "
        "(prompt > rag) or eval artifact.",
    ),
    output: str = typer.Option(
        "report.html", "--output", "-o",
        help="HTML output filename, relative to the workspace.",
    ),
) -> None:
    """Render an HTML report from a workspace artifact."""
    import json

    from ragdx.ui.experiment_report import render_report

    ws = load_workspace(name)
    if source:
        src_path = ws.path(source)
    else:
        # Prefer prompt-tune > rag-tune > eval as the "most informative".
        for kind in ("tune-prompt", "tune-rag", "eval"):
            latest = ws.latest(kind)
            if latest is not None and latest.exists():
                src_path = latest
                break
        else:
            raise typer.BadParameter(
                f"Workspace {ws.name!r} has no artifacts to report. "
                f"Run an eval / tune first."
            )

    if not src_path.exists():
        raise typer.BadParameter(f"Source bundle not found: {src_path}")

    bundle = json.loads(src_path.read_text(encoding="utf-8"))
    html = render_report(bundle, title=f"ragdx workspace - {ws.name}")
    out_path = ws.path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    ws.log("report", output=out_path, extra={"source": src_path.name})
    print(f"[green]Wrote[/green] {out_path}")
    print(f"[dim]Open in a browser:[/dim] file:///{out_path.resolve().as_posix()}")


# =====================================================================
# compare (multi-workspace) -- Phase 4g
# =====================================================================
@_ws_app.command("compare")
def compare_cmd(
    names: list[str] = typer.Argument(
        ..., help="Two or more workspace names to compare.",
    ),
    output: str = typer.Option(
        "", "--output", "-o",
        help="Optional HTML output path. Without it, prints a plain "
        "table to stdout.",
    ),
) -> None:
    """Cross-experiment delta: align each workspace's latest scores +
    causal-graph posteriors side-by-side."""
    import json

    if len(names) < 2:
        raise typer.BadParameter("compare needs at least two workspace names.")

    snapshots: list[dict[str, Any]] = []
    for n in names:
        ws = load_workspace(n)
        snap: dict[str, Any] = {"name": n, "scores": {}, "posteriors": {}}
        eval_path = ws.latest("eval")
        if eval_path and eval_path.exists():
            try:
                d = json.loads(eval_path.read_text(encoding="utf-8"))
                snap["scores"] = {
                    **(d.get("retrieval") or {}),
                    **(d.get("generation") or {}),
                    **(d.get("e2e") or {}),
                }
            except json.JSONDecodeError:
                pass
        diag_path = ws.latest("diagnose")
        if diag_path and diag_path.exists():
            try:
                d = json.loads(diag_path.read_text(encoding="utf-8"))
                snap["posteriors"] = {
                    s.get("node", ""): s.get("posterior", 0.0)
                    for s in (d.get("causal_signals") or [])
                }
            except json.JSONDecodeError:
                pass
        snapshots.append(snap)

    # Print a rich table covering scores + top posteriors.
    metric_keys = sorted({k for s in snapshots for k in s["scores"]})
    posterior_keys = sorted({k for s in snapshots for k in s["posteriors"]})

    table = Table(title=f"Compare: {' vs '.join(names)}", show_lines=False)
    table.add_column("dimension", style="cyan")
    for n in names:
        table.add_column(n, justify="right")
    if metric_keys:
        table.add_row("[bold]Metrics[/bold]", *[""] * len(names))
        for m in metric_keys:
            row = [m]
            for s in snapshots:
                v = s["scores"].get(m)
                row.append(f"{v:.3f}" if isinstance(v, (int, float)) else "-")
            table.add_row(*row)
    if posterior_keys:
        table.add_row("[bold]Causal posteriors[/bold]", *[""] * len(names))
        for p in posterior_keys:
            row = [p]
            for s in snapshots:
                v = s["posteriors"].get(p)
                row.append(f"{v:.3f}" if isinstance(v, (int, float)) else "-")
            table.add_row(*row)
    print(table)

    if output:
        rows_html = []
        rows_html.append("<table><thead><tr><th>dimension</th>"
                         + "".join(f"<th>{n}</th>" for n in names)
                         + "</tr></thead><tbody>")
        for label, keys, picker in (
            ("Metric", metric_keys, lambda s, k: s["scores"].get(k)),
            ("Causal posterior", posterior_keys, lambda s, k: s["posteriors"].get(k)),
        ):
            for k in keys:
                vals = []
                for s in snapshots:
                    v = picker(s, k)
                    vals.append(f"{v:.3f}" if isinstance(v, (int, float)) else "-")
                cells = "".join(f"<td>{v}</td>" for v in vals)
                rows_html.append(f"<tr><td>{label}: {k}</td>{cells}</tr>")
        rows_html.append("</tbody></table>")
        from datetime import datetime
        html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>ragdx compare {' vs '.join(names)}</title>
<style>body{{font-family:sans-serif;margin:24px}}table{{border-collapse:collapse}}th,td{{border:1px solid #ddd;padding:6px 12px}}</style>
</head><body><h1>ragdx workspace compare</h1>
<p>Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
{''.join(rows_html)}</body></html>"""
        Path(output).write_text(html, encoding="utf-8")
        print(f"[green]Wrote[/green] {output}")


# Register subapp on the main app.
app.add_typer(_ws_app, name="workspace")


__all__ = [
    "compare_cmd",
    "diagnose_cmd",
    "eval_cmd",
    "init",
    "list_cmd",
    "report_cmd",
    "show",
    "tune_prompt_cmd",
    "tune_rag_cmd",
]
