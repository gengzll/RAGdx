"""``ragdx tune`` -- stage-targeted RAG-config optimization.

Distinct from ``ragdx optimize`` (which takes a pre-computed
EvaluationResult and runs the executor / runner script flow). ``tune``
takes a base RAGConfig + an eval suite and asks one of the four
stage optimizers (chunking / retrieval / generation / joint) to
improve a single slice of it.

::

    ragdx tune --base-config rag_config.yaml \\
        --questions my_eval.jsonl --corpus docs/report.pdf \\
        --stage retrieval --budget 8 \\
        --output optimized_retrieval.json

The output bundle has the same shape as ``ragdx experiment``'s
``bayes_search[mode]`` section so the experiment dashboard renders it
(or the smaller report renderer if you prefer).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich import print

from ragdx.cli._app import app

_STAGE_CHOICES = ("chunking", "retrieval", "generation", "joint")


@app.command("tune")
def tune(
    base_config: str = typer.Option(
        "", "--base-config", "-c",
        help="Path to the production RAGConfig YAML to optimize on top of. "
        "Required unless ``--from-run`` is set (which inherits the YAML "
        "from the saved run's stored ``rag_config``). When both are set, "
        "``--base-config`` overrides the inherited config.",
    ),
    questions: str = typer.Option(
        "", "--questions", "-q",
        help="Path to a JSONL eval suite ({question, ground_truth?, "
        "contexts?} per line). Required unless ``--from-run`` is set "
        "and the saved run's metadata records a questions_path.",
    ),
    from_run: str = typer.Option(
        "", "--from-run",
        help="Inherit base config + plan + baseline link from a SavedRun. "
        "When set: ``--base-config`` / ``--questions`` / ``--corpus`` / "
        "``--stage`` / ``--budget`` / ``--baseline-run-id`` are pulled "
        "from the run by default (explicit flags still win). Requires "
        "the saved run was produced by ``ragdx evaluate --save`` on "
        "PR6+ (so SavedRun.rag_config is populated). When neither "
        "``--from-run`` nor ``--base-config`` is set, defaults to the "
        "most recent run in the RunStore -- the common 'tune the thing "
        "I just evaluated' case needs zero arguments.",
    ),
    experiment_name: str = typer.Option(
        "", "--experiment",
        help="When the inherited plan has multiple experiments, pick this "
        "one by name. Default: first experiment (plan experiments are "
        "pre-sorted by priority).",
    ),
    stage: str = typer.Option(
        "auto", "--stage", "-s",
        help=f"Which slice to optimize. One of: {', '.join(_STAGE_CHOICES)}, "
        "or 'auto'. 'auto' resolves from ``--from-run``'s planned "
        "experiment.stage; without ``--from-run`` it defaults to 'joint'. "
        "``chunking`` re-chunks per trial (slow), ``retrieval`` reuses "
        "the vstore (fast), ``generation`` runs DSPy MIPROv2 on the "
        "prompt, ``joint`` matches ``ragdx experiment``'s BO behaviour.",
    ),
    corpus: str = typer.Option(
        "", "--corpus",
        help="Override of the corpus path. Falls back to ``--from-run``'s "
        "saved metadata, then ``base_config.corpus.path``.",
    ),
    budget: int = typer.Option(
        0, "--budget", "-b",
        help="BO trial budget for BO-driven stages. 0 means: inherit from "
        "``--from-run``'s ``experiment.max_trials``, or default to 8 when "
        "``--from-run`` is unset. Ignored for generation.",
    ),
    bo_init: int = typer.Option(
        3, "--bo-init",
        help="Random initialisation rounds for BO.",
    ),
    seed: int = typer.Option(7, "--seed"),
    mipro_auto: str = typer.Option(
        "light", "--mipro-auto",
        help="DSPy MIPROv2 search budget for ``--stage generation``: "
        "``light`` (~3 candidates, ~5 min), ``medium`` (~10-20 "
        "candidates, ~30 min), ``heavy`` (~30+ candidates, ~90 min). "
        "Bigger budgets give the proposer more room to find a prompt "
        "that beats the seed. Ignored for non-generation stages.",
    ),
    dspy_metric: str = typer.Option(
        "auto", "--dspy-metric",
        help="Inner-loop metric for ``--stage generation``: "
        "``auto`` (default: ``token_f1`` with GT, ``embed_rubric`` without), "
        "``embed_rubric`` (2 embedding cosines + 1 multi-output LLM "
        "rubric; cheaper and more discriminative than ``ragas``), "
        "``geval`` (deepeval G-Eval + AnswerRelevancy + Faithfulness; "
        "G-Eval is a CoT LLM-as-judge that resists saturation on "
        "permissive judges like GLM-4-Flash; requires "
        "``pip install ragdx[deepeval]``), "
        "``ragas`` (legacy ragas composite — context_precision + "
        "faithfulness + answer_relevancy; kept for back-compat), "
        "``llm_judge`` (single-LLM faithfulness; cheap but saturates), "
        "``token_f1`` (token-F1 vs GT; requires GT-populated records).",
    ),
    dspy_optimizer: str = typer.Option(
        "mipro", "--dspy-optimizer",
        help="Which DSPy teleprompter to use for ``--stage generation``: "
        "``mipro`` (default; MIPROv2 — BO over instruction x demos, "
        "writes BOTH system_instruction AND few_shot_demos to the "
        "winning config), "
        "``copro`` (COPRO — iterative LLM-driven instruction rewrite; "
        "writes ONLY system_instruction; fastest), "
        "``bootstrap_fewshot`` (BootstrapFewShot — produces few-shot "
        "demos from the seed program; writes ONLY few_shot_demos; "
        "no instruction change), "
        "``gepa`` (GEPA — reflective text-based evolution, arXiv "
        "2507.19457; writes ONLY system_instruction; "
        "EXPERIMENTAL — DSPy 3.x ``gepa`` package required). "
        "Ignored for non-generation stages.",
    ),
    resume: str = typer.Option(
        "", "--resume",
        help="Resume a previously-interrupted tune. Pass a specific "
        "checkpoint id (e.g. ``ckpt_a1b2c3d4``), or use the bare flag "
        "(``--resume=auto`` / ``--resume=latest``) to pick the most "
        "recent interrupted checkpoint for the same stage. BO stages "
        "(retrieval / chunking / joint) resume per-trial; ``generation`` "
        "resumes per-phase (baseline / miprov2 / re_eval).",
    ),
    no_checkpoint: bool = typer.Option(
        False, "--no-checkpoint",
        help="Don't create a checkpoint for this run. Use in CI / tests "
        "where every invocation is a clean slate. Default is to "
        "checkpoint after every BO trial (BO stages) or every phase "
        "boundary (generation stage).",
    ),
    output: str = typer.Option(
        "tune_result.json", "--output", "-o",
        help="Where to write the stage result bundle.",
    ),
    write_optimized_config: str = typer.Option(
        "", "--write-optimized-config",
        help="If set, also write the winning RAGConfig as YAML to this "
        "path -- ready to commit alongside your production config.",
    ),
    api_key: str = typer.Option(
        "", "--api-key",
        help="Override the YAML's generator.api_key.",
    ),
    save: bool = typer.Option(
        False, "--save",
        help="Persist the tuned best-config evaluation to the RunStore "
        "as a SavedRun (visible via ``ragdx runs`` / ``ragdx "
        "dashboard``). The synthesized EvaluationResult is built from "
        "the best trial's ragas scores (for BO stages) or from the "
        "MIPROv2 optimized run (for the generation stage).",
    ),
    name: str = typer.Option(
        "", "--name",
        help="Saved-run name when ``--save`` is set. Falls back to "
        "``{base_config.name}-tune-{stage}`` if empty.",
    ),
    baseline_run_id: str = typer.Option(
        "", "--baseline-run-id",
        help="When ``--save`` is set, link this tuned run to an existing "
        "baseline run so ``ragdx compare`` / dashboard render the delta.",
    ),
    use_llm: bool = typer.Option(
        False, "--use-llm",
        help="When ``--save`` is set, LLM-only diagnosis. Requires "
        "ZHIPU_API_KEY / OPENAI_API_KEY env vars.",
    ),
    use_both: bool = typer.Option(
        False, "--use-both",
        help="When ``--save`` is set, rule + LLM diagnosis. Mutually "
        "exclusive with ``--use-llm``.",
    ),
    use_llm_planner: bool = typer.Option(
        False, "--use-llm-planner",
        help="When ``--save`` is set, refine the optimization plan with an LLM.",
    ),
):
    """Optimize a single stage of a production RAGConfig.

    Examples::

        # Sweep chunk_size + chunk_overlap on a PDF, hold retriever fixed
        ragdx tune --base-config rag.yaml --questions q.jsonl \\
            --corpus docs/report.pdf --stage chunking --budget 6

        # Try different top_k values (fast: vstore is built once)
        ragdx tune --base-config rag.yaml --questions q.jsonl \\
            --corpus docs/report.pdf --stage retrieval --budget 8

        # Tune the prompt at the existing retrieval config (DSPy MIPROv2)
        ragdx tune --base-config rag.yaml --questions q.jsonl \\
            --corpus docs/report.pdf --stage generation

        # Tune + persist to RunStore so the result shows up in the dashboard
        ragdx tune --base-config rag.yaml --questions q.jsonl \\
            --corpus docs/report.pdf --stage retrieval --budget 8 \\
            --save --name "retrieval-sweep-v3" \\
            --baseline-run-id <run_id_from_evaluate>

        # Closed loop: inherit base_config + questions + corpus + stage +
        # budget + baseline_run_id from a previously-saved evaluate run
        # (PR6+). Just point tune at the run id.
        ragdx tune --from-run <baseline_run_id> --save --name retrieval-sweep
    """
    if stage != "auto" and stage not in _STAGE_CHOICES:
        raise typer.BadParameter(
            f"--stage must be one of {_STAGE_CHOICES} or 'auto', got {stage!r}."
        )
    _MIPRO_AUTO_CHOICES = ("light", "medium", "heavy")
    if mipro_auto not in _MIPRO_AUTO_CHOICES:
        raise typer.BadParameter(
            f"--mipro-auto must be one of {_MIPRO_AUTO_CHOICES}, got {mipro_auto!r}."
        )
    _DSPY_METRIC_CHOICES = (
        "auto", "embed_rubric", "geval", "ragas", "llm_judge", "token_f1",
    )
    if dspy_metric not in _DSPY_METRIC_CHOICES:
        raise typer.BadParameter(
            f"--dspy-metric must be one of {_DSPY_METRIC_CHOICES}, "
            f"got {dspy_metric!r}."
        )
    _DSPY_OPTIMIZER_CHOICES = ("mipro", "copro", "bootstrap_fewshot", "gepa")
    if dspy_optimizer not in _DSPY_OPTIMIZER_CHOICES:
        raise typer.BadParameter(
            f"--dspy-optimizer must be one of {_DSPY_OPTIMIZER_CHOICES}, "
            f"got {dspy_optimizer!r}."
        )
    if use_llm and use_both:
        raise typer.BadParameter("Use either --use-llm or --use-both, not both.")
    if (use_llm or use_both or use_llm_planner or baseline_run_id) and not save:
        raise typer.BadParameter(
            "--use-llm / --use-both / --use-llm-planner / --baseline-run-id "
            "only apply when --save is set."
        )
    # NB: ``experiment_name`` requires --from-run, but ``from_run`` may
    # be defaulted from RunStore.latest() below. Defer the check until
    # after the default resolves.

    # Lazy imports keep `ragdx --help` light.
    import os

    from ragdx.cli._shared import _diagnose_and_plan, _store
    from ragdx.experiments import (
        ExperimentConfig,
        _build_ragas_metrics_for_mode,
        _load_corpus_and_records,
        _load_jsonl_questions,
        _make_pdf_re_chunk_fn,
        _normalize_corpus,
    )
    from ragdx.optim.objectives import default_objective
    from ragdx.optim.stages import (
        ChunkingOptimizer,
        GenerationOptimizer,
        JointOptimizer,
        RetrievalOptimizer,
        StageContext,
    )
    from ragdx.runtime.factories import build_runtime
    from ragdx.runtime.pipeline import RAGPipeline
    from ragdx.schemas.models import DatasetRecord
    from ragdx.schemas.rag_config import RAGConfig

    # ------------------------------------------------------------------
    # Default --from-run to RunStore.latest() when nothing is provided
    # so the most common path ("tune what I just evaluated") is a
    # zero-argument call. ``--base-config`` opts out of the default
    # (the user is bringing a new YAML, not iterating on a saved run).
    # ------------------------------------------------------------------
    if not from_run and not base_config:
        latest = _store().latest()
        if latest is not None:
            from_run = latest.run_id
            print(
                f"[dim]No --from-run / --base-config given; defaulting "
                f"to latest run [bold]{from_run}[/bold] "
                f"([cyan]{latest.name}[/cyan]).[/dim]"
            )

    # Deferred validation: --experiment is only meaningful with --from-run
    # (either explicitly passed or defaulted from RunStore.latest()).
    # If from_run is still empty here, no inheritance can happen.
    if experiment_name and not from_run:
        raise typer.BadParameter(
            "--experiment is only meaningful with --from-run "
            "(pass --from-run explicitly, or populate the RunStore "
            "with `ragdx evaluate --save` so the default kicks in)."
        )

    # ------------------------------------------------------------------
    # Inheritance from a SavedRun (--from-run).
    # ------------------------------------------------------------------
    inherited_experiment = None      # OptimizationExperiment | None
    inherited_rag_config = None      # RAGConfig | None
    inherited_questions = ""
    inherited_corpus = ""
    inherited_baseline_run_id = ""
    if from_run:
        saved = _store().load_run(from_run)
        if saved.rag_config is None:
            raise typer.BadParameter(
                f"Run {from_run!r} has no stored rag_config (likely a "
                "pre-PR6 SavedRun). Either pass --base-config explicitly, "
                "or re-run `ragdx evaluate --save` on the latest ragdx "
                "so the saved run carries its RAGConfig."
            )
        inherited_rag_config = saved.rag_config
        experiments = list(saved.optimization_plan.experiments)
        if not experiments:
            raise typer.BadParameter(
                f"Run {from_run!r}'s optimization_plan has zero "
                "experiments -- nothing for tune to inherit. Pass "
                "--stage / --budget explicitly, or re-diagnose to "
                "populate experiments."
            )
        if experiment_name:
            matching = [e for e in experiments if e.name == experiment_name]
            if not matching:
                available = ", ".join(e.name for e in experiments)
                raise typer.BadParameter(
                    f"No experiment named {experiment_name!r} in run "
                    f"{from_run!r}. Available: {available}"
                )
            inherited_experiment = matching[0]
        else:
            inherited_experiment = experiments[0]
        md = saved.evaluation.metadata or {}
        inherited_questions = md.get("questions_path") or ""
        inherited_corpus = md.get("corpus") or ""
        inherited_baseline_run_id = from_run
        print(
            f"[bold]Inheriting from run[/bold] {from_run}: "
            f"experiment=[cyan]{inherited_experiment.name}[/cyan], "
            f"stage=[cyan]{inherited_experiment.stage}[/cyan], "
            f"max_trials=[cyan]{inherited_experiment.max_trials}[/cyan], "
            f"search_space=[dim]{inherited_experiment.search_space}[/dim]"
        )

    # ------------------------------------------------------------------
    # Resolve effective values: explicit flag > inherited > hardcoded default.
    # ------------------------------------------------------------------
    eff_stage = stage if stage != "auto" else (
        inherited_experiment.stage if inherited_experiment else "joint"
    )
    # The plan model's OptimizerStage allows "corpus" / "orchestration"
    # which are not StageOptimizer choices yet. Map "corpus" → "chunking"
    # (closest stage we have); fail loudly on anything else unmapped.
    if eff_stage == "corpus":
        print(
            "[yellow]Note:[/yellow] plan stage 'corpus' mapped to 'chunking' "
            "(the closest available StageOptimizer)."
        )
        eff_stage = "chunking"
    if eff_stage not in _STAGE_CHOICES:
        raise typer.BadParameter(
            f"Resolved --stage={eff_stage!r} is not a StageOptimizer "
            f"choice. Available: {_STAGE_CHOICES}. Pass --stage "
            "explicitly to override."
        )
    eff_budget = budget if budget > 0 else (
        inherited_experiment.max_trials if inherited_experiment else 8
    )
    eff_baseline_run_id = baseline_run_id or inherited_baseline_run_id

    # ------------------------------------------------------------------
    # Resolve RAGConfig: explicit --base-config beats inherited.
    # ------------------------------------------------------------------
    if base_config:
        base_path = Path(base_config)
        if not base_path.exists():
            raise typer.BadParameter(f"--base-config not found: {base_path}")
        rag_config = RAGConfig.from_yaml(base_path)
        base_path_str = str(base_path.resolve())
    elif inherited_rag_config is not None:
        rag_config = inherited_rag_config.model_copy(deep=True)
        base_path_str = f"<inherited from run {from_run}>"
    else:
        raise typer.BadParameter(
            "--base-config is required when --from-run is not set."
        )

    if api_key:
        rag_config.generator.api_key = api_key
    elif not rag_config.generator.api_key:
        rag_config.generator.api_key = (
            os.environ.get("ZHIPU_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
    if not rag_config.generator.api_key:
        raise typer.BadParameter(
            "No api_key resolved. Set --api-key, generator.api_key in YAML, "
            "or the ZHIPU_API_KEY / OPENAI_API_KEY environment variable. "
            "(SavedRun.rag_config has api_key scrubbed by design.)"
        )

    corpus_value = corpus or inherited_corpus or rag_config.corpus.path
    if not corpus_value:
        raise typer.BadParameter(
            "Corpus path required: pass --corpus, set corpus.path in the "
            "YAML, or use --from-run pointing to a run whose metadata "
            "recorded the corpus path."
        )

    eff_questions = questions or inherited_questions
    if not eff_questions:
        raise typer.BadParameter(
            "--questions is required when --from-run is not set or the "
            "saved run's metadata doesn't include a questions_path."
        )
    q_path = Path(eff_questions)
    if not q_path.exists():
        raise typer.BadParameter(f"questions file not found: {q_path}")
    records = _load_jsonl_questions(q_path)
    if not records:
        raise typer.BadParameter(f"{q_path} contained zero records.")

    # GT mode is determined by the loaded records, NOT by the user.
    # An optimizer doesn't care; metric selection does.
    has_gt = any((r.ground_truth or "").strip() for r in records)
    mode_label = "with_gt" if has_gt else "no_gt"

    # Stub ExperimentConfig solely to reach _load_corpus_and_records.
    stub_cfg = ExperimentConfig(
        corpus=_normalize_corpus(corpus_value) if "," in corpus_value else corpus_value,
        has_gt=has_gt,
        mode="auto",
        questions_path=str(q_path),
        api_key=rag_config.generator.api_key,
        api_base=rag_config.generator.api_base,
        model=rag_config.generator.model,
        n_bo_trials=eff_budget,
        n_bo_init=bo_init,
        seed=seed,
        llm_max_concurrent=rag_config.judge.llm_max_concurrent,
        llm_max_retries=rag_config.judge.llm_max_retries,
        system_instruction=rag_config.generator.system_instruction,
    )
    runtime = build_runtime(rag_config)
    chunks_master, _records_from_corpus, _source = _load_corpus_and_records(
        stub_cfg, runtime,
    )

    objective = default_objective(mode_label)
    metrics = _build_ragas_metrics_for_mode(mode_label)
    re_chunk_fn = _make_pdf_re_chunk_fn(stub_cfg, chunks_master)

    # ------------------------------------------------------------------
    # Dispatch to the requested StageOptimizer.
    # ------------------------------------------------------------------
    stage_records: list[DatasetRecord]
    if eff_stage == "generation":
        # Generation needs records pre-retrieved at the base config's
        # retriever -- otherwise MIPROv2 has nothing to chew on.
        pipeline = RAGPipeline.build(
            rag_config, chunks_master,
            embedder=runtime.embeddings, llm_callable=runtime.llm_callable,
        )
        stage_records = [
            DatasetRecord(
                question=r.question,
                ground_truth=r.ground_truth,
                contexts=pipeline.retrieve(r.question),
            )
            for r in records
        ]
    else:
        stage_records = list(records)

    # ------------------------------------------------------------------
    # Checkpoint: load (--resume) or create. Skipped by --no-checkpoint.
    # The stage optimizer auto-saves after every BO trial / generation
    # phase, so a crash mid-run loses at most one unit of work.
    # ------------------------------------------------------------------
    checkpoint_obj = None
    checkpoint_store = None
    if not no_checkpoint:
        from ragdx.checkpoint import Checkpoint, CheckpointStore
        checkpoint_store = CheckpointStore()
        ckpt_kind = f"tune.{eff_stage}"
        if resume:
            target = resume
            if target.lower() in {"auto", "latest", "true"}:
                incomplete = [
                    c for c in checkpoint_store.list_incomplete()
                    if c.kind == ckpt_kind
                ]
                if not incomplete:
                    raise typer.BadParameter(
                        f"No interrupted {ckpt_kind} checkpoint to resume. "
                        "Run `ragdx checkpoints` to see what's available."
                    )
                target = incomplete[0].checkpoint_id
            try:
                checkpoint_obj = checkpoint_store.load(target)
            except FileNotFoundError as exc:
                raise typer.BadParameter(str(exc)) from exc
            # Re-mark as running -- ctrl-C'd checkpoints come back as
            # "interrupted" but we're now actively continuing them.
            checkpoint_obj.status = "running"
            checkpoint_obj.interrupted_reason = ""
            print(
                f"[bold]Resuming checkpoint[/bold] [cyan]{checkpoint_obj.checkpoint_id}[/cyan] "
                f"(kind=[cyan]{checkpoint_obj.kind}[/cyan], "
                f"trials_done=[cyan]{len(checkpoint_obj.trials_completed)}[/cyan], "
                f"phase=[cyan]{checkpoint_obj.generation_phase or '—'}[/cyan])"
            )
        else:
            checkpoint_obj = Checkpoint(
                kind=ckpt_kind,
                cli_args={
                    "stage": eff_stage,
                    "budget": eff_budget,
                    "bo_init": bo_init,
                    "seed": seed,
                    "mipro_auto": mipro_auto,
                    "dspy_metric": dspy_metric,
                    "from_run": from_run,
                    "experiment_name": experiment_name,
                },
                name=name,
                rag_config_yaml=rag_config.scrubbed_for_commit().to_yaml_string(),
            )
            checkpoint_store.save(checkpoint_obj)
            print(
                f"[dim]Checkpoint:[/dim] [cyan]{checkpoint_obj.checkpoint_id}[/cyan] "
                f"[dim](resumable via `ragdx tune --resume "
                f"{checkpoint_obj.checkpoint_id}`)[/dim]"
            )

    ctx_kwargs: dict[str, Any] = dict(
        base_config=rag_config,
        chunks_master=chunks_master,
        records=stage_records,
        objective=objective,
        metrics=metrics,
        runtime=runtime,
        n_bo_trials=eff_budget,
        n_bo_init=bo_init,
        seed=seed,
        re_chunk_fn=re_chunk_fn,
        label=mode_label,
        mipro_auto=mipro_auto,
        dspy_metric=dspy_metric,
        dspy_optimizer=dspy_optimizer,
        checkpoint=checkpoint_obj,
        checkpoint_store=checkpoint_store,
    )
    # When --from-run inherited a planned experiment, thread its
    # search_space into the BO axes the StageContext exposes. The plan
    # encodes the user-meaningful sweep ranges; defaults (256/512/1024
    # for chunk_size, 1/3/5/7 for top_k) are only used when the plan
    # didn't specify -- or when --from-run isn't used at all.
    if inherited_experiment is not None:
        space = inherited_experiment.search_space or {}
        if space.get("top_k"):
            ctx_kwargs["top_ks"] = [int(v) for v in space["top_k"]]
        if space.get("chunk_size"):
            ctx_kwargs["chunk_sizes"] = [int(v) for v in space["chunk_size"]]
        if space.get("chunk_overlap"):
            ctx_kwargs["chunk_overlaps"] = [int(v) for v in space["chunk_overlap"]]
    ctx = StageContext(**ctx_kwargs)

    optimizer = {
        "chunking": ChunkingOptimizer,
        "retrieval": RetrievalOptimizer,
        "generation": GenerationOptimizer,
        "joint": JointOptimizer,
    }[eff_stage]()

    print(
        f"[bold]Running[/bold] {eff_stage} optimizer on "
        f"{len(chunks_master)} chunks x {len(records)} records "
        f"(GT mode: {mode_label}, budget: {eff_budget})..."
    )
    try:
        result = optimizer.optimize(ctx)
    except KeyboardInterrupt:
        if checkpoint_obj is not None and checkpoint_store is not None:
            checkpoint_store.mark_interrupted(
                checkpoint_obj.checkpoint_id, reason="keyboard_interrupt",
            )
            print(
                f"\n[yellow]Interrupted.[/yellow] Resume with "
                f"`ragdx tune --resume {checkpoint_obj.checkpoint_id}` "
                "(any flag overrides still apply)."
            )
        raise
    except Exception as exc:
        if checkpoint_obj is not None and checkpoint_store is not None:
            checkpoint_store.mark_interrupted(
                checkpoint_obj.checkpoint_id,
                reason=f"unhandled_exception:{type(exc).__name__}",
            )
            print(
                f"\n[red]Crashed:[/red] {type(exc).__name__}: {exc}\n"
                f"[yellow]Resume with[/yellow] "
                f"`ragdx tune --resume {checkpoint_obj.checkpoint_id}` "
                "once the underlying issue is fixed."
            )
        raise
    if checkpoint_obj is not None and checkpoint_store is not None:
        checkpoint_store.mark_completed(checkpoint_obj.checkpoint_id)

    # ------------------------------------------------------------------
    # Serialize the result.
    # ------------------------------------------------------------------
    # Scrub credentials before serializing the base_config into the
    # bundle JSON. ``rag_config.generator.api_key`` was hydrated above
    # from --api-key or env, and the unscrubbed copy would leak into
    # the on-disk bundle (same bug class as --write-optimized-config,
    # fixed for that path in commit 73ee990; this is the JSON sibling).
    # Tune bundles are shape-compatible with ``ragdx experiment`` bundles
    # so the existing ``experiment-dashboard`` (Streamlit) and
    # ``experiment-report`` (HTML) renderers work on them without
    # special-casing. The renderer iterates ``bayes_search`` / ``dspy_a_b``
    # as ``{gt_mode: payload}`` so we wrap by ``mode_label`` (a single-key
    # dict for tune, vs. two-key for full experiment runs).
    bundle: dict[str, Any] = {
        "schema_version": 1,
        "stage": eff_stage,
        "gt_mode": mode_label,
        "base_config": rag_config.scrubbed_for_commit().model_dump(mode="json"),
        "best_params": result.best_params,
        "best_composite": result.best_composite,
        "objective_spec": result.objective_spec,
        # Minimal ``meta`` block so experiment-report's header has
        # something to render (model, source, mode list).
        "meta": {
            "model": rag_config.generator.model,
            "model_endpoint": rag_config.generator.api_base,
            "experiment_mode": eff_stage,
            "modes_run": [mode_label],
            "has_gt": has_gt,
            "detected_gt_mode": mode_label,
            "source": "ragdx tune",
        },
    }
    if from_run:
        bundle["inherited"] = {
            "from_run": from_run,
            "experiment_name": inherited_experiment.name,
            "experiment_stage": inherited_experiment.stage,
            "experiment_max_trials": inherited_experiment.max_trials,
            "experiment_search_space": inherited_experiment.search_space,
        }
        bundle["meta"]["source"] = f"ragdx tune --from-run {from_run}"
    if eff_stage == "generation":
        # Experiment bundles call this section ``dspy_a_b`` (baseline vs.
        # MIPROv2-optimized A/B comparison). Use the same name + mode
        # wrapping so experiment-report's DSPy section renders.
        bundle["dspy_a_b"] = {mode_label: result.extras}
    else:
        bundle["bayes_search"] = {mode_label: result.to_bayes_search_bundle()}

    # Rule-based diagnosis -- baked into the bundle so the static HTML
    # report tells the user (1) why this tune stage was worth running
    # given the baseline's bottlenecks and (2) what's still broken at
    # the tuned config. Same shape as the ``ragdx experiment`` bundle
    # (see :func:`ragdx.experiments._diagnose_per_mode`): a per-mode
    # dict with ``{"baseline": <report>, "optimized": <report>}``.
    # Rule-based only, no LLM calls. ``ragdx diagnose --use-llm``
    # on the saved run still gives an LLM-refined view.
    try:
        from ragdx.core.diagnosis import RAGDiagnosisEngine
        from ragdx.experiments import (  # type: ignore[attr-defined]
            _compare_diagnoses,
            _synth_eval_result,
        )

        if eff_stage == "generation":
            _baseline_diag_scores = dict(
                result.extras.get("baseline_scores", {}) or {}
            )
            _optimized_diag_scores = dict(
                result.extras.get("optimized_scores", {}) or {}
            )
        else:
            # BO-style stage: the first trial approximates the baseline,
            # the best-trial scores are the "optimized" outcome.
            _baseline_diag_scores = (
                dict(result.trials[0].scores or {}) if result.trials else {}
            )
            _matching = [
                t for t in result.trials if t.params == result.best_params
            ]
            _optimized_diag_scores = (
                dict(_matching[0].scores or {}) if _matching else {}
            )

        _engine = RAGDiagnosisEngine()

        def _diag(scores: dict[str, Any], phase: str) -> dict | None:
            if not scores:
                return None
            try:
                rep = _engine.diagnose(_synth_eval_result(
                    scores, mode=mode_label,
                    extra_metadata={"phase": phase},
                ))
                return rep.model_dump()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[yellow]{phase} diagnosis skipped:[/yellow] {exc}")
                return None

        _baseline_rep = _diag(_baseline_diag_scores, "baseline")
        _optimized_rep = (
            _diag(_optimized_diag_scores, "optimized")
            if _optimized_diag_scores != _baseline_diag_scores
            else None
        )
        if _baseline_rep or _optimized_rep:
            _entry: dict[str, Any] = {
                "baseline": _baseline_rep,
                "optimized": _optimized_rep,
            }
            if _baseline_rep and _optimized_rep:
                _entry["comparison"] = _compare_diagnoses(
                    _baseline_rep, _optimized_rep,
                    baseline_scores=_baseline_diag_scores,
                    optimized_scores=_optimized_diag_scores,
                )
            bundle["diagnosis"] = {mode_label: _entry}
    except Exception as _exc:  # pragma: no cover - defensive
        print(f"[yellow]Diagnosis skipped:[/yellow] {_exc}")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[green]Wrote[/green] {out_path}")

    if write_optimized_config and result.best_config is not None:
        opt_path = Path(write_optimized_config)
        opt_path.parent.mkdir(parents=True, exist_ok=True)
        # Scrub credentials before writing -- otherwise the env-resolved
        # api_key (which we set on ``rag_config.generator.api_key`` above)
        # would persist into the YAML, leaking the key if the user
        # commits the file. Discovered in the PR4 e2e test (2026-05-30).
        result.best_config.scrubbed_for_commit().to_yaml(opt_path)
        print(
            f"[green]Wrote optimized config[/green] {opt_path} "
            "[dim](credentials scrubbed)[/dim]"
        )

    if result.best_composite is not None:
        print(f"[bold]Best composite:[/bold] {result.best_composite:.3f}")
    if result.best_params:
        print(f"[bold]Best params:[/bold] {result.best_params}")

    if save:
        # Synthesize an EvaluationResult from the best trial's ragas
        # scores so the tuned run plugs into the same RunStore /
        # diagnose / dashboard path that `ragdx save` and
        # `ragdx evaluate --save` use.
        from ragdx.workflows.evaluate import _scores_to_evaluation_result

        if eff_stage == "generation":
            # Generation has no BO trials; the optimized program's ragas
            # scores live in extras.
            best_scores = dict(result.extras.get("optimized_scores", {}) or {})
        else:
            # Find the trial whose params match best_params. Fall back to
            # the highest-composite feasible trial if a match isn't found
            # (defensive: a custom StageOptimizer could in theory drift).
            matching = [
                t for t in result.trials if t.params == result.best_params
            ]
            if matching:
                best_scores = dict(matching[0].scores or {})
            else:
                feasible = [t for t in result.trials if t.feasible]
                pool = feasible or list(result.trials)
                best_trial = max(
                    pool,
                    key=lambda t: t.composite_score
                    if t.composite_score is not None
                    else float("-inf"),
                    default=None,
                )
                best_scores = dict(best_trial.scores or {}) if best_trial else {}

        synth_metadata: dict[str, Any] = {
            "source": "ragdx tune",
            "tune_stage": eff_stage,
            "gt_mode": mode_label,
            "best_params": result.best_params,
            "best_composite": result.best_composite,
            "base_config_path": base_path_str,
            "questions_path": str(q_path.resolve()),
            "corpus": str(corpus_value),
            "bundle_path": str(out_path.resolve()),
        }
        if name:
            synth_metadata["name"] = name
        if from_run:
            synth_metadata["from_run"] = from_run
            synth_metadata["inherited_experiment"] = inherited_experiment.name
        if write_optimized_config:
            synth_metadata["optimized_config_path"] = str(
                Path(write_optimized_config).resolve()
            )

        eval_result = _scores_to_evaluation_result(
            best_scores, metadata=synth_metadata,
        )
        report, opt_plan = _diagnose_and_plan(
            eval_result,
            use_llm=use_llm,
            use_both=use_both,
            use_llm_planner=use_llm_planner,
        )
        run_name = name or f"{rag_config.name or 'rag'}-tune-{eff_stage}"
        run = _store().save_run(
            eval_result, report, opt_plan,
            name=run_name,
            baseline_run_id=eff_baseline_run_id or None,
            # Persist the tuned (scrubbed) config so a future
            # `tune --from-run` can chain off this result too.
            rag_config=rag_config.scrubbed_for_commit(),
        )
        print(f"[green]Saved run:[/green] {run.run_id} [dim](name: {run_name})[/dim]")
        print(
            f"[dim]Next: `ragdx runs` / `ragdx dashboard` / "
            f"`ragdx export-report {run.run_id} report.md`.[/dim]"
        )


__all__ = ["tune"]
