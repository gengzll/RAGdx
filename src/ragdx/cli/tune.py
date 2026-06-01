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
        ..., "--base-config", "-c",
        help="Path to the production RAGConfig YAML to optimize on top of.",
    ),
    questions: str = typer.Option(
        ..., "--questions", "-q",
        help="Path to a JSONL eval suite ({question, ground_truth?, "
        "contexts?} per line).",
    ),
    stage: str = typer.Option(
        "joint", "--stage", "-s",
        help=f"Which slice to optimize. One of: {', '.join(_STAGE_CHOICES)}. "
        "``chunking`` re-chunks per trial (slow), ``retrieval`` reuses "
        "the vstore (fast), ``generation`` runs DSPy MIPROv2 on the "
        "prompt, ``joint`` matches ``ragdx experiment``'s BO behaviour.",
    ),
    corpus: str = typer.Option(
        "", "--corpus",
        help="Optional override of the corpus path declared in the YAML. "
        "Falls back to ``base_config.corpus.path``.",
    ),
    budget: int = typer.Option(
        8, "--budget", "-b",
        help="BO trial budget for BO-driven stages (chunking / retrieval "
        "/ joint). Ignored for generation (MIPROv2 has its own budget).",
    ),
    bo_init: int = typer.Option(
        3, "--bo-init",
        help="Random initialisation rounds for BO.",
    ),
    seed: int = typer.Option(7, "--seed"),
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
    """
    if stage not in _STAGE_CHOICES:
        raise typer.BadParameter(
            f"--stage must be one of {_STAGE_CHOICES}, got {stage!r}."
        )
    if use_llm and use_both:
        raise typer.BadParameter("Use either --use-llm or --use-both, not both.")
    if (use_llm or use_both or use_llm_planner or baseline_run_id) and not save:
        raise typer.BadParameter(
            "--use-llm / --use-both / --use-llm-planner / --baseline-run-id "
            "only apply when --save is set."
        )

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

    base_path = Path(base_config)
    if not base_path.exists():
        raise typer.BadParameter(f"--base-config not found: {base_path}")
    rag_config = RAGConfig.from_yaml(base_path)

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
            "or the ZHIPU_API_KEY / OPENAI_API_KEY environment variable."
        )

    corpus_value = corpus or rag_config.corpus.path
    if not corpus_value:
        raise typer.BadParameter(
            "Corpus path required: pass --corpus or set corpus.path in the YAML."
        )

    q_path = Path(questions)
    if not q_path.exists():
        raise typer.BadParameter(f"--questions not found: {q_path}")
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
        n_bo_trials=budget,
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
    if stage == "generation":
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

    ctx = StageContext(
        base_config=rag_config,
        chunks_master=chunks_master,
        records=stage_records,
        objective=objective,
        metrics=metrics,
        runtime=runtime,
        n_bo_trials=budget,
        n_bo_init=bo_init,
        seed=seed,
        re_chunk_fn=re_chunk_fn,
        label=mode_label,
    )

    optimizer = {
        "chunking": ChunkingOptimizer,
        "retrieval": RetrievalOptimizer,
        "generation": GenerationOptimizer,
        "joint": JointOptimizer,
    }[stage]()

    print(
        f"[bold]Running[/bold] {stage} optimizer on "
        f"{len(chunks_master)} chunks x {len(records)} records "
        f"(GT mode: {mode_label}, budget: {budget})..."
    )
    result = optimizer.optimize(ctx)

    # ------------------------------------------------------------------
    # Serialize the result.
    # ------------------------------------------------------------------
    # Scrub credentials before serializing the base_config into the
    # bundle JSON. ``rag_config.generator.api_key`` was hydrated above
    # from --api-key or env, and the unscrubbed copy would leak into
    # the on-disk bundle (same bug class as --write-optimized-config,
    # fixed for that path in commit 73ee990; this is the JSON sibling).
    bundle: dict[str, Any] = {
        "stage": stage,
        "gt_mode": mode_label,
        "base_config": rag_config.scrubbed_for_commit().model_dump(mode="json"),
        "best_params": result.best_params,
        "best_composite": result.best_composite,
        "objective_spec": result.objective_spec,
    }
    if stage == "generation":
        bundle.update({"generation": result.extras})
    else:
        bundle.update({"bayes_search": result.to_bayes_search_bundle()})

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

        if stage == "generation":
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
            "tune_stage": stage,
            "gt_mode": mode_label,
            "best_params": result.best_params,
            "best_composite": result.best_composite,
            "base_config_path": str(base_path.resolve()),
            "questions_path": str(q_path.resolve()),
            "corpus": str(corpus_value),
            "bundle_path": str(out_path.resolve()),
        }
        if name:
            synth_metadata["name"] = name
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
        run_name = name or f"{rag_config.name or 'rag'}-tune-{stage}"
        run = _store().save_run(
            eval_result, report, opt_plan,
            name=run_name,
            baseline_run_id=baseline_run_id or None,
        )
        print(f"[green]Saved run:[/green] {run.run_id} [dim](name: {run_name})[/dim]")
        print(
            f"[dim]Next: `ragdx runs` / `ragdx dashboard` / "
            f"`ragdx export-report {run.run_id} report.md`.[/dim]"
        )


__all__ = ["tune"]
