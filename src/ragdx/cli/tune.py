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
    """
    if stage not in _STAGE_CHOICES:
        raise typer.BadParameter(
            f"--stage must be one of {_STAGE_CHOICES}, got {stage!r}."
        )

    # Lazy imports keep `ragdx --help` light.
    import os

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
    bundle: dict[str, Any] = {
        "stage": stage,
        "gt_mode": mode_label,
        "base_config": rag_config.model_dump(mode="json"),
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
        result.best_config.to_yaml(opt_path)
        print(f"[green]Wrote optimized config[/green] {opt_path}")

    if result.best_composite is not None:
        print(f"[bold]Best composite:[/bold] {result.best_composite:.3f}")
    if result.best_params:
        print(f"[bold]Best params:[/bold] {result.best_params}")


__all__ = ["tune"]
