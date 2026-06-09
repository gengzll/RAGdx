"""``ragdx evaluate`` -- run a RAGConfig over an eval suite.

Produces a normalized :class:`EvaluationResult` JSON that the rest of
ragdx's control plane (``diagnose`` / ``plan`` / ``compare`` /
``save``) already consumes. This is the bridge that closes the loop:
describe your production RAG in YAML, score it against your own eval
suite, then drive ragdx's optimization machinery against the result.

When ``--save`` is passed, ``evaluate`` also diagnoses + plans the
result and persists everything to the :class:`RunStore`. The saved
run shows up in ``ragdx runs`` / ``ragdx dashboard`` without the
caller having to chain ``ragdx save`` manually.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich import print

from ragdx.cli._app import app


@app.command("evaluate")
def evaluate(
    config: str = typer.Option(
        ..., "--config", "-c",
        help="Path to a RAGConfig YAML describing your production RAG.",
    ),
    questions: str = typer.Option(
        ..., "--questions", "-q",
        help="Path to a JSONL eval suite. One ``{question, ground_truth?, "
        "contexts?}`` record per line. ``ground_truth`` is optional -- "
        "its presence determines whether reference-based ragas metrics "
        "(``context_recall``, ``answer_correctness``) run.",
    ),
    corpus: str = typer.Option(
        "", "--corpus",
        help="Optional override of the corpus path declared in the YAML "
        "config. If empty, the YAML's ``corpus.path`` is used. Accepts "
        "the same forms as ``ragdx experiment``: HF dataset name, "
        ".pdf path, .jsonl corpus path, or a comma-separated list.",
    ),
    output: str = typer.Option(
        "evaluation_result.json", "--output", "-o",
        help="Where to write the resulting EvaluationResult JSON.",
    ),
    api_key: str = typer.Option(
        "", "--api-key",
        help="Override the YAML's generator.api_key. Falls back to "
        "ZHIPU_API_KEY / OPENAI_API_KEY env vars.",
    ),
    name: str = typer.Option(
        "", "--name",
        help="Optional run name. Stamped on result metadata and reused "
        "as the saved-run name when ``--save`` is set.",
    ),
    save: bool = typer.Option(
        False, "--save",
        help="Diagnose + plan the result and persist it to the RunStore "
        "(visible via ``ragdx runs`` / ``ragdx dashboard``). Without "
        "this flag, ``evaluate`` only writes the EvaluationResult JSON.",
    ),
    baseline_run_id: str = typer.Option(
        "", "--baseline-run-id",
        help="When ``--save`` is set, link this evaluation to an existing "
        "run as its baseline (used by ``ragdx compare`` / dashboard "
        "delta views).",
    ),
    use_llm: bool = typer.Option(
        False, "--use-llm",
        help="When ``--save`` is set, run LLM-only diagnosis instead of "
        "the rule-based default. Requires ZHIPU_API_KEY / OPENAI_API_KEY "
        "to be exported (the diagnosis LLM is built from package "
        "settings, not from ``--api-key``).",
    ),
    use_both: bool = typer.Option(
        False, "--use-both",
        help="When ``--save`` is set, run rule + LLM diagnosis and "
        "synthesize. Mutually exclusive with ``--use-llm``.",
    ),
    use_llm_planner: bool = typer.Option(
        False, "--use-llm-planner",
        help="When ``--save`` is set, refine the optimization plan with "
        "an LLM. Independent of ``--use-llm`` / ``--use-both``.",
    ),
    evaluator: str = typer.Option(
        "ragas", "--evaluator",
        help="Which evaluation library to use: ``ragas`` (default; the "
        "established RAG metric suite) or ``deepeval`` (Confident AI's "
        "test-case-style framework, includes G-Eval for less saturation "
        "on permissive judges). Both produce the same EvaluationResult "
        "schema downstream.",
    ),
):
    """Evaluate a RAGConfig against an eval suite.

    The output is a normalized EvaluationResult JSON ready for
    ``ragdx diagnose`` / ``ragdx plan`` / ``ragdx compare``.

    Example::

        # Just score + emit JSON:
        ragdx evaluate --config rag_config.yaml \\
            --questions my_eval.jsonl --output baseline.json
        ragdx diagnose baseline.json --use-llm

        # Score + diagnose + plan + persist to RunStore in one call:
        ragdx evaluate --config rag_config.yaml \\
            --questions my_eval.jsonl --output baseline.json \\
            --save --name "esg-baseline" --use-llm
        ragdx runs        # baseline now appears here
        ragdx dashboard   # and here
    """
    # Validate mutually-exclusive flags before paying for ragas.
    if use_llm and use_both:
        raise typer.BadParameter("Use either --use-llm or --use-both, not both.")
    if (use_llm or use_both or use_llm_planner or baseline_run_id) and not save:
        raise typer.BadParameter(
            "--use-llm / --use-both / --use-llm-planner / --baseline-run-id "
            "only apply when --save is set."
        )

    # Lazy imports so `ragdx --help` doesn't drag in dspy / langchain.
    import os

    from ragdx.cli._shared import _diagnose_and_plan, _store
    from ragdx.experiments import (
        ExperimentConfig,
        _load_corpus_and_records,
        _load_jsonl_questions,
        _normalize_corpus,
    )
    from ragdx.schemas.rag_config import RAGConfig
    from ragdx.workflows.evaluate import evaluate as workflow_evaluate

    cfg_path = Path(config)
    if not cfg_path.exists():
        raise typer.BadParameter(f"--config not found: {cfg_path}")
    rag_config = RAGConfig.from_yaml(cfg_path)

    # Resolve API key: explicit --api-key beats YAML beats env var.
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

    # Resolve corpus path.
    corpus_value = corpus or rag_config.corpus.path
    if not corpus_value:
        raise typer.BadParameter(
            "Corpus path required: pass --corpus or set corpus.path in the YAML."
        )

    # Load questions (JSONL).
    q_path = Path(questions)
    if not q_path.exists():
        raise typer.BadParameter(f"--questions not found: {q_path}")
    records = _load_jsonl_questions(q_path)
    if not records:
        raise typer.BadParameter(f"{q_path} contained zero records.")

    # Load corpus chunks. Re-use the experiments helper so PDF /
    # JSONL / HF dispatch is consistent with ``ragdx experiment``.
    # That helper takes an ``ExperimentConfig``; we synthesize a
    # minimal one purely for the corpus dispatch -- the LLM-side
    # fields are unused on this path.
    has_gt = any((r.ground_truth or "").strip() for r in records)
    stub_cfg = ExperimentConfig(
        corpus=_normalize_corpus(corpus_value)
        if "," in corpus_value
        else corpus_value,
        has_gt=has_gt,
        mode="auto",
        questions_path=str(q_path),  # so HF datasets aren't asked to synthesize
        api_key=rag_config.generator.api_key,
        api_base=rag_config.generator.api_base,
        model=rag_config.generator.model,
    )
    # Build the runtime so HF embeddings exist before _load_corpus_and_records
    # (it may need an llm_callable for question synthesis on no-GT JSONL,
    # but with questions_path set it won't synthesize).
    from ragdx.runtime.factories import build_runtime

    runtime = build_runtime(rag_config)
    chunks, _records_unused, source_meta = _load_corpus_and_records(stub_cfg, runtime)

    metadata: dict[str, Any] = {
        "config_path": str(cfg_path.resolve()),
        "questions_path": str(q_path.resolve()),
        "corpus": str(corpus_value),
        "corpus_source": source_meta,
    }
    if name:
        metadata["name"] = name

    if evaluator not in {"ragas", "deepeval"}:
        raise typer.BadParameter(
            f"--evaluator must be 'ragas' or 'deepeval', got {evaluator!r}."
        )

    result = workflow_evaluate(
        rag_config,
        chunks=chunks,
        records=records,
        runtime=runtime,
        metadata=metadata,
        evaluator=evaluator,
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"[green]Wrote[/green] {out_path}")
    print(
        "  Retrieval:",
        json.dumps({k: round(v, 4) for k, v in result.retrieval.items()}),
    )
    print(
        "  Generation:",
        json.dumps({k: round(v, 4) for k, v in result.generation.items()}),
    )
    print(
        "  E2E:",
        json.dumps({k: round(v, 4) for k, v in result.e2e.items()}),
    )

    if save:
        # Diagnose + plan + persist so the run is visible in `ragdx runs`
        # / `ragdx dashboard` without the caller having to invoke
        # `ragdx save` separately. This is the closed-loop path docs/12
        # describes: a single command from RAGConfig to dashboard.
        report, opt_plan = _diagnose_and_plan(
            result,
            use_llm=use_llm,
            use_both=use_both,
            use_llm_planner=use_llm_planner,
        )
        run = _store().save_run(
            result, report, opt_plan,
            name=name or None,
            baseline_run_id=baseline_run_id or None,
            # Persist the (scrubbed) RAGConfig so ``ragdx tune --from-run
            # <id>`` can inherit it. RunStore.save_run defensively scrubs
            # again, but we pass the scrubbed copy explicitly for clarity.
            rag_config=rag_config.scrubbed_for_commit(),
        )
        print(f"[green]Saved run:[/green] {run.run_id}")
        print(
            f"\n[dim]Next: `ragdx runs` / `ragdx dashboard` / "
            f"`ragdx export-report {run.run_id} report.md`.[/dim]"
        )
    else:
        print(
            "\n[dim]Next: `ragdx diagnose <out>` / `ragdx plan <out>` / "
            "`ragdx compare <out> <baseline>` "
            "(or re-run with --save to persist to the RunStore).[/dim]"
        )


__all__ = ["evaluate"]
