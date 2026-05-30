"""``ragdx evaluate`` -- run a RAGConfig over an eval suite.

Produces a normalized :class:`EvaluationResult` JSON that the rest of
ragdx's control plane (``diagnose`` / ``plan`` / ``compare`` /
``save``) already consumes. This is the bridge that closes the loop:
describe your production RAG in YAML, score it against your own eval
suite, then drive ragdx's optimization machinery against the result.
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
        help="Optional run name to stamp on the result's metadata "
        "(useful when comparing multiple configs).",
    ),
):
    """Evaluate a RAGConfig against an eval suite.

    The output is a normalized EvaluationResult JSON ready for
    ``ragdx diagnose`` / ``ragdx plan`` / ``ragdx compare``.

    Example::

        ragdx evaluate --config rag_config.yaml \\
            --questions my_eval.jsonl --output baseline.json
        ragdx diagnose baseline.json --use-llm
    """
    # Lazy imports so `ragdx --help` doesn't drag in dspy / langchain.
    import os

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

    result = workflow_evaluate(
        rag_config,
        chunks=chunks,
        records=records,
        runtime=runtime,
        metadata=metadata,
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
    print(
        "\n[dim]Next: `ragdx diagnose <out>` / `ragdx plan <out>` / "
        "`ragdx compare <out> <baseline>`.[/dim]"
    )


__all__ = ["evaluate"]
