"""``ragdx show-config`` / ``normalize-tools`` -- environment-facing utilities.

* ``show-config`` -- inspect the resolved ``ragdx.config.LLMSettings``
  so users can debug provider/model/api-key resolution.
* ``normalize-tools`` -- bring external evaluator outputs (ragas /
  RAGChecker JSON dumps) into the unified ``EvaluationResult`` schema
  so the rest of the pipeline can consume them.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print

from ragdx.cli._app import app
from ragdx.config import get_settings
from ragdx.core.evaluator import UnifiedEvaluator


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


__all__ = ["normalize_tools", "show_config"]
