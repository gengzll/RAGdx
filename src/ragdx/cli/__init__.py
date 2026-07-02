"""Command-line interface for the ragdx library.

The CLI exposes the end-to-end ``experiment`` pipeline and the UI /
report wrappers around it. LLM-backed features delegate to
:mod:`ragdx.llm`, which selects a provider via
:class:`ragdx.config.LLMSettings`.

Run ``ragdx --help`` for the command listing:

* :mod:`ragdx.cli.experiment` -- ``experiment`` (headless run),
  ``experiment-dashboard`` / ``experiment-report`` (render a bundle),
  and ``ui`` (the upload → run → report Streamlit studio).

The :data:`app` singleton lives in :mod:`ragdx.cli._app`; the shared
``_store`` helper lives in :mod:`ragdx.cli._shared`.
"""

from __future__ import annotations

import os
import sys

import typer
from rich import print

from ragdx.cli._app import app
from ragdx.errors import LLMConfigError, RagdxError
from ragdx.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------- root flags
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
    project: str | None = typer.Option(
        None,
        "--project",
        help="Per-project namespace for RunStore / sessions / feedback / "
        "causal priors. When set, storage root becomes "
        "``.ragdx/projects/<name>/`` for this invocation, isolating one "
        "project's runs from another's. Overrides RAGDX_PROJECT. Use "
        "``ragdx show-config`` to see the resolved storage root. "
        "Ignored when RAGDX_ROOT is set explicitly (RAGDX_ROOT is the "
        "fully-qualified override).",
    ),
) -> None:
    """Global flags applied to every subcommand."""
    if log_level:
        os.environ["RAGDX_LOG_LEVEL"] = log_level.upper()
    if log_file:
        os.environ["RAGDX_LOG_FILE"] = log_file
    if project:
        # Setting the env var (rather than a module-level singleton) means
        # every helper that calls ``get_settings()`` -- including library
        # callers that ragdx delegates to (DSPy, ragas) -- sees the same
        # storage root, without us having to thread the value through
        # every API.
        os.environ["RAGDX_PROJECT"] = project
    configure_logging(force=True)


# ---------------------------------------------------------------- command submodules
# Importing this module causes its ``@app.command`` decorators to fire,
# registering each command with the :data:`app` singleton.
from ragdx.cli import experiment as _experiment_mod  # noqa: E402

# ---------------------------------------------------------------- re-exports
# ``tests/test_experiments.py`` imports the command functions directly
# from ``ragdx.cli``; keep them at the package namespace.
experiment = _experiment_mod.experiment
experiment_dashboard = _experiment_mod.experiment_dashboard
experiment_report = _experiment_mod.experiment_report
ui = _experiment_mod.ui


# ---------------------------------------------------------------- main entry
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


__all__ = [
    "app",
    "experiment",
    "experiment_dashboard",
    "experiment_report",
    "main",
    "ui",
]
