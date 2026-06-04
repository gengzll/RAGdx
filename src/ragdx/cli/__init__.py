"""Command-line interface for the ragdx library.

Subcommands cover the full diagnose → plan → optimize → save → inspect
workflow plus the end-to-end ``experiment`` pipeline. LLM-backed
features delegate to :mod:`ragdx.llm`, which selects a provider via
:class:`ragdx.config.LLMSettings`.

Run ``ragdx --help`` for a full command listing. Use ``ragdx show-config``
to print the effective configuration resolved from environment variables.

Module layout
-------------

The CLI used to live in a single 786-line ``cli.py``. PR2 split it
into one submodule per command group:

* :mod:`ragdx.cli.experiment` -- ``experiment``, ``experiment-dashboard``,
  ``experiment-report``
* :mod:`ragdx.cli.diagnose` -- ``diagnose``, ``plan``, ``explain-plan``
* :mod:`ragdx.cli.optimize` -- ``optimize``, ``monitor-session``,
  ``show-runner-templates``
* :mod:`ragdx.cli.runs` -- ``save``, ``runs``, ``sessions``, ``compare``,
  ``attach-feedback``, ``feedback-summary``, ``export-report``
* :mod:`ragdx.cli.dashboard` -- ``dashboard`` (legacy RunStore browser)
* :mod:`ragdx.cli.config` -- ``show-config``, ``normalize-tools``

The :data:`app` singleton lives in :mod:`ragdx.cli._app`; shared
helpers (``_store`` / ``_build_llm_callable`` / ``_load_eval`` /
``_build_engine`` / ``_diagnose_and_plan``) live in
:mod:`ragdx.cli._shared`. Command functions are re-exported at the
package level so existing callers (notably ``ragdx.cli.experiment``
imported by tests) keep working.
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
# Importing these modules causes their ``@app.command`` decorators to
# fire, registering each command with the :data:`app` singleton. The
# order doesn't matter for typer; we use alphabetical for readability.
from ragdx.cli import (  # noqa: E402, I001
    checkpoints as _checkpoints_mod,
    config as _config_mod,
    dashboard as _dashboard_mod,
    diagnose as _diagnose_mod,
    evaluate as _evaluate_mod,
    experiment as _experiment_mod,
    optimize as _optimize_mod,
    providers as _providers_mod,
    runs as _runs_mod,
    tune as _tune_mod,
)

# ---------------------------------------------------------------- backward-compat re-exports
# Existing callers (notably ``tests/test_experiments.py``) import command
# functions directly from ``ragdx.cli``. Re-export them at the package
# namespace so those imports keep working without changes.
diagnose = _diagnose_mod.diagnose
plan = _diagnose_mod.plan
explain_plan = _diagnose_mod.explain_plan

optimize = _optimize_mod.optimize
monitor_session = _optimize_mod.monitor_session
show_runner_templates = _optimize_mod.show_runner_templates

save = _runs_mod.save
runs = _runs_mod.runs
sessions = _runs_mod.sessions
attach_feedback = _runs_mod.attach_feedback
feedback_summary = _runs_mod.feedback_summary
export_report = _runs_mod.export_report
compare = _runs_mod.compare

experiment = _experiment_mod.experiment
experiment_dashboard = _experiment_mod.experiment_dashboard
experiment_report = _experiment_mod.experiment_report

dashboard = _dashboard_mod.dashboard

evaluate = _evaluate_mod.evaluate

tune = _tune_mod.tune

checkpoints = _checkpoints_mod.checkpoints
checkpoint_clean = _checkpoints_mod.checkpoint_clean

show_config = _config_mod.show_config
normalize_tools = _config_mod.normalize_tools


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
    "attach_feedback",
    "compare",
    "dashboard",
    "diagnose",
    "evaluate",
    "experiment",
    "experiment_dashboard",
    "experiment_report",
    "explain_plan",
    "export_report",
    "feedback_summary",
    "main",
    "monitor_session",
    "normalize_tools",
    "optimize",
    "plan",
    "runs",
    "save",
    "sessions",
    "show_config",
    "show_runner_templates",
    "tune",
]
