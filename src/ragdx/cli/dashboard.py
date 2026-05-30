"""``ragdx dashboard`` -- the RunStore browser.

This is the legacy dashboard for the persisted-runs view; the
experiment-focused dashboard / report live in :mod:`ragdx.cli.experiment`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ragdx.cli._app import app


@app.command()
def dashboard():
    """Launch the Streamlit dashboard for the local RunStore."""
    # ``Path(__file__).parents[1]`` reaches ``ragdx/``; the dashboard
    # script lives at ``ragdx/ui/dashboard.py``.
    script = Path(__file__).resolve().parents[1] / "ui" / "dashboard.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(script)],
        check=False,
    )


__all__ = ["dashboard"]
