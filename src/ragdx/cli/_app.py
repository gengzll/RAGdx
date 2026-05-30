"""The typer ``app`` singleton imported by every cli submodule.

Lives in its own module to avoid the circular-import dance that would
otherwise be needed (``cli/__init__.py`` would have to define ``app``
before importing submodules, and submodules would need ``app`` before
their ``@app.command`` decorators run).
"""

from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False,
    help="ragdx — RAG diagnosis & optimization toolkit.",
)

__all__ = ["app"]
