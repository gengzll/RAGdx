"""``ragdx checkpoints`` / ``ragdx checkpoint-clean``.

User-facing introspection over the checkpoint store (analogous to
``ragdx runs`` for SavedRuns).
"""

from __future__ import annotations

import typer
from rich import print
from rich.table import Table

from ragdx.checkpoint import CheckpointStore
from ragdx.cli._app import app


@app.command("checkpoints")
def checkpoints(
    all_: bool = typer.Option(
        False, "--all",
        help="Show completed checkpoints too (default: only running / interrupted).",
    ),
):
    """List the checkpoints in ``.ragdx/checkpoints/``.

    By default only shows checkpoints that are still resumable
    (``status="running"`` or ``"interrupted"``); pass ``--all`` to
    also see ``"completed"`` ones still on disk.

    The ``Resume`` column shows the command to continue:
    ``ragdx tune --resume <id>``. For completed checkpoints, you'll
    usually want to ``ragdx checkpoint-clean <id>`` instead.
    """
    store = CheckpointStore()
    rows = store.list()
    if not all_:
        rows = [c for c in rows if c.status in ("running", "interrupted")]
    if not rows:
        print(
            "[dim]No checkpoints to show. They'll appear here while a "
            "tune is running or after one is interrupted.[/dim]"
        )
        return
    table = Table(title="ragdx experiment checkpoints")
    table.add_column("Checkpoint ID")
    table.add_column("Kind")
    table.add_column("Status")
    table.add_column("Trials / Phase")
    table.add_column("Updated")
    table.add_column("Resume hint")
    for c in rows:
        progress = (
            f"{len(c.trials_completed)} / {c.max_trials or '?'}"
            if c.kind != "tune.generation"
            else f"phase={c.generation_phase or 'pre-a'}"
        )
        if c.status == "completed":
            hint = "[dim]done; ragdx checkpoint-clean[/dim]"
        else:
            hint = f"ragdx tune --resume {c.checkpoint_id}"
        table.add_row(
            c.checkpoint_id,
            c.kind,
            c.status,
            progress,
            c.updated_at[:19] if c.updated_at else "—",
            hint,
        )
    print(table)


@app.command("checkpoint-clean")
def checkpoint_clean(
    checkpoint_id: str = typer.Argument(
        "",
        help="Checkpoint id to delete. Empty + ``--completed`` deletes "
        "all completed checkpoints.",
    ),
    completed: bool = typer.Option(
        False, "--completed",
        help="Delete every completed checkpoint (the ones that don't "
        "need to be resumed anymore).",
    ),
):
    """Delete one or more checkpoint directories from disk.

    Either pass a specific ``<checkpoint_id>``, or use
    ``--completed`` to sweep every ``status=="completed"`` entry.
    """
    store = CheckpointStore()
    if not checkpoint_id and not completed:
        raise typer.BadParameter(
            "Pass a checkpoint id, or use --completed to sweep all done."
        )
    if checkpoint_id:
        if store.delete(checkpoint_id):
            print(f"[green]Deleted[/green] {checkpoint_id}")
        else:
            raise typer.BadParameter(
                f"No checkpoint dir at {store.dir_for(checkpoint_id)}"
            )
        return
    # --completed sweep.
    cleaned = []
    for c in store.list():
        if c.status == "completed":
            store.delete(c.checkpoint_id)
            cleaned.append(c.checkpoint_id)
    if cleaned:
        print(f"[green]Deleted[/green] {len(cleaned)} completed checkpoint(s):")
        for cid in cleaned:
            print(f"  - {cid}")
    else:
        print("[dim]No completed checkpoints to clean.[/dim]")


__all__ = ["checkpoint_clean", "checkpoints"]
