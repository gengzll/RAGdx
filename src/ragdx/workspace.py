"""Per-experiment workspace storage.

A *workspace* is a single self-contained folder for one experiment:
* ``workspace.yaml`` -- the config + history manifest.
* ``rag_config.yaml`` -- the RAGConfig under test.
* ``questions.jsonl`` -- the eval suite.
* Every produced artifact (``*.eval.json``, ``*.diagnose.json``,
  ``tune_*.json``, ``report.html``) lands here.

Background: the pre-workspace UX had each command take an ``--output``
flag and store runs in a global ``.ragdx/runs/<id>/`` namespace, so
artifacts were scattered across the filesystem and tied together only
by RunStore IDs. Workspaces flip the model: pick a *name* and all the
commands target the same folder. The CLI verbs become short
("``ragdx workspace eval my-exp``") because the workspace already knows
the config / questions / corpus.

Public surface used by CLI:
* :class:`Workspace` -- read + mutate a workspace.
* :func:`init_workspace` -- create a new one (mkdir + write manifest).
* :func:`load_workspace` -- open an existing one by name.
* :func:`list_workspaces` -- enumerate names under the workspaces root.
* :func:`workspaces_root` -- resolve the workspaces directory.

The legacy ``ragdx evaluate / diagnose / tune`` commands keep working
without any workspace -- workspaces are an additive convenience.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ragdx.errors import RagdxError

# Workspace manifest schema version; bump when the on-disk shape
# changes in a breaking way so loaders can pick the right migration.
WORKSPACE_SCHEMA_VERSION = 1


def workspaces_root() -> Path:
    """Resolve the directory that holds workspace folders.

    Priority:
    1. ``RAGDX_WORKSPACES`` env var (absolute path).
    2. ``./workspaces/`` relative to the current working directory.

    Caller is responsible for creating it if it doesn't exist.
    """
    override = os.environ.get("RAGDX_WORKSPACES")
    if override:
        return Path(override).expanduser().resolve()
    return Path.cwd() / "workspaces"


class WorkspaceError(RagdxError):
    """Raised on workspace-level errors (missing manifest, bad shape).

    Inherits from :class:`RagdxError` so the CLI's error wrapper turns
    these into clean ``exit(1)`` instead of tracebacks.
    """


@dataclass
class HistoryEntry:
    """One row in the workspace's command history.

    Stored under ``workspace.yaml::history`` so the user can scroll
    back through "what did I run, in what order, with what output".
    """

    ts: str  # ISO-8601 UTC timestamp
    command: str  # e.g. "eval", "diagnose", "tune-rag"
    output: str | None = None  # relative path to the artifact this produced
    run_id: str | None = None  # RunStore id, when --save was used
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Workspace:
    """A single experiment workspace.

    Lifecycle:
    * Created by :func:`init_workspace` -- writes ``workspace.yaml``.
    * Loaded by :func:`load_workspace` -- reads the manifest, gives
      callers a mutable view + ``save()`` / ``log()`` methods.
    """

    name: str
    root: Path  # absolute path to workspaces/<name>/
    rag_config: str = "rag_config.yaml"  # relative to root
    questions: str = "questions.jsonl"  # relative to root
    corpus: str | None = None  # relative to root, or absolute, or None
    evaluator: str = "ragas"  # "ragas" | "deepeval"
    mode: str = "auto"  # "auto" | "with_gt" | "no_gt"
    created_at: str = ""
    history: list[HistoryEntry] = field(default_factory=list)
    # Pointers to the most-recent artifact of each kind, so subsequent
    # commands can resolve "what's my baseline" without re-traversing
    # history. Updated by ``log()`` based on the command verb.
    current: dict[str, str | None] = field(default_factory=dict)
    # Optional free-form notes the user can edit in workspace.yaml.
    notes: str = ""

    # --------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.root / "workspace.yaml"

    def path(self, rel: str | Path) -> Path:
        """Resolve a path that's relative to the workspace root.

        Absolute paths pass through unchanged. Use this when a caller
        gives you ``"baseline.eval.json"`` and you need
        ``workspaces/my-exp/baseline.eval.json``.
        """
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def rag_config_path(self) -> Path:
        return self.path(self.rag_config)

    def questions_path(self) -> Path:
        return self.path(self.questions)

    def corpus_path(self) -> Path | None:
        """Resolve ``self.corpus``, or fall back to the YAML config's
        ``corpus.path`` field. ``None`` if neither is set.

        Relative paths are resolved against the workspace root first,
        then against the current working directory (so a corpus file
        sitting at the repo root still resolves even though the YAML's
        ``corpus.path`` is a bare filename). HuggingFace dataset names
        ("org/dataset") are returned as-is.
        """
        candidate: str | None = self.corpus
        if candidate is None:
            # Peek at the YAML config (cheap -- just yaml.safe_load).
            cfg_path = self.rag_config_path()
            if cfg_path.exists():
                try:
                    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        candidate = (data.get("corpus") or {}).get("path")
                except yaml.YAMLError:
                    return None
        if not candidate:
            return None

        candidate_path = Path(candidate)
        if candidate_path.is_absolute():
            return candidate_path

        # HF datasets look like "org/name" but don't exist on disk;
        # for those just return the workspace-rooted path so the
        # downstream loader sees the unchanged string.
        ws_rel = self.root / candidate
        if ws_rel.exists():
            return ws_rel

        cwd_rel = Path.cwd() / candidate
        if cwd_rel.exists():
            return cwd_rel.resolve()

        # Neither location holds the file -- fall back to the
        # workspace-rooted form so the loader's error message points
        # the user at the obvious place to drop it.
        return ws_rel

    # --------------------------------------------------------------
    # Manifest IO
    # --------------------------------------------------------------
    def save(self) -> None:
        """Write the manifest back to ``workspace.yaml``."""
        payload: dict[str, Any] = {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "name": self.name,
            "created_at": self.created_at,
            "rag_config": self.rag_config,
            "questions": self.questions,
            "corpus": self.corpus,
            "evaluator": self.evaluator,
            "mode": self.mode,
            "current": dict(self.current),
            "notes": self.notes,
            "history": [
                {
                    "ts": h.ts,
                    "command": h.command,
                    "output": h.output,
                    "run_id": h.run_id,
                    **({"extra": h.extra} if h.extra else {}),
                }
                for h in self.history
            ],
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def log(
        self,
        command: str,
        *,
        output: str | Path | None = None,
        run_id: str | None = None,
        extra: dict[str, Any] | None = None,
        update_current: bool = True,
    ) -> HistoryEntry:
        """Append a history entry + save the manifest.

        Returns the entry so callers can chain (e.g. ``ws.log(...).ts``).
        When ``update_current=True`` (default), the workspace's
        ``current`` pointer for that command kind is refreshed to point
        at the new output. The mapping mirrors the CLI verbs:

        * ``eval``         → ``current["latest_eval"]``
        * ``diagnose``     → ``current["latest_diagnose"]``
        * ``tune-rag``     → ``current["latest_tune_rag"]``
        * ``tune-prompt``  → ``current["latest_tune_prompt"]``
        * ``report``       → ``current["latest_report"]``
        """
        # Normalize ``output`` to a workspace-relative string.
        rel_out: str | None = None
        if output is not None:
            out_path = Path(output)
            try:
                rel_out = str(out_path.resolve().relative_to(self.root.resolve()))
            except ValueError:
                # Output sits outside the workspace -- keep absolute.
                rel_out = str(out_path)

        entry = HistoryEntry(
            ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            command=command,
            output=rel_out,
            run_id=run_id,
            extra=dict(extra or {}),
        )
        self.history.append(entry)

        if update_current and rel_out:
            key_map = {
                "eval": "latest_eval",
                "diagnose": "latest_diagnose",
                "tune-rag": "latest_tune_rag",
                "tune-prompt": "latest_tune_prompt",
                "report": "latest_report",
            }
            key = key_map.get(command)
            if key:
                self.current[key] = rel_out
        if run_id and command == "eval":
            self.current["baseline_run_id"] = run_id

        self.save()
        return entry

    # --------------------------------------------------------------
    # Convenience: pick the artifact to feed into the next command
    # --------------------------------------------------------------
    def latest(self, kind: str) -> Path | None:
        """Return the absolute path of the most-recent artifact of ``kind``.

        ``kind`` is one of: ``"eval"``, ``"diagnose"``,
        ``"tune-rag"``, ``"tune-prompt"``, ``"report"``. ``None`` if
        the workspace hasn't produced one yet.
        """
        key_map = {
            "eval": "latest_eval",
            "diagnose": "latest_diagnose",
            "tune-rag": "latest_tune_rag",
            "tune-prompt": "latest_tune_prompt",
            "report": "latest_report",
        }
        key = key_map.get(kind)
        if not key:
            return None
        rel = self.current.get(key)
        return self.path(rel) if rel else None


# =====================================================================
# Module-level helpers
# =====================================================================
def init_workspace(
    name: str,
    *,
    rag_config: str = "rag_config.yaml",
    questions: str = "questions.jsonl",
    corpus: str | None = None,
    evaluator: str = "ragas",
    mode: str = "auto",
    notes: str = "",
    overwrite: bool = False,
) -> Workspace:
    """Create a new workspace under :func:`workspaces_root` / ``name``.

    Does NOT copy the rag_config / questions files in -- they're paths
    the caller is responsible for placing or already pointing at.
    Raises :class:`WorkspaceError` if the workspace already exists and
    ``overwrite`` is False.
    """
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise WorkspaceError(
            f"Invalid workspace name {name!r}: must be a simple "
            "directory name (no slashes, no leading dot)."
        )
    root = workspaces_root() / name
    if root.exists() and not overwrite:
        if (root / "workspace.yaml").exists():
            raise WorkspaceError(
                f"Workspace {name!r} already exists at {root}. "
                "Use --overwrite to recreate the manifest, or pick a "
                "different name."
            )
    root.mkdir(parents=True, exist_ok=True)

    ws = Workspace(
        name=name,
        root=root.resolve(),
        rag_config=rag_config,
        questions=questions,
        corpus=corpus,
        evaluator=evaluator,
        mode=mode,
        notes=notes,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    ws.save()
    return ws


def load_workspace(name_or_path: str) -> Workspace:
    """Open an existing workspace.

    ``name_or_path`` can be:
    * a bare name (looked up under :func:`workspaces_root`)
    * an absolute or relative path to the workspace directory itself.
    """
    p = Path(name_or_path)
    root = p if p.exists() and (p / "workspace.yaml").exists() else workspaces_root() / name_or_path
    manifest = root / "workspace.yaml"
    if not manifest.exists():
        raise WorkspaceError(
            f"No workspace found at {root}. Create one with "
            f"`ragdx workspace init {name_or_path}`."
        )

    try:
        data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise WorkspaceError(
            f"Malformed workspace.yaml at {manifest}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise WorkspaceError(
            f"workspace.yaml at {manifest} must be a mapping; got {type(data).__name__}."
        )

    history_raw = data.get("history") or []
    history = [
        HistoryEntry(
            ts=str(h.get("ts", "")),
            command=str(h.get("command", "")),
            output=h.get("output"),
            run_id=h.get("run_id"),
            extra=dict(h.get("extra") or {}),
        )
        for h in history_raw if isinstance(h, dict)
    ]

    return Workspace(
        name=str(data.get("name") or root.name),
        root=root.resolve(),
        rag_config=str(data.get("rag_config") or "rag_config.yaml"),
        questions=str(data.get("questions") or "questions.jsonl"),
        corpus=data.get("corpus"),
        evaluator=str(data.get("evaluator") or "ragas"),
        mode=str(data.get("mode") or "auto"),
        created_at=str(data.get("created_at") or ""),
        history=history,
        current=dict(data.get("current") or {}),
        notes=str(data.get("notes") or ""),
    )


def list_workspaces() -> list[str]:
    """Return the names of every workspace under :func:`workspaces_root`.

    A directory counts as a workspace iff it contains a
    ``workspace.yaml`` manifest. Sorted alphabetically.
    """
    root = workspaces_root()
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "workspace.yaml").exists()
    )


__all__ = [
    "WORKSPACE_SCHEMA_VERSION",
    "HistoryEntry",
    "Workspace",
    "WorkspaceError",
    "init_workspace",
    "list_workspaces",
    "load_workspace",
    "workspaces_root",
]
