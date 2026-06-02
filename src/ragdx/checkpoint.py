"""Checkpointing for long-running ragdx experiments.

Most ragdx commands that take more than a few minutes do so because
they're iterating an inner BO loop or a DSPy MIPROv2 search: ~10-30
trials, each making several LLM calls. A network hiccup or a hung
HTTP connection mid-run wastes everything done so far.

This module provides:

* :class:`Checkpoint` — pydantic schema for what gets persisted between
  trials.
* :class:`CheckpointStore` — file-based CRUD analogous to
  :class:`ragdx.storage.run_store.RunStore`, rooted at
  ``.ragdx/checkpoints/<id>/``.

Granularity:

* **BO stages** (``tune --stage retrieval/chunking/joint`` and
  ``ragdx experiment``'s BO loop): per-trial. After every completed
  trial the optimizer writes ``trials_completed``,
  ``best_so_far``, and the chunks-master path. Resuming replays the
  completed trial scores into the BO sampler (``BayesianSearch.report``)
  before the loop continues.
* **Generation stage** (DSPy MIPROv2): per-phase. We checkpoint
  between (a) baseline run, (b) MIPROv2 ``compile()``, and (c)
  optimised re-run. Phase (b) is monolithic — DSPy doesn't expose
  mid-``compile()`` state — so a crash inside MIPROv2 still costs
  the whole phase (b). But the expensive setup (chunks, baseline
  ragas) is preserved.

Storage layout::

    .ragdx/checkpoints/
    └── ckpt_<8hex>/
        ├── state.json           ← the Checkpoint
        ├── chunks.json          ← chunks_master pool (lazy: only when
        │                          the stage signals re-chunking is
        │                          expensive enough to warrant caching)
        └── partial_bundle.json  ← what we'd write to --output now

Atomic write + best-effort cross-platform file locking are reused
from :mod:`ragdx.storage.run_store`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from ragdx.storage.run_store import _file_lock, atomic_write_text
from ragdx.utils.logging import get_logger

logger = get_logger(__name__)


CheckpointKind = Literal[
    "tune.retrieval",
    "tune.chunking",
    "tune.joint",
    "tune.generation",
    "experiment",
]

CheckpointStatus = Literal["running", "completed", "interrupted"]


class CompletedTrial(BaseModel):
    """One BO trial's worth of state — enough to skip on resume.

    Same shape as ``StageTrial.to_bundle_dict()`` so the checkpoint and
    the final bundle stay aligned.
    """

    trial_index: int
    params: dict[str, Any]
    n_chunks: int
    scores: dict[str, float]
    composite_score: float
    feasible: bool
    violations: list[str] = Field(default_factory=list)
    answers_preview: list[str] = Field(default_factory=list)
    records: list[dict[str, Any]] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


class Checkpoint(BaseModel):
    """Persisted state for one in-progress experiment."""

    schema_version: int = 1
    checkpoint_id: str = Field(default_factory=lambda: "ckpt_" + uuid4().hex[:8])
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    kind: CheckpointKind
    status: CheckpointStatus = "running"

    cli_args: dict[str, Any] = Field(default_factory=dict)
    """Raw CLI kwargs (sans secrets). Used to (a) describe the run in
    ``ragdx checkpoints`` and (b) validate that ``--resume`` is being
    applied to a compatible new invocation."""

    name: str = ""
    """Optional human-readable name (mirrors ``--name`` flag)."""

    rag_config_yaml: str = ""
    """The (scrubbed) RAGConfig the run started with. Stored so resume
    doesn't depend on the original YAML still being on disk."""

    # --- BO-stage state ---------------------------------------------
    stage_label: str = ""  # "with_gt" | "no_gt"
    search_space: dict[str, list[Any]] = Field(default_factory=dict)
    n_bo_init: int = 0
    max_trials: int = 0
    seed: int = 0
    trials_completed: list[CompletedTrial] = Field(default_factory=list)
    chunks_path: str = ""  # relative to checkpoint dir
    objective_spec: dict[str, Any] = Field(default_factory=dict)

    # --- Generation-stage state -------------------------------------
    generation_phase: Literal["", "baseline", "miprov2", "re_eval"] = ""
    """Empty for BO stages. For generation: which phase completed last."""

    generation_artifacts: dict[str, Any] = Field(default_factory=dict)
    """Per-phase artifacts: ``baseline_scores`` / ``opt_program_json``
    / ``opt_eval_scores`` etc. Survives across phase boundaries so a
    crash inside phase (b) doesn't lose phase (a)'s expensive ragas
    baseline."""

    # --- Provenance --------------------------------------------------
    interrupted_reason: str = ""
    """Free-text reason set by the optimizer's ``finally`` block or
    the signal handler: ``"network_timeout"`` / ``"kill_signal"`` /
    ``"unhandled_exception:<repr>"``."""

    def touch(self) -> None:
        """Update ``updated_at`` to ``now``. Call before each save."""
        self.updated_at = datetime.now(timezone.utc).isoformat()


class CheckpointStore:
    """File-rooted CRUD for :class:`Checkpoint`.

    Mirrors :class:`RunStore`'s patterns: atomic writes, best-effort
    file locking, defensive ``list_*`` that logs corrupt files rather
    than swallowing.

    Layout::

        <root>/<checkpoint_id>/state.json
        <root>/<checkpoint_id>/chunks.json   (optional)
    """

    def __init__(self, root: str | Path = ".ragdx/checkpoints") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def dir_for(self, checkpoint_id: str) -> Path:
        return self.root / checkpoint_id

    def state_path(self, checkpoint_id: str) -> Path:
        return self.dir_for(checkpoint_id) / "state.json"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def save(self, checkpoint: Checkpoint) -> Checkpoint:
        """Atomically persist ``checkpoint`` to ``state.json``."""
        checkpoint.touch()
        path = self.state_path(checkpoint.checkpoint_id)
        with _file_lock(path):
            atomic_write_text(path, checkpoint.model_dump_json(indent=2))
        logger.debug(
            "checkpoint saved: id=%s kind=%s status=%s trials=%d phase=%r",
            checkpoint.checkpoint_id, checkpoint.kind, checkpoint.status,
            len(checkpoint.trials_completed), checkpoint.generation_phase,
        )
        return checkpoint

    def load(self, checkpoint_id: str) -> Checkpoint:
        path = self.state_path(checkpoint_id)
        if not path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {path}. Run `ragdx checkpoints` to "
                "see the available ids."
            )
        return Checkpoint.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[Checkpoint]:
        """Return all checkpoints, most-recently-updated first.

        Corrupt state.json files are logged and skipped (so one bad
        checkpoint doesn't break ``ragdx checkpoints``)."""
        out: list[Checkpoint] = []
        if not self.root.exists():
            return out
        for entry in sorted(self.root.iterdir(), reverse=True):
            if not entry.is_dir():
                continue
            state = entry / "state.json"
            if not state.exists():
                continue
            try:
                out.append(
                    Checkpoint.model_validate_json(state.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                logger.warning("Skipping corrupt checkpoint %s: %s", entry.name, exc)
        return sorted(out, key=lambda c: c.updated_at, reverse=True)

    def list_incomplete(self) -> list[Checkpoint]:
        """Just the ``status=="running"`` ones — what ``--resume``
        (no arg) targets by default."""
        return [c for c in self.list() if c.status == "running"]

    def mark_completed(self, checkpoint_id: str) -> Checkpoint:
        c = self.load(checkpoint_id)
        c.status = "completed"
        return self.save(c)

    def mark_interrupted(self, checkpoint_id: str, reason: str = "") -> Checkpoint:
        try:
            c = self.load(checkpoint_id)
        except FileNotFoundError:
            return None  # type: ignore[return-value]
        c.status = "interrupted"
        if reason:
            c.interrupted_reason = reason
        return self.save(c)

    def delete(self, checkpoint_id: str) -> bool:
        """Remove a checkpoint directory and its contents. Returns
        ``True`` if anything was removed."""
        import shutil
        d = self.dir_for(checkpoint_id)
        if not d.exists():
            return False
        shutil.rmtree(d, ignore_errors=True)
        return True

    # ------------------------------------------------------------------
    # Side-files (chunks, etc.) — small enough to ship as JSON, big
    # enough that we don't want to round-trip them inside state.json.
    # ------------------------------------------------------------------
    def save_chunks(self, checkpoint: Checkpoint, chunks: list[str]) -> str:
        """Persist ``chunks_master`` next to ``state.json``. Returns
        the relative path stored on the checkpoint."""
        self.dir_for(checkpoint.checkpoint_id).mkdir(parents=True, exist_ok=True)
        rel = "chunks.json"
        target = self.dir_for(checkpoint.checkpoint_id) / rel
        atomic_write_text(target, json.dumps(chunks, ensure_ascii=False))
        checkpoint.chunks_path = rel
        return rel

    def load_chunks(self, checkpoint: Checkpoint) -> list[str]:
        if not checkpoint.chunks_path:
            return []
        target = self.dir_for(checkpoint.checkpoint_id) / checkpoint.chunks_path
        if not target.exists():
            return []
        return json.loads(target.read_text(encoding="utf-8"))


__all__ = [
    "Checkpoint",
    "CheckpointKind",
    "CheckpointStatus",
    "CheckpointStore",
    "CompletedTrial",
]
