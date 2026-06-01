"""Persistent storage for ragdx runs, sessions, feedback and causal priors.

Production-hardened changes vs. the original:

- ``atomic_write_text`` writes to a temp file in the same directory then renames,
  so a crash mid-write never leaves a half-written JSON behind.
- A per-store ``threading.RLock`` plus an OS-level ``fcntl`` / ``msvcrt`` file
  lock guards concurrent writes from multiple processes (best-effort: graceful
  degradation when locking is unavailable).
- ``list_runs`` / ``list_sessions`` / ``list_feedback`` log corrupt files rather
  than swallowing them silently.
- Explicit ``RunNotFoundError`` / ``SessionNotFoundError`` exceptions instead of
  bubbling raw ``FileNotFoundError``.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ragdx.schemas.models import (
    DiagnosisReport,
    EvaluationResult,
    FeedbackEvent,
    OptimizationPlan,
    OptimizationSession,
    SavedRun,
)
from ragdx.schemas.rag_config import RAGConfig
from ragdx.utils.logging import get_logger

logger = get_logger(__name__)


class RunNotFoundError(KeyError):
    """Raised when a run id cannot be located."""


class SessionNotFoundError(KeyError):
    """Raised when a session id cannot be located."""


# ---------------------------------------------------------------------------
# Cross-platform best-effort file locking. Falls back to a no-op when neither
# fcntl (POSIX) nor msvcrt (Windows) is available.
# ---------------------------------------------------------------------------
try:  # POSIX
    import fcntl  # type: ignore[attr-defined]

    @contextmanager
    def _file_lock(path: Path) -> Iterator[None]:
        path.parent.mkdir(parents=True, exist_ok=True)
        lockfile = path.with_suffix(path.suffix + ".lock")
        with open(lockfile, "w") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
except ImportError:  # pragma: no cover - exercised on Windows
    try:
        import msvcrt  # type: ignore[attr-defined]

        @contextmanager
        def _file_lock(path: Path) -> Iterator[None]:
            path.parent.mkdir(parents=True, exist_ok=True)
            lockfile = path.with_suffix(path.suffix + ".lock")
            with open(lockfile, "w") as fh:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                    yield
                finally:
                    try:
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
    except ImportError:  # pragma: no cover
        @contextmanager
        def _file_lock(path: Path) -> Iterator[None]:
            yield


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically via temp file + ``os.replace``.

    Uses the destination directory for the temp file so that ``os.replace`` is
    a same-filesystem rename and therefore atomic on POSIX and Windows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class RunStore:
    def __init__(self, root: str | Path = ".ragdx"):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.sessions_dir = self.root / "optimization" / "sessions"
        self.feedback_dir = self.root / "feedback"
        self.causal_dir = self.root / "causal"
        for d in (self.runs_dir, self.sessions_dir, self.feedback_dir, self.causal_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ---- path helpers ------------------------------------------------------
    def _run_path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.json"

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def _feedback_path(self, feedback_id: str) -> Path:
        return self.feedback_dir / f"{feedback_id}.json"

    def _causal_priors_path(self) -> Path:
        return self.causal_dir / "priors.json"

    # ---- safe writers ------------------------------------------------------
    def _write_json(self, path: Path, content: str) -> None:
        with self._lock, _file_lock(path):
            atomic_write_text(path, content)

    # ---- run CRUD ----------------------------------------------------------
    def save_run(
        self,
        evaluation: EvaluationResult,
        diagnosis: DiagnosisReport,
        plan: OptimizationPlan,
        name: str | None = None,
        tags: list[str] | None = None,
        notes: str = "",
        baseline_run_id: str | None = None,
        latest_session_id: str | None = None,
        rag_config: RAGConfig | None = None,
    ) -> SavedRun:
        ts = datetime.now(timezone.utc).isoformat()
        # ``rag_config`` is the production RAG description that produced
        # ``evaluation``. Storing it lets ``ragdx tune --from-run <id>``
        # inherit the base config + plan without an explicit
        # ``--base-config`` flag. Defensive scrub: even if the caller
        # forgot ``scrubbed_for_commit()``, we never persist credentials.
        safe_config = rag_config.scrubbed_for_commit() if rag_config is not None else None
        run = SavedRun(
            run_id=uuid4().hex[:12],
            created_at=ts,
            name=name or evaluation.metadata.get("dataset", "unnamed-run"),
            tags=tags or [],
            notes=notes,
            baseline_run_id=baseline_run_id,
            latest_session_id=latest_session_id,
            evaluation=evaluation,
            diagnosis=diagnosis,
            optimization_plan=plan,
            rag_config=safe_config,
        )
        self._write_json(self._run_path(run.run_id), run.model_dump_json(indent=2))
        logger.info("Saved run %s name=%s", run.run_id, run.name, extra={"run_id": run.run_id})
        return run

    def update_run_latest_session(self, run_id: str, session_id: str) -> SavedRun:
        with self._lock:
            run = self.load_run(run_id)
            run.latest_session_id = session_id
            self._write_json(self._run_path(run_id), run.model_dump_json(indent=2))
            return run

    def attach_feedback_to_run(self, run_id: str, feedback_events: list[FeedbackEvent]) -> SavedRun:
        with self._lock:
            run = self.load_run(run_id)
            run.evaluation.feedback_events.extend(feedback_events)
            self._write_json(self._run_path(run_id), run.model_dump_json(indent=2))
            for event in feedback_events:
                self._write_json(
                    self._feedback_path(event.feedback_id),
                    event.model_dump_json(indent=2),
                )
            logger.info(
                "Attached %d feedback events to run=%s", len(feedback_events), run_id,
                extra={"run_id": run_id},
            )
            return run

    def load_run(self, run_id: str) -> SavedRun:
        path = self._run_path(run_id)
        if not path.exists():
            raise RunNotFoundError(run_id)
        return SavedRun.model_validate_json(path.read_text(encoding="utf-8"))

    def list_runs(self) -> list[SavedRun]:
        runs: list[SavedRun] = []
        for path in sorted(self.runs_dir.glob("*.json"), reverse=True):
            try:
                runs.append(SavedRun.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.warning("Skipping corrupt run file %s: %s", path.name, exc)
        return sorted(runs, key=lambda r: r.created_at, reverse=True)

    def latest(self) -> SavedRun | None:
        runs = self.list_runs()
        return runs[0] if runs else None

    # ---- session CRUD ------------------------------------------------------
    def save_session(self, session: OptimizationSession) -> OptimizationSession:
        self._write_json(self._session_path(session.session_id), session.model_dump_json(indent=2))
        return session

    def load_session(self, session_id: str) -> OptimizationSession:
        path = self._session_path(session_id)
        if not path.exists():
            raise SessionNotFoundError(session_id)
        return OptimizationSession.model_validate_json(path.read_text(encoding="utf-8"))

    def upsert_session(self, session: OptimizationSession) -> OptimizationSession:
        return self.save_session(session)

    def session_exists(self, session_id: str) -> bool:
        return self._session_path(session_id).exists()

    def list_sessions(self) -> list[OptimizationSession]:
        sessions: list[OptimizationSession] = []
        for path in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            try:
                sessions.append(OptimizationSession.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.warning("Skipping corrupt session file %s: %s", path.name, exc)
        return sorted(sessions, key=lambda s: s.created_at, reverse=True)

    def latest_session(self) -> OptimizationSession | None:
        sessions = self.list_sessions()
        return sessions[0] if sessions else None

    # ---- feedback ----------------------------------------------------------
    def list_feedback(self) -> list[FeedbackEvent]:
        items: list[FeedbackEvent] = []
        for path in sorted(self.feedback_dir.glob("*.json"), reverse=True):
            try:
                items.append(FeedbackEvent.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.warning("Skipping corrupt feedback file %s: %s", path.name, exc)
        return items

    def feedback_summary(self) -> dict[str, float]:
        items = self.list_feedback()
        if not items:
            return {"total_feedback": 0.0}
        total = len(items)
        negative = sum(
            1 for x in items
            if x.kind in {"thumbs_down", "user_correction", "escalation", "hallucination", "policy"}
        )
        rated = [x.rating for x in items if x.rating is not None]
        return {
            "total_feedback": float(total),
            "negative_feedback_rate": round(negative / total, 4),
            "avg_rating": round(sum(rated) / max(1, len(rated)), 4) if rated else 0.0,
        }

    # ---- causal priors -----------------------------------------------------
    def load_causal_priors(self, defaults: dict[str, float]) -> dict[str, float]:
        path = self._causal_priors_path()
        if not path.exists():
            return dict(defaults)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            merged = dict(defaults)
            merged.update({k: float(v) for k, v in payload.items() if k in defaults})
            return merged
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Causal priors unreadable, using defaults: %s", exc)
            return dict(defaults)

    def save_causal_priors(self, priors: dict[str, float]) -> dict[str, float]:
        self._write_json(self._causal_priors_path(), json.dumps(priors, indent=2))
        return priors

    def update_causal_priors_from_report(
        self,
        diagnosis: DiagnosisReport,
        feedback_events: list[FeedbackEvent] | None = None,
        learning_rate: float = 0.15,
    ) -> dict[str, float]:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1].")
        priors = self.load_causal_priors({s.node: s.prior for s in diagnosis.causal_signals} or {})
        severity_bonus = 0.0
        feedback_events = feedback_events or []
        if feedback_events:
            severe = sum(
                1 for e in feedback_events
                if e.kind in {"hallucination", "policy", "escalation", "user_correction"}
            )
            severity_bonus = min(0.08, 0.02 * severe)
        for signal in diagnosis.causal_signals:
            old = priors.get(signal.node, signal.prior)
            target = signal.posterior + severity_bonus if signal.posterior >= signal.prior else signal.posterior
            target = min(0.98, max(0.01, target))
            priors[signal.node] = round((1 - learning_rate) * old + learning_rate * target, 4)
        self.save_causal_priors(priors)
        return priors

    # ---- markdown report ---------------------------------------------------
    def export_markdown(self, run_id: str, output_path: str | Path) -> Path:
        run = self.load_run(run_id)
        lines = [
            f"# ragdx run report: {run.name}",
            "",
            f"- Run ID: `{run.run_id}`",
            f"- Created at: {run.created_at}",
            f"- Baseline run: `{run.baseline_run_id or 'none'}`",
            f"- Tags: {', '.join(run.tags) if run.tags else 'none'}",
            f"- Latest optimization session: `{run.latest_session_id or 'none'}`",
            "",
            "## Summary",
            run.diagnosis.summary,
            "",
            f"- Diagnosis confidence: {run.diagnosis.diagnosis_confidence:.2f}",
            f"- Feedback events attached: {len(run.evaluation.feedback_events)}",
            f"- Query traces attached: {len(run.evaluation.traces)}",
            "",
            "## Retrieval metrics",
        ]
        for k, v in run.evaluation.retrieval.items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("\n## Generation metrics")
        for k, v in run.evaluation.generation.items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("\n## End-to-end metrics")
        for k, v in run.evaluation.e2e.items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("\n## Evaluator agreement")
        for k, v in run.diagnosis.evaluator_agreement.items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("\n## Hypotheses")
        for h in run.diagnosis.hypotheses:
            lines.append(
                f"- **{h.root_cause}** ({h.component}, severity={h.severity}, "
                f"confidence={h.confidence:.2f})"
            )
        lines.append("\n## Causal signals")
        for s in run.diagnosis.causal_signals[:5]:
            lines.append(
                f"- **{s.node}** ({s.component}) posterior={s.posterior:.2f}, prior={s.prior:.2f}"
            )
        lines.append("\n## Planned experiments")
        for e in run.optimization_plan.experiments:
            lines.append(
                f"- **{e.name}** stage=`{e.stage}` via `{e.tool}` targeting "
                f"`{e.target_component}`: {e.description}"
            )
        if run.latest_session_id:
            try:
                session = self.load_session(run.latest_session_id)
                lines.append("\n## Latest optimization session")
                lines.append(f"- Session ID: `{session.session_id}`")
                lines.append(f"- Strategy: `{session.strategy}`")
                lines.append(f"- Mode: `{session.mode}`")
                lines.append(f"- Status: `{session.status}`")
                lines.append(
                    f"- Completed trials: {session.completed_trials}/{session.total_trials}"
                )
            except SessionNotFoundError as exc:
                logger.warning("Run %s references missing session %s", run_id, exc)
        out = Path(output_path)
        atomic_write_text(out, "\n".join(lines))
        return out
