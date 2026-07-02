"""Internal helpers shared across the cli/ subpackage."""

from __future__ import annotations

from ragdx.config import get_settings
from ragdx.storage.run_store import RunStore


def _store() -> RunStore:
    """Build a :class:`RunStore` rooted at the configured storage path."""
    return RunStore(root=str(get_settings().storage.root))


__all__ = ["_store"]
