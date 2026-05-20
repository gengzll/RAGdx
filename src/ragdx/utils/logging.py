"""Unified logging for ragdx.

Configures a single hierarchical logger ``ragdx`` and exposes a ``get_logger``
helper. Level and format can be controlled via environment variables:

- ``RAGDX_LOG_LEVEL``: one of DEBUG/INFO/WARNING/ERROR/CRITICAL (default INFO).
- ``RAGDX_LOG_FORMAT``: ``plain`` (default) or ``json``.
- ``RAGDX_LOG_FILE``: optional file path to also stream logs to.

All sub-modules should call ``get_logger(__name__)`` rather than configuring
their own loggers. ``configure_logging`` is idempotent and safe to call
multiple times (subsequent calls reconfigure handlers in place).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from logging import Logger
from typing import Any

_CONFIGURED = False
_LOCK = threading.Lock()
_ROOT_NAME = "ragdx"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key in ("trial_id", "session_id", "tool", "stage", "run_id"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def _resolve_level() -> int:
    raw = os.environ.get("RAGDX_LOG_LEVEL", "INFO").upper().strip()
    return getattr(logging, raw, logging.INFO)


def configure_logging(force: bool = False) -> Logger:
    """Configure the root ``ragdx`` logger. Idempotent unless ``force=True``."""
    global _CONFIGURED
    with _LOCK:
        logger = logging.getLogger(_ROOT_NAME)
        if _CONFIGURED and not force:
            return logger
        logger.handlers.clear()
        logger.setLevel(_resolve_level())
        logger.propagate = False

        fmt_kind = os.environ.get("RAGDX_LOG_FORMAT", "plain").lower()
        if fmt_kind == "json":
            formatter: logging.Formatter = _JsonFormatter()
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        stream = logging.StreamHandler(stream=sys.stderr)
        stream.setFormatter(formatter)
        logger.addHandler(stream)

        path = os.environ.get("RAGDX_LOG_FILE")
        if path:
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        _CONFIGURED = True
        return logger


def get_logger(name: str | None = None) -> Logger:
    """Return a namespaced ``ragdx.*`` logger.

    ``name`` is usually ``__name__``; we strip a leading ``ragdx.`` to avoid
    double-prefixing and ensure all loggers descend from the root.
    """
    configure_logging()
    if not name or name == _ROOT_NAME:
        return logging.getLogger(_ROOT_NAME)
    short = name[len(_ROOT_NAME) + 1 :] if name.startswith(_ROOT_NAME + ".") else name
    return logging.getLogger(f"{_ROOT_NAME}.{short}")
