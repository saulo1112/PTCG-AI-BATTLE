"""Logging setup for the ptcg_ai package.

One function, called once by entry points (CLI, tests). Library modules just
use ``logging.getLogger(__name__)`` and never configure handlers themselves.
"""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure root logging for a process.

    Args:
        level: Root log level name (e.g. ``"DEBUG"``, ``"INFO"``).
        log_dir: If given, also write ``ptcg_ai.log`` inside this directory.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "ptcg_ai.log", encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_FORMAT,
        handlers=handlers,
        force=True,
    )
