"""Repository path resolution.

All default locations are derived from the repository root so that CLI
commands work from any working directory. The vendored competition material
(see ADR-0002) is *referenced*, never copied.
"""

from __future__ import annotations

from pathlib import Path

_REPO_MARKER = "pyproject.toml"


def repo_root() -> Path:
    """Return the repository root directory.

    Walks upward from this file (src/ptcg_ai/utils/paths.py) looking for the
    repo marker, falling back to the current working directory's ancestry.
    """
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / _REPO_MARKER).is_file():
                return candidate
    raise FileNotFoundError(
        f"Could not locate repository root (no {_REPO_MARKER} found above "
        f"{Path(__file__)} or {Path.cwd()})."
    )


def default_sdk_dir() -> Path:
    """Directory containing the vendored ``cg/`` package (read-only vendor code)."""
    return (
        repo_root()
        / "pokemon-tcg-ai-battle"
        / "sample_submission"
        / "sample_submission"
    )


def default_deck_path() -> Path:
    """The sample deck shipped with the vendored SDK."""
    return default_sdk_dir() / "deck.csv"


def configs_dir() -> Path:
    """Directory holding the YAML configuration profiles."""
    return repo_root() / "configs"
