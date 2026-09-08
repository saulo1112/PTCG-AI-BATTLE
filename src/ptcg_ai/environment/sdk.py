"""Loader for the vendored ``cg`` SDK (ADR-0002).

The SDK lives read-only under ``pokemon-tcg-ai-battle/`` and is imported from
there by path. Importing ``cg.sim`` loads the native engine library
(``cg.dll`` / ``libcg.so``) and calls ``GameInitialize()`` once.

License note: the SDK and engine are competition-use-only and must not be
copied into our source tree or redistributed.

Process model caveat: the SDK keeps battle state in module globals
(``cg.sim.Battle``), so there is at most ONE live battle per process.
Parallel self-play must use multiprocessing.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


class SdkNotAvailableError(RuntimeError):
    """The vendored SDK or its native library could not be loaded."""


@dataclass(frozen=True)
class SdkModules:
    """The imported vendored modules, bundled for injection."""

    api: ModuleType
    game: ModuleType
    sim: ModuleType


_loaded: SdkModules | None = None
_loaded_from: Path | None = None


def load_sdk(sdk_dir: Path) -> SdkModules:
    """Import the ``cg`` package from ``sdk_dir`` (idempotent).

    Args:
        sdk_dir: Directory containing the ``cg/`` package (typically
            ``pokemon-tcg-ai-battle/sample_submission/sample_submission``).

    Raises:
        SdkNotAvailableError: If the package or its native library is missing
            or fails to load on this platform.
        RuntimeError: If a *different* SDK directory was already loaded in
            this process (the native library cannot be swapped at runtime).
    """
    global _loaded, _loaded_from
    sdk_dir = sdk_dir.resolve()

    if _loaded is not None:
        if sdk_dir != _loaded_from:
            raise RuntimeError(
                f"SDK already loaded from {_loaded_from}; cannot load {sdk_dir} "
                "in the same process (native library is a process-wide singleton)."
            )
        return _loaded

    if not (sdk_dir / "cg" / "api.py").is_file():
        raise SdkNotAvailableError(f"No cg/ package found under {sdk_dir}")

    if str(sdk_dir) not in sys.path:
        sys.path.insert(0, str(sdk_dir))
    try:
        api = importlib.import_module("cg.api")
        game = importlib.import_module("cg.game")
        sim = importlib.import_module("cg.sim")
    except (ImportError, OSError) as exc:  # OSError: native lib load failure
        raise SdkNotAvailableError(f"Failed to import cg from {sdk_dir}: {exc}") from exc

    _loaded = SdkModules(api=api, game=game, sim=sim)
    _loaded_from = sdk_dir
    return _loaded


def sdk_available(sdk_dir: Path) -> bool:
    """True if :func:`load_sdk` would succeed (used by test skip logic)."""
    try:
        load_sdk(sdk_dir)
        return True
    except (SdkNotAvailableError, RuntimeError):
        return False
