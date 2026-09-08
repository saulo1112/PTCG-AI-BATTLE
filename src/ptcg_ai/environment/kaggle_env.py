"""Optional wrapper over the ``kaggle-environments`` "cabt" host.

Purpose: submission-fidelity runs — the exact harness Kaggle uses, including
the initial deck-submission call that local SDK hosting skips.

``kaggle-environments`` is NOT a project dependency: every release containing
the cabt env requires Python >= 3.11 and open_spiel (no Windows wheels).
Run fidelity checks from a separate Linux/WSL Python 3.11+ environment —
see docs/environment.md. This module degrades with a clear error otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ptcg_ai.environment.runner import AgentCallable

_INSTALL_HINT = (
    "kaggle-environments with the 'cabt' env requires Python >= 3.11 on Linux "
    "(open_spiel has no Windows wheels). Create a separate WSL/Linux venv and "
    "`pip install kaggle-environments`. See docs/environment.md."
)


class KaggleEnvUnavailableError(RuntimeError):
    """kaggle-environments (or its cabt env) is not importable here."""


def run_kaggle_battle(
    agent0: AgentCallable,
    agent1: AgentCallable,
    deck0: list[int],
    deck1: list[int],
    html_out: Path | None = None,
) -> list[Any]:
    """Run one battle under the official Kaggle harness.

    Returns the final env state list (per kaggle-environments conventions).
    Optionally renders the match to an HTML replay file.

    Raises:
        KaggleEnvUnavailableError: When kaggle-environments/cabt cannot be
            imported in this interpreter.
    """
    try:
        from kaggle_environments import make
    except ImportError as exc:
        raise KaggleEnvUnavailableError(_INSTALL_HINT) from exc

    try:
        env = make("cabt", configuration={"decks": [deck0, deck1]})
    except Exception as exc:  # unknown env name, version too old, etc.
        raise KaggleEnvUnavailableError(f"make('cabt') failed: {exc}. {_INSTALL_HINT}") from exc

    result = env.run([agent0, agent1])
    if html_out is not None:
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(env.render(mode="html"), encoding="utf-8")
    return result
