"""Regression test for the Kaggle exec-loading crash (greedy-v1 incident).

Kaggle loads ``main.py`` via ``kaggle_environments.agent.get_last_callable``,
which does ``exec(code_object, env)`` — and ``env`` has no ``__file__`` key.
A previous submission (see docs/decisions/0014-submission-packaging.md,
"Implementation note (2026-07-06 hotfix)") crashed the whole agent at module
load because a bare module-level ``os.path.abspath(__file__)`` raised
``NameError`` under that loader. Local validation missed it because
``import main`` (the old smoke-test loading mechanism) *does* define
``__file__``.

This test reproduces Kaggle's exact loading mechanism directly against the
entrypoint source — no tarball build, no SDK needed — so this bug class can
never regress silently.
"""

from __future__ import annotations

from pathlib import Path

_ENTRYPOINT = (
    Path(__file__).parents[2] / "src" / "ptcg_ai" / "submission" / "_entrypoint.py"
)


def _exec_like_kaggle() -> dict:
    """Load the entrypoint source in a namespace with NO ``__file__`` key,
    exactly as ``kaggle_environments.agent.get_last_callable`` does."""
    source = _ENTRYPOINT.read_text(encoding="utf-8")
    ns: dict = {"__name__": "__main__"}
    exec(compile(source, "main.py", "exec"), ns)
    return ns


def test_module_loads_without_dunder_file() -> None:
    """Module-level code must never raise NameError('__file__') under exec."""
    ns = _exec_like_kaggle()
    assert callable(ns.get("agent"))


def test_first_kaggle_call_never_raises_after_exec_load() -> None:
    """The exact call that crashed on Kaggle: the initial deck-submission call
    (``select is None``) on a freshly exec-loaded module. No card_data.json/
    deck.csv is present in this test's cwd, so this exercises the last-resort
    fallback path too — it must return a list, never raise."""
    ns = _exec_like_kaggle()
    agent = ns["agent"]
    result = agent({"select": None, "logs": [], "current": None})
    assert isinstance(result, list)
