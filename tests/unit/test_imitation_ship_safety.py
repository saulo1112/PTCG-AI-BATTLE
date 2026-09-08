"""The shipped imitation modules must import only stdlib + the bundled package.

The Kaggle runtime has no numpy/sklearn (ADR-0014); a stray third-party import
in ``imitation/features.py`` or ``imitation/policy.py`` would crash the agent at
load time. This test parses their imports and fails on anything that is not
stdlib or ``ptcg_ai``. It also asserts they are on the submission allowlist.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from ptcg_ai.submission.builder import _AGENT_MODULES

_SHIPPED = ("imitation/deck_profiles.py", "imitation/features.py", "imitation/policy.py",
            "imitation/setnet.py")
_PKG_ROOT = Path(__file__).parents[2] / "src" / "ptcg_ai"
_ALLOWED_ROOTS = set(sys.stdlib_module_names) | {"ptcg_ai", "__future__"}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_shipped_imitation_modules_are_stdlib_only() -> None:
    for rel in _SHIPPED:
        roots = _imported_roots(_PKG_ROOT / rel)
        offenders = roots - _ALLOWED_ROOTS
        assert not offenders, f"{rel} imports non-stdlib/non-ptcg_ai modules: {sorted(offenders)}"


def test_shipped_modules_on_allowlist() -> None:
    for rel in _SHIPPED:
        assert rel in _AGENT_MODULES, f"{rel} not in builder _AGENT_MODULES"
