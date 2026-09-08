"""Pre-upload validation of a submission tarball (ADR-0001).

Two layers:

- :func:`validate_submission` — structural checks, no code execution.
- :func:`smoke_test_entrypoint` — extracts the tarball and exercises the
  Kaggle contract in a subprocess (deck call must return 60 ints).

Both are wired into ``ptcg validate-submission``; run them before EVERY
upload — a submission that fails Kaggle's validation episode wastes one of
the 5 daily slots.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from ptcg_ai.config.schema import DECK_SIZE, SUBMISSION_SIZE_LIMIT_MIB

#: The native library Kaggle's Linux runtime will load.
_REQUIRED_NATIVE_LIB = "cg/libcg.so"

#: Files ADR-0014 requires for the greedy agent (beyond main.py/deck.csv).
_REQUIRED_AGENT_FILES = ("card_data.json", "ptcg_ai/decision/greedy.py")

# The smoke script exercises the real Kaggle contract in a clean subprocess,
# loading main.py the way kaggle_environments.agent.get_last_callable does:
# exec(code, env) with NO `__file__` in the namespace. A previous submission
# (greedy-v1) crashed on Kaggle because `import main` (which DOES define
# __file__) let a bare module-level `os.path.abspath(__file__)` pass local
# validation while failing under Kaggle's exec-based loader. Never regress to
# `import main` here.
_SMOKE_SCRIPT = r"""
import json

_ns = {"__name__": "__main__"}
with open("main.py", "r", encoding="utf-8") as _fh:
    exec(compile(_fh.read(), "main.py", "exec"), _ns)
_agent = _ns["agent"]

deck = _agent({"select": None, "logs": [], "current": None})

_poke = {"id": 721, "serial": 1, "hp": 60, "maxHp": 60, "appearThisTurn": False,
         "energies": [3], "energyCards": [], "tools": [], "preEvolution": []}
_me = {"active": [_poke], "bench": [], "benchMax": 5, "deckCount": 40, "discard": [],
       "prize": [None] * 6, "handCount": 1,
       "hand": [{"id": 722, "serial": 2, "playerIndex": 0}],
       "poisoned": False, "burned": False, "asleep": False,
       "paralyzed": False, "confused": False}
_opp = dict(_me, hand=None, handCount=4)
main_obs = {
    "select": {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
               "remainDamageCounter": 0, "remainEnergyCost": 0,
               "option": [{"type": 7, "index": 0}, {"type": 14}],
               "deck": None, "contextCard": None, "effect": None},
    "logs": [], "current": {"turn": 3, "turnActionCount": 0, "yourIndex": 0,
        "firstPlayer": 0, "supporterPlayed": False, "stadiumPlayed": False,
        "energyAttached": False, "retreated": False, "result": -1,
        "stadium": [], "looking": None, "players": [_me, _opp]},
    "search_begin_input": "AAAA"}
action = _agent(main_obs)

# M37: a broken learned scorer does NOT raise -- policy._decide catches everything and
# plays greedy, so the bundle would look healthy while shipping the greedy pilot. Report
# the policy's own counters so the caller can tell "the model ran" from "the model died
# quietly". _budget_degraded must also be 0: degrading on the very first decision would
# mean the time guard's thresholds are misconfigured.
_pol = _ns.get("_imitation_policy")
_diag = {}
if _pol is not None:
    _diag = {"bc_used": getattr(_pol, "_bc_used", None),
             "bc_failures": getattr(_pol, "_bc_failures", None),
             "budget_degraded": getattr(_pol, "_budget_degraded", None),
             "elapsed": round(getattr(_pol, "_elapsed", 0.0), 3),
             "profile": getattr(getattr(_pol, "_profile", None), "name", None),
             "feature_dim": getattr(getattr(_pol, "_profile", None), "feature_dim", None),
             "set_contexts": sorted(getattr(_pol, "_set_specs", {}))}
print(json.dumps({"deck": deck, "ready": bool(_ns.get("_AGENT_READY")),
                  "imitation_ready": bool(_ns.get("_IMITATION_READY")), "action": action,
                  "diag": _diag}))
"""


def validate_submission(
    tarball: Path, size_limit_mib: float = SUBMISSION_SIZE_LIMIT_MIB
) -> list[str]:
    """Return a list of problems (empty = structurally valid)."""
    problems: list[str] = []
    size_mib = tarball.stat().st_size / (1024 * 1024)
    if size_mib > size_limit_mib:
        problems.append(f"tarball is {size_mib:.1f} MiB (limit {size_limit_mib} MiB)")

    with tarfile.open(tarball, "r:gz") as tar:
        names = {name.removeprefix("./") for name in tar.getnames()}
        if "main.py" not in names:
            problems.append("main.py missing at archive top level")
        if _REQUIRED_NATIVE_LIB not in names:
            problems.append(f"{_REQUIRED_NATIVE_LIB} missing (Kaggle runs Linux)")
        for required in _REQUIRED_AGENT_FILES:
            if required not in names:
                problems.append(f"{required} missing (ADR-0014 greedy agent)")
        if "deck.csv" not in names:
            problems.append("deck.csv missing")
        else:
            member = next(n for n in tar.getnames() if n.removeprefix("./") == "deck.csv")
            fh = tar.extractfile(member)
            assert fh is not None
            lines = [ln.strip() for ln in fh.read().decode().splitlines() if ln.strip()]
            if len(lines) != DECK_SIZE:
                problems.append(f"deck.csv has {len(lines)} entries (need {DECK_SIZE})")
            elif not all(ln.isdigit() for ln in lines):
                problems.append("deck.csv contains non-integer entries")
    return problems


def smoke_test_entrypoint(tarball: Path, timeout_s: float = 60.0) -> list[str]:
    """Extract and exercise the entrypoint in a clean subprocess.

    Checks the Kaggle contract's deck call, that the greedy policy actually
    engaged (``_AGENT_READY``, not a silent degrade to random — ADR-0014), and
    that a synthetic MAIN decision returns a legal selection. The subprocess
    runs with ``PYTHONPATH`` stripped so the *bundled* ``ptcg_ai`` subtree is
    imported, never the dev research tree.

    Returns problems (empty = the bundled greedy agent works end to end).
    """
    problems: list[str] = []
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    with tempfile.TemporaryDirectory(prefix="ptcg-smoke-") as tmp:
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(tmp)
        proc = subprocess.run(
            [sys.executable, "-c", _SMOKE_SCRIPT],
            cwd=tmp, capture_output=True, text=True, timeout=timeout_s, env=env,
        )
        if proc.returncode != 0:
            problems.append(f"entrypoint crashed: {proc.stderr.strip()[-500:]}")
            return problems
        try:
            out = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            problems.append(f"entrypoint printed unexpected output: {proc.stdout[:200]!r}")
            return problems

        deck = out.get("deck")
        if not (
            isinstance(deck, list)
            and len(deck) == DECK_SIZE
            and all(isinstance(c, int) for c in deck)
        ):
            problems.append(f"deck call returned invalid deck (len {len(deck) if isinstance(deck, list) else 'n/a'})")
        if not out.get("ready"):
            problems.append("greedy policy did not engage (_AGENT_READY False; silently degraded to random)")
        action = out.get("action")
        if not (isinstance(action, list) and len(action) == 1 and action[0] in (0, 1)):
            problems.append(f"MAIN decision returned an illegal selection: {action!r}")

        # M37: an imitation bundle whose scorer quietly threw would pass every check
        # above -- it returns a perfectly legal greedy action. These catch that.
        if out.get("imitation_ready"):
            diag = out.get("diag") or {}
            if diag.get("bc_failures"):
                problems.append(
                    f"imitation scorer raised on {diag['bc_failures']} decision(s) and fell "
                    f"back to greedy silently — the bundle would play greedy on the ladder")
            if not diag.get("bc_used"):
                problems.append(
                    "imitation policy never scored a decision (_bc_used == 0); the MAIN "
                    "context is missing from the payload or every call fell through")
            if diag.get("budget_degraded"):
                problems.append(
                    f"time guard degraded on the very first decision "
                    f"(budget_degraded={diag['budget_degraded']}); thresholds are misconfigured")
    return problems
