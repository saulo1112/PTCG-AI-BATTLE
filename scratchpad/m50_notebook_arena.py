"""M50 — measure the public community agent (jazivxt's "A Better Hand / Alakazam Rising
Tide v21") head-to-head against our champion.

WHY THIS, AND WHY IT WAS MISSED. Nineteen milestones of work here optimised ONE family:
behaviour-cloning a teacher with an MLP (plus a set transformer, plus RL, plus search over
a learned value). The user pointed out that a notebook they had already put in the repo
root -- `a-better-hand-alakazam-rising-tide-v21.ipynb` -- was never opened. It is not a
neural architecture at all, which is exactly why it is worth measuring:

  * ~1,200 lines of HAND-WRITTEN heuristics with tuned priority weights (`WEIGHTS`,
    `heuristic_scores`) -- domain knowledge this project has never had;
  * OPPONENT BELIEF MODELLING (`_match_archetype`, `_TEMPLATES`, `_visible_opponent_line`)
    -- it identifies the opposing archetype from what has been revealed and switches plan.
    Our clone has no notion of who it is playing;
  * its OWN determinized search (`_sample_hidden`, `_leaf_eval`, `_search_decide`) over a
    hand-built leaf evaluator, rather than the learned V that M10/M18/M48 could never make
    generalise;
  * explicit Rocket Energy denial (`_rocket_energy_hammer_scores`).

Same archetype as us (4x Abra/Kadabra/Alakazam) but a DIFFERENT 60-card build: theirs adds
Night Stretcher x2, Lillie's Determination, Neutralization Zone and a different
Dunsparce/Dudunsparce printing; ours has Fezandipiti ex, Shaymin, Enhanced Hammer,
Nighttime Mine x2. So this measures deck AND policy together, as one alternative agent.

The notebook is published for community use ("exposes the exact deck, policy, archive
construction, and validation boundary for community review" -- jazivxt). Its own reported
local diagnostics are 24-16 against the public v21 and 70-50 on a five-control set; those
are ITS numbers on ITS harness, which is precisely why we re-measure on OURS, against the
agent we actually ship, with the control that has predicted the ladder twice (M43 -> M44).

THE INTERFACE MAKES THIS CHEAP. Their `agent(obs_dict)` takes the raw observation and
returns option indices, returning the 60-card deck when `select` and `current` are both
absent -- the same contract as `BasePolicy`. So no reimplementation: a thin adapter, and
the existing arena decides.

Run:
  PYTHONPATH=src python -u scratchpad/m50_notebook_arena.py probe --n 10
  PYTHONPATH=src python -u scratchpad/m50_notebook_arena.py arena --n 600
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.base import BasePolicy, DecisionContext
from ptcg_ai.decision.safety import SafePolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.runner import BattleRunner
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.evaluation.metrics import MatchStats
from ptcg_ai.imitation.policy import ImitationPolicy

NOTEBOOK = Path("a-better-hand-alakazam-rising-tide-v21.ipynb")
EXTRACTED = Path("scratchpad/_jazivxt_main.py")
NB_DECK = Path("decks/jazivxt_v21.csv")
CHAMPION = "data/models/bc_alakazam_final.json"
OUR_DECK = "decks/yushinito.csv"

_SDK = None
_CARDS = None


def _sdk_cards():
    global _SDK, _CARDS
    if _SDK is None:
        cfg = load_config(profile="benchmark")
        _SDK = load_sdk(cfg.paths.sdk_dir)
        _CARDS = CardDatabase.from_sdk(_SDK)
    return _SDK, _CARDS


def _extract() -> tuple[Path, list[int]]:
    """Pull main.py and the 60-card deck out of the notebook, verbatim.

    Re-extracted every run rather than trusting a stale copy: the whole point is to
    measure THEIR agent, and a hand-edited copy would silently measure something else.
    """
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    src = "".join(nb["cells"][7]["source"]).replace("%%writefile main.py", "", 1)
    EXTRACTED.write_text(src, encoding="utf-8")
    deck_src = "".join(nb["cells"][5]["source"])
    deck = eval(re.search(r"DECK = (\[.*?\])", deck_src, re.S).group(1))
    if len(deck) != 60:
        raise SystemExit(f"notebook deck is {len(deck)} cards, expected 60")
    NB_DECK.write_text("\n".join(map(str, deck)) + "\n", encoding="utf-8")
    return EXTRACTED, deck


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("jazivxt_v21", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["jazivxt_v21"] = mod
    spec.loader.exec_module(mod)
    return mod


class NotebookPolicy(BasePolicy):
    """Adapter: drive the notebook's `agent(obs_dict)` through our policy interface.

    Its module carries per-game global state (`my_deck`, `_TEMPLATE_SIG`, rebuilt by its
    own entrypoint), so a FRESH module instance is loaded per policy object and
    `_TEMPLATE_SIG` is reset at every battle start -- otherwise archetype beliefs inferred
    in game N would leak into game N+1 and the measurement would drift as the arena runs.
    """

    name = "notebook_v21"

    def __init__(self, path: Path, deck: list[int]) -> None:
        super().__init__(deck=deck)
        self._mod = _load_module(path)
        self._mod.my_deck = list(deck)
        self._pristine_templates = list(getattr(self._mod, "_TEMPLATE_SIG", []))
        self.errors = 0

    def on_battle_start(self) -> None:
        super().on_battle_start()
        self._mod._TEMPLATE_SIG = list(self._pristine_templates)
        self._mod.my_deck = list(self._deck)

    def choose(self, ctx: DecisionContext) -> list[int]:
        try:
            out = self._mod.agent(ctx.raw)
        except Exception as exc:  # never crash the arena on their code
            self.errors += 1
            self.last_note = f"notebook agent raised {type(exc).__name__}: {exc}"
            n = len(ctx.observation.select.option) if ctx.observation.select else 0
            return [0] if n else []
        return [int(i) for i in out] if isinstance(out, (list, tuple)) else [int(out)]

    def choose_deck(self, ctx: DecisionContext) -> list[int]:
        return list(self._deck)


def _run(n: int, workers_note: str = "") -> None:
    path, deck = _extract()
    sdk, cards = _sdk_cards()
    base_cfg = load_config(profile="benchmark")
    config = dataclasses.replace(base_cfg, paths=dataclasses.replace(
        base_cfg.paths, deck_path=NB_DECK, opponent_deck_path=Path(OUR_DECK)))

    da = load_deck(NB_DECK); dids = da.as_list()
    db = load_deck(Path(OUR_DECK)); odids = db.as_list()
    env = BattleEnvironment(config, sdk=sdk)
    runner = BattleRunner(env, max_decisions=config.battle.max_decisions)

    pol_a = SafePolicy(NotebookPolicy(path, dids), deck=dids, seed=1)
    pol_b = SafePolicy(ImitationPolicy(CHAMPION, deck=odids), deck=odids, seed=2)
    stats = MatchStats()
    t0 = time.perf_counter()
    for game in range(n):
        a_side = game % 2
        pol_a.on_battle_start(); pol_b.on_battle_start()
        if a_side == 0:
            rec = runner.run(PTCGAgent(pol_a, deck=da, cards=cards),
                             PTCGAgent(pol_b, deck=db, cards=cards), dids, odids)
        else:
            rec = runner.run(PTCGAgent(pol_b, deck=db, cards=cards),
                             PTCGAgent(pol_a, deck=da, cards=cards), odids, dids)
        stats.add(rec.outcome.winner, a_played_as=a_side)
    wall = time.perf_counter() - t0
    lo, hi = stats.wilson_interval()
    inner = pol_a._inner
    print(f"\n=== notebook v21 vs CHAMPION (n={n}, swapped) {workers_note}===")
    print(f"A[notebook]: {stats.summary()}")
    print(f"  notebook-agent errors={inner.errors}  safety interventions={pol_a.interventions}")
    print(f"  wall={wall:.0f}s ({wall / max(n,1):.2f}s/game)")
    print()
    if inner.errors:
        print(f"  *** {inner.errors} decisions fell back to option 0 after an exception —")
        print("      the number above UNDERSTATES their agent. Fix before trusting it. ***")
    if lo > 0.500:
        print("  Their agent BEATS the champion with the CI entirely above 0.500.")
    elif hi < 0.500:
        print("  Their agent LOSES to the champion with the CI entirely below 0.500.")
    else:
        print("  CI touches 0.500 — no separation at this sample size.")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, default_n in (("probe", 10), ("arena", 600)):
        p = sub.add_parser(name)
        p.add_argument("--n", type=int, default=default_n)
    args = ap.parse_args()
    _run(args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
