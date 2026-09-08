"""M8 arena: imitation-v2 (BC of the 800-elo Lucario pilot) vs imitation-v1.

The promotion gate is now our OWN champion (imitation-v1, ladder ~657), not
greedy. imitation-v2 must beat it to ship — that is the "guarantee superior to
v1" contract. Swapped sides, Wilson 95% CI + end-reason breakdown.

Modes (n defaults in parens):
  smoke  v2 vs v1                                        (20)   sanity
  a      imitation-v2 vs imitation-v1     [PROMOTION]    (300)  ship gate: lo>0.5
  b      imitation-v2 vs greedy(800deck)  [PILOT-LIFT]   (200)  same-deck skill
  c      imitation-v2 vs {greedy-v5 sample, meta_lucario, meta_dragapult}  (150) gauntlet
  d      imitation-v2 mirror              [SELF-SANITY]  (50)   traces
  all    a + b + c + d
"""

from __future__ import annotations

import collections
import dataclasses
import sys
import time
from pathlib import Path

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.decision.safety import SafePolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.runner import BattleRunner
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.evaluation.metrics import MatchStats
from ptcg_ai.imitation.policy import ImitationPolicy

REASON = {1: "prizes", 2: "deck-out", 3: "no-active", 4: "card-effect", None: "?"}
SAMPLE = "pokemon-tcg-ai-battle/sample_submission/sample_submission/deck.csv"
G650 = "decks/greengreenpurple.csv"
L800 = "decks/lucario800.csv"
LUC = "decks/meta_lucario.csv"
DRAG = "decks/meta_dragapult.csv"
W_V1 = "data/models/bc_650_v1.json"
W_V2 = "data/models/bc_800_v1.json"

_SDK = None
_CARDS = None


def _mk(kind, deck_ids, seed, weights=None):
    if kind == "imitation":
        inner = ImitationPolicy(weights, deck=deck_ids)
    elif kind == "greedy":
        inner = GreedyPolicy(deck=deck_ids)
    else:
        raise ValueError(kind)
    return SafePolicy(inner, deck=deck_ids, seed=seed)


def run(a_kind, deck_a, b_kind, deck_b, label, n=200, wa=None, wb=None):
    global _SDK, _CARDS
    base = load_config(profile="benchmark")
    config = dataclasses.replace(base, paths=dataclasses.replace(
        base.paths, deck_path=Path(deck_a), opponent_deck_path=Path(deck_b)))
    if _SDK is None:
        _SDK = load_sdk(config.paths.sdk_dir)
        _CARDS = CardDatabase.from_sdk(_SDK)
    sdk, cards = _SDK, _CARDS
    da, db = load_deck(Path(deck_a)), load_deck(Path(deck_b))
    pol_a = _mk(a_kind, da.as_list(), 1, wa)
    pol_b = _mk(b_kind, db.as_list(), 2, wb)
    agent_a = PTCGAgent(pol_a, deck=da, cards=cards)
    agent_b = PTCGAgent(pol_b, deck=db, cards=cards)
    env = BattleEnvironment(config, sdk=sdk)
    runner = BattleRunner(env, max_decisions=config.battle.max_decisions)
    stats = MatchStats()
    wr, lr = collections.Counter(), collections.Counter()
    t_start = time.perf_counter()

    for game in range(n):
        a_side = game % 2
        pol_a.on_battle_start()
        pol_b.on_battle_start()
        if a_side == 0:
            rec = runner.run(agent_a, agent_b, da.as_list(), db.as_list())
        else:
            rec = runner.run(agent_b, agent_a, db.as_list(), da.as_list())
        stats.add(rec.outcome.winner, a_played_as=a_side)
        if rec.outcome.winner == a_side:
            wr[REASON.get(rec.outcome.reason, rec.outcome.reason)] += 1
        elif rec.outcome.winner is not None:
            lr[REASON.get(rec.outcome.reason, rec.outcome.reason)] += 1

    wall = time.perf_counter() - t_start
    lo, hi = stats.wilson_interval()
    print(f"\n=== {label} (n={n}, swapped) ===")
    print(f"A[{a_kind}]={Path(deck_a).stem}   B[{b_kind}]={Path(deck_b).stem}")
    print(f"A: {stats.summary()}   clears0.5? {'YES' if lo > 0.5 else 'no'}  "
          f">=0.65(lo>0.5)? {'SHIP' if (lo > 0.5 and stats.score_rate >= 0.65) else 'no'}")
    print(f"  A wins:{dict(wr)}  A loses:{dict(lr)}")
    print(f"  interventions A={pol_a.interventions} B={pol_b.interventions}")
    ai = pol_a._inner
    if isinstance(ai, ImitationPolicy):
        print(f"  bc_used={ai._bc_used} bc_failures={ai._bc_failures}  "
              f"wall={wall:.0f}s ({wall/n:.2f}s/game)")
    return stats


def _a(n=300):
    run("imitation", L800, "imitation", G650, "A PROMOTION v2(lucario800) vs v1(650)", n=n, wa=W_V2, wb=W_V1)

def _b(n=200):
    run("imitation", L800, "greedy", L800, "B PILOT-LIFT v2 vs greedy(lucario800)", n=n, wa=W_V2)

def _c(n=150):
    run("imitation", L800, "greedy", SAMPLE, "C v2 vs greedy-v5(sample) [bar 0.65]", n=n, wa=W_V2)
    run("imitation", L800, "greedy", LUC, "C v2 vs greedy(meta_lucario) [near-mirror]", n=n, wa=W_V2)
    run("imitation", L800, "greedy", DRAG, "C v2 vs greedy(meta_dragapult) [bar 0.55]", n=n, wa=W_V2)

def _d(n=50):
    run("imitation", L800, "imitation", L800, "D SELF v2 mirror", n=n, wa=W_V2, wb=W_V2)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if mode == "smoke":
        run("imitation", L800, "imitation", G650, "smoke v2 vs v1", n=n or 20, wa=W_V2, wb=W_V1)
    elif mode == "a":
        _a(n or 300)
    elif mode == "b":
        _b(n or 200)
    elif mode == "c":
        _c(n or 150)
    elif mode == "d":
        _d(n or 50)
    elif mode == "all":
        _a(n or 300); _b(200); _c(150); _d(50)
    else:
        raise SystemExit(f"unknown mode {mode!r}")
