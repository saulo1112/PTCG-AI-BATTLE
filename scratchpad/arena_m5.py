"""M5 arena: rung-4 rule-based vs rung-3 greedy, and does bench-first energy
unlock the lean decks. Sequential (one-battle-per-process SDK), sides swapped.

Each matchup: policyA(deckA) vs policyB(deckB), n games. Prints score rate +
95% Wilson CI + end-reason breakdown + Mega-on-board for side A.
"""

from __future__ import annotations

import collections
import dataclasses
import sys
from pathlib import Path

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.decision.random_policy import RandomPolicy
from ptcg_ai.decision.rule_based import RuleBasedPolicy
from ptcg_ai.decision.safety import SafePolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.runner import BattleRunner
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.evaluation.metrics import MatchStats

REASON = {1: "prizes", 2: "deck-out", 3: "no-active", 4: "card-effect", None: "?"}
SAMPLE = "pokemon-tcg-ai-battle/sample_submission/sample_submission/deck.csv"


def _mk(kind: str, deck_ids, seed):
    inner = {
        "rule": RuleBasedPolicy(deck=deck_ids),
        "greedy": GreedyPolicy(deck=deck_ids),
        "random": RandomPolicy(deck=deck_ids, seed=seed),
    }[kind]
    return SafePolicy(inner, deck=deck_ids, seed=seed)


def run(a_kind, deck_a, b_kind, deck_b, label, n=300):
    base = load_config(profile="benchmark")
    config = dataclasses.replace(base, paths=dataclasses.replace(
        base.paths, deck_path=Path(deck_a), opponent_deck_path=Path(deck_b)))
    sdk = load_sdk(config.paths.sdk_dir)
    cards = CardDatabase.from_sdk(sdk)
    da, db = load_deck(config.paths.deck_path), load_deck(config.paths.opponent_deck_path)
    pol_a, pol_b = _mk(a_kind, da.as_list(), 1), _mk(b_kind, db.as_list(), 2)
    agent_a = PTCGAgent(pol_a, deck=da, cards=cards)
    agent_b = PTCGAgent(pol_b, deck=db, cards=cards)
    env = BattleEnvironment(config, sdk=sdk)
    runner = BattleRunner(env, max_decisions=config.battle.max_decisions)
    stats = MatchStats()
    wr, lr = collections.Counter(), collections.Counter()
    mega = 0
    for game in range(n):
        a_side = game % 2
        pol_a.on_battle_start(); pol_b.on_battle_start()
        seen = {"v": False}

        def hook(raw, action, player, ms, a_side=a_side, seen=seen):
            if player != a_side:
                return
            cur = raw.get("current") or {}
            pl = (cur.get("players") or [{}, {}])[cur.get("yourIndex", 0)]
            board = [x for x in (pl.get("active") or []) if x] + list(pl.get("bench") or [])
            if any(b.get("id") == 723 for b in board):
                seen["v"] = True

        if a_side == 0:
            rec = runner.run(agent_a, agent_b, da.as_list(), db.as_list(), on_decision=hook)
        else:
            rec = runner.run(agent_b, agent_a, db.as_list(), da.as_list(), on_decision=hook)
        stats.add(rec.outcome.winner, a_played_as=a_side)
        if seen["v"]:
            mega += 1
        if rec.outcome.winner == a_side:
            wr[REASON.get(rec.outcome.reason, rec.outcome.reason)] += 1
        elif rec.outcome.winner is not None:
            lr[REASON.get(rec.outcome.reason, rec.outcome.reason)] += 1
    lo, hi = stats.wilson_interval()
    print(f"\n=== {label} (n={n}, swapped) ===")
    print(f"A[{a_kind}]={Path(deck_a).stem}   B[{b_kind}]={Path(deck_b).stem}")
    print(f"A: {stats.summary()}   clears0.5? {'YES' if lo > 0.5 else 'no'}  "
          f">=0.65? {'YES' if lo > 0.65 else 'no'}")
    print(f"  A Mega-on-board: {mega}/{n} ({100*mega//n}%)  A wins:{dict(wr)}  A loses:{dict(lr)}")
    print(f"  interventions A={pol_a.interventions} B={pol_b.interventions}")


V2 = "decks/greedy_water_v2.csv"
LUC = "decks/meta_lucario.csv"
OGER = "decks/ogerpon_grass.csv"
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "inc1"
    DRAG = "decks/meta_dragapult.csv"
    if mode == "pilot":
        # Does the rung-4 pilot beat plain greedy ON OUR BEST DECK vs the meta?
        run("greedy", SAMPLE, "greedy", LUC, "greedy(sample) vs Lucario [baseline]", n=200)
        run("rule", SAMPLE, "greedy", LUC, "rule(sample)  vs Lucario", n=200)
        run("greedy", SAMPLE, "greedy", DRAG, "greedy(sample) vs Dragapult [baseline]", n=200)
        run("rule", SAMPLE, "greedy", DRAG, "rule(sample)  vs Dragapult", n=200)
    elif mode == "meta":
        # Ladder proxy: how each SHIPPING config fares vs the real meta decks
        # (piloted by greedy as a mediocre-bot stand-in), n=200 each.
        run("greedy", SAMPLE, "greedy", LUC, "v5(sample) vs Lucario", n=200)
        run("greedy", SAMPLE, "greedy", V2, "v5(sample) vs water_v2", n=200)
        run("greedy", SAMPLE, "greedy", OGER, "v5(sample) vs Ogerpon-deck", n=200)
        run("rule", OGER, "greedy", LUC, "rule(ogerpon) vs Lucario", n=200)
        run("rule", OGER, "greedy", V2, "rule(ogerpon) vs water_v2", n=200)
    elif mode == "oger":
        # THE gate: ability-piloted Ogerpon engine vs the shipped v5 config.
        run("rule", OGER, "greedy", SAMPLE, "rule(ogerpon) vs greedy(sample)=v5-config")
        run("rule", OGER, "random", OGER, "rule(ogerpon) vs safe-random [floor]")
        run("rule", OGER, "rule", SAMPLE, "rule(ogerpon) vs rule(sample)")
    elif mode == "inc1":
        # KEY TEST: new pilot + lean deck vs the shipped v5 config (greedy+sample)
        run("rule", V2, "greedy", SAMPLE, "rule(water_v2) vs greedy(sample)=v5-config")
        run("rule", SAMPLE, "greedy", SAMPLE, "rule(sample) vs greedy(sample) [pilot only]")
        run("rule", V2, "rule", SAMPLE, "rule(water_v2) vs rule(sample) [deck under new pilot]")
    elif mode == "decks":
        run("rule", LUC, "greedy", SAMPLE, "rule(lucario) vs greedy(sample)=v5-config")
        run("rule", LUC, "rule", V2, "rule(lucario) vs rule(water_v2)")
