"""M4 arena: ablate the W2 Trainer whitelist and gate deck v2.

Sequential battles (one-per-process SDK). Side A = paths.deck_path, side B =
opponent_deck_path, sides swapped. Prints score rate + 95% Wilson CI +
end-reason breakdown + a Mega-Abomasnow-on-board rate for side A.

Modes:
  w2   : new-whitelist greedy vs M3-whitelist greedy, both on the sample deck
         (isolates the W2 policy change); plus new-whitelist vs safe-random.
  deck : W2 greedy on deck v2 vs W2 greedy on sample (isolates the deck).
Pass deck paths as extra args for `deck` mode: `deck <candidate.csv>`.
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
from ptcg_ai.decision.safety import SafePolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.runner import BattleRunner
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.evaluation.metrics import MatchStats

REASON = {1: "prizes", 2: "deck-out", 3: "no-active", 4: "card-effect", None: "?"}
SAMPLE = "pokemon-tcg-ai-battle/sample_submission/sample_submission/deck.csv"
# The M3 (pre-W2) whitelist, for ablation.
M3_SEARCH = frozenset({1152, 1102, 1142})
M3_FETCH: frozenset = frozenset()


def _greedy(deck_ids, kind="new"):
    if kind == "m3":
        return GreedyPolicy(deck=deck_ids, search_items=M3_SEARCH, fetch_supporters=M3_FETCH)
    return GreedyPolicy(deck=deck_ids)  # new = full W2 whitelist


def run(deck_a, deck_b, label, a_kind="new", b_kind="new", b_random=False, n=300):
    base = load_config(profile="benchmark")
    config = dataclasses.replace(
        base, paths=dataclasses.replace(
            base.paths, deck_path=Path(deck_a), opponent_deck_path=Path(deck_b)))
    sdk = load_sdk(config.paths.sdk_dir)
    cards = CardDatabase.from_sdk(sdk)
    da, db = load_deck(config.paths.deck_path), load_deck(config.paths.opponent_deck_path)

    ia = _greedy(da.as_list(), a_kind)
    ib = RandomPolicy(deck=db.as_list(), seed=2) if b_random else _greedy(db.as_list(), b_kind)
    pol_a = SafePolicy(ia, deck=da.as_list(), seed=1)
    pol_b = SafePolicy(ib, deck=db.as_list(), seed=2)
    agent_a = PTCGAgent(pol_a, deck=da, cards=cards)
    agent_b = PTCGAgent(pol_b, deck=db, cards=cards)

    env = BattleEnvironment(config, sdk=sdk)
    runner = BattleRunner(env, max_decisions=config.battle.max_decisions)
    stats = MatchStats()
    wr: collections.Counter = collections.Counter()
    lr: collections.Counter = collections.Counter()
    mega_games = 0

    for game in range(n):
        a_side = game % 2
        pol_a.on_battle_start()
        pol_b.on_battle_start()
        seen_mega = {"v": False}

        def hook(raw, action, player, ms, a_side=a_side, seen=seen_mega):
            if player != a_side:
                return
            cur = raw.get("current") or {}
            yi = cur.get("yourIndex", 0)
            pl = (cur.get("players") or [{}, {}])[yi]
            board = [x for x in (pl.get("active") or []) if x] + list(pl.get("bench") or [])
            if any(b.get("id") == 723 for b in board):
                seen["v"] = True

        if a_side == 0:
            rec = runner.run(agent_a, agent_b, da.as_list(), db.as_list(), on_decision=hook)
        else:
            rec = runner.run(agent_b, agent_a, db.as_list(), da.as_list(), on_decision=hook)
        stats.add(rec.outcome.winner, a_played_as=a_side)
        if seen_mega["v"]:
            mega_games += 1
        if rec.outcome.winner == a_side:
            wr[REASON.get(rec.outcome.reason, rec.outcome.reason)] += 1
        elif rec.outcome.winner is not None:
            lr[REASON.get(rec.outcome.reason, rec.outcome.reason)] += 1

    lo, hi = stats.wilson_interval()
    b_label = "random" if b_random else b_kind
    print(f"\n=== {label} (n={n}, swapped) ===")
    print(f"A[{a_kind}]={deck_a}")
    print(f"B[{b_label}]={deck_b}")
    print(f"A: {stats.summary()}   clears0.5? {'YES' if lo > 0.5 else 'no'}")
    print(f"  A Mega-on-board: {mega_games}/{n} ({100*mega_games/n:.0f}%)")
    print(f"  reasons A wins:  {dict(wr)}")
    print(f"  reasons A loses: {dict(lr)}")
    print(f"  interventions A={pol_a.interventions} B={pol_b.interventions}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "w2"
    if mode == "w2":
        run(SAMPLE, SAMPLE, "W2 new-whitelist vs M3-whitelist (sample)", a_kind="new", b_kind="m3")
        run(SAMPLE, SAMPLE, "W2 new-whitelist vs safe-random (sample)", a_kind="new", b_random=True)
    elif mode == "deck":
        cand = sys.argv[2]
        run(cand, SAMPLE, f"deck v2 ({Path(cand).stem}) vs sample [both W2]")
        run(cand, cand, f"deck v2 ({Path(cand).stem}) vs safe-random", b_random=True)
