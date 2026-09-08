"""M3 throwaway arena.

Measures deck/policy effects with `run_matchup`-style hosting. Sequential
battles (one-battle-per-process SDK). Side A = paths.deck_path; side B =
opponent_deck_path. Sides swapped each game. Prints score rate + 95% Wilson CI
+ end-reason breakdown for side A.

Usage: python scratchpad/arena_m3.py <mode>   where mode in {A, B, head}.
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


def _policy(kind: str, deck_ids: list[int], seed: int):
    inner = GreedyPolicy(deck=deck_ids) if kind == "greedy" else RandomPolicy(deck=deck_ids, seed=seed)
    return SafePolicy(inner, deck=deck_ids, seed=seed)


def run(deck_a_path: str, deck_b_path: str, label: str,
        policy_a: str = "greedy", policy_b: str = "greedy", n: int = 300) -> None:
    base = load_config(profile="benchmark")
    config = dataclasses.replace(
        base,
        paths=dataclasses.replace(
            base.paths, deck_path=Path(deck_a_path), opponent_deck_path=Path(deck_b_path)
        ),
    )
    sdk = load_sdk(config.paths.sdk_dir)
    cards = CardDatabase.from_sdk(sdk)
    deck_a = load_deck(config.paths.deck_path)
    deck_b = load_deck(config.paths.opponent_deck_path)

    pol_a = _policy(policy_a, deck_a.as_list(), seed=1)
    pol_b = _policy(policy_b, deck_b.as_list(), seed=2)
    agent_a = PTCGAgent(pol_a, deck=deck_a, cards=cards)
    agent_b = PTCGAgent(pol_b, deck=deck_b, cards=cards)

    env = BattleEnvironment(config, sdk=sdk)
    runner = BattleRunner(env, max_decisions=config.battle.max_decisions)
    stats = MatchStats()
    wins_reasons: collections.Counter = collections.Counter()
    loss_reasons: collections.Counter = collections.Counter()

    for game in range(n):
        a_side = game % 2
        pol_a.on_battle_start()
        pol_b.on_battle_start()
        if a_side == 0:
            rec = runner.run(agent_a, agent_b, deck_a.as_list(), deck_b.as_list())
        else:
            rec = runner.run(agent_b, agent_a, deck_b.as_list(), deck_a.as_list())
        stats.add(rec.outcome.winner, a_played_as=a_side)
        if rec.outcome.winner == a_side:
            wins_reasons[REASON.get(rec.outcome.reason, rec.outcome.reason)] += 1
        elif rec.outcome.winner is not None:
            loss_reasons[REASON.get(rec.outcome.reason, rec.outcome.reason)] += 1

    lo, hi = stats.wilson_interval()
    print(f"\n=== {label} (n={n}, sides swapped) ===")
    print(f"A[{policy_a}]={deck_a_path}")
    print(f"B[{policy_b}]={deck_b_path}")
    print(f"A: {stats.summary()}")
    print(f"  clears 0.5? {'YES' if lo > 0.5 else 'no'}  (CI {lo:.3f}..{hi:.3f})")
    print(f"  reasons A wins:  {dict(wins_reasons)}")
    print(f"  reasons A loses: {dict(loss_reasons)}")
    print(f"  interventions A={pol_a.interventions} B={pol_b.interventions}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "B"
    if mode == "A":
        run("decks/meta_lucario.csv", SAMPLE, "Lucario vs Sample [old-policy repro]")
        run("decks/meta_dragapult.csv", SAMPLE, "Dragapult vs Sample")
    elif mode == "B":
        # B1: deck decision under the shipping (trainer-enabled) policy.
        run("decks/meta_lucario.csv", SAMPLE, "B1 greedy(Lucario) vs greedy(Sample)")
        # B2: does trainer play regress the sample-deck agent vs the M2 floor?
        run(SAMPLE, SAMPLE, "B2 greedy(Sample) vs safe-random(Sample)", policy_b="random")
        # B3: candidate absolute strength vs the common floor.
        run("decks/meta_lucario.csv", "decks/meta_lucario.csv",
            "B3 greedy(Lucario) vs safe-random(Lucario)", policy_b="random")
    elif mode == "head":
        run("decks/meta_lucario.csv", "decks/meta_dragapult.csv", "Lucario vs Dragapult")
