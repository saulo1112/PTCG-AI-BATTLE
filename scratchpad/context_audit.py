"""Audit which SelectContexts greedy encounters piloting a given deck, and
flag any that fall through to the safe default (unhandled) — a Kaggle-
robustness check before shipping a new deck.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.runner import BattleRunner
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.observation.parser import ObservationParser

deck_path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
config = load_config(profile="benchmark")
sdk = load_sdk(config.paths.sdk_dir)
cards = CardDatabase.from_sdk(sdk)
deck = load_deck(Path(deck_path))
parser = ObservationParser()

seen = collections.Counter()
unhandled = collections.Counter()
HANDLED = {"MAIN", "IS_FIRST", "SETUP_ACTIVE_POKEMON", "TO_ACTIVE", "SWITCH",
           "SETUP_BENCH_POKEMON", "TO_HAND", "ATTACH_FROM", "ATTACH_TO",
           "DRAW_COUNT", "DAMAGE_COUNTER_COUNT", "REMOVE_DAMAGE_COUNTER_COUNT"}

pol = GreedyPolicy(deck=deck.as_list())
agent = PTCGAgent(pol, deck=deck, cards=cards)
env = BattleEnvironment(config, sdk=sdk)
runner = BattleRunner(env, max_decisions=config.battle.max_decisions)


def hook(raw, action, player, ms):
    obs = parser.parse(raw)
    sel = obs.select
    if sel is None:
        return
    key = f"{sel.type.name}/{sel.context.name}"
    seen[key] += 1
    is_count = sel.type.name == "COUNT"
    if sel.context.name not in HANDLED and not is_count and sel.type.name != "MAIN":
        unhandled[key] += 1


for g in range(n):
    pol.on_battle_start()
    runner.run(agent, PTCGAgent(GreedyPolicy(deck=deck.as_list()), deck=deck, cards=cards),
               deck.as_list(), deck.as_list(), on_decision=hook)

print(f"deck={deck_path}  games={n}")
print("contexts seen (top 25):")
for k, v in seen.most_common(25):
    flag = "  <-- UNHANDLED (safe-default)" if k in unhandled else ""
    print(f"  {v:>6}  {k}{flag}")
print("\nUNHANDLED contexts:", dict(unhandled) or "NONE — all routed to a real handler")
print("SafePolicy interventions:", getattr(pol, "interventions", "n/a"))
