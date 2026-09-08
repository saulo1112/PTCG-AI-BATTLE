"""Determinizer fallback rate across full greedy-vs-greedy games, by context."""
from __future__ import annotations

import collections
import random
from pathlib import Path

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.planning.determinize import Determinizer, DeterminizeError

SAMPLE = "pokemon-tcg-ai-battle/sample_submission/sample_submission/deck.csv"


def main():
    config = load_config(profile="benchmark")
    sdk = load_sdk(config.paths.sdk_dir)
    cards = CardDatabase.from_sdk(sdk)
    deck = load_deck(Path(SAMPLE))
    ids = deck.as_list()
    parser = ObservationParser()
    agents = [PTCGAgent(GreedyPolicy(deck=ids), deck=deck, cards=cards),
              PTCGAgent(GreedyPolicy(deck=ids), deck=deck, cards=cards)]
    det = Determinizer(ids, cards, random.Random(0))
    total = collections.Counter()
    fail = collections.Counter()
    for g in range(8):
        env = BattleEnvironment(config, sdk=sdk)
        raw = env.start(ids, ids)
        for _ in range(400):
            if env.result() is not None:
                break
            parsed = parser.parse(raw)
            if parsed.current is not None and parsed.select is not None:
                ctx = parsed.select.context.name
                total[ctx] += 1
                try:
                    det.sample(parsed, 1)
                except DeterminizeError:
                    fail[ctx] += 1
            raw = env.select(agents[env.acting_player()](raw))
        env.close()
    n = sum(total.values())
    nf = sum(fail.values())
    print(f"decisions={n} fallbacks={nf} rate={100*nf/max(1,n):.1f}%")
    print(f"{'context':30s} {'total':>6s} {'fail':>6s}")
    for ctx in sorted(total, key=lambda c: -total[c]):
        print(f"{ctx:30s} {total[ctx]:6d} {fail.get(ctx,0):6d}")


if __name__ == "__main__":
    main()
