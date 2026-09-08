"""Find the card the determinizer under-subtracts on real observations."""
from __future__ import annotations

import random
from collections import Counter
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


def dump(state, me, deck_ids, cards):
    deck = Counter(deck_ids)
    seen = Counter()
    for c in me.hand or ():
        seen[c.id] += 1
    for c in me.discard:
        seen[c.id] += 1
    for slot in me.active:
        if slot:
            seen[slot.id] += 1
            for e in slot.energyCards: seen[e.id] += 1
            for t in slot.tools: seen[t.id] += 1
            for p in slot.preEvolution: seen[p.id] += 1
    for pk in me.bench:
        seen[pk.id] += 1
        for e in pk.energyCards: seen[e.id] += 1
        for t in pk.tools: seen[t.id] += 1
        for p in pk.preEvolution: seen[p.id] += 1
    for c in state.stadium:
        seen[c.id] += 1  # count ALL stadiums to see
    for c in me.prize:
        if c is not None:
            seen[c.id] += 1
    print("  yourIndex", state.yourIndex, "deckCount", me.deckCount,
          "hand", me.handCount, "bench", len(me.bench),
          "discard", len(me.discard), "prizes", len(me.prize),
          "stadium", [(c.id, c.playerIndex) for c in state.stadium],
          "looking", me is not None and (state.looking is not None))
    # identity check
    total = me.deckCount + (me.handCount) + sum(1 for s in me.active if s) + len(me.bench)
    total += len(me.discard) + len(me.prize)
    print("  identity deck+hand+board+discard+prizes =", total,
          "(+stadium not counted)")
    extra = seen - deck
    if extra:
        print("  *** cards seen MORE than in deck:", dict(extra))
    residual = deck - seen
    print("  pool(all-stadiums-counted) =", sum(residual.values()))


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
    fails = 0
    for g in range(3):
        env = BattleEnvironment(config, sdk=sdk)
        raw = env.start(ids, ids)
        for _ in range(60):
            if env.result() is not None:
                break
            parsed = parser.parse(raw)
            if parsed.current is not None and parsed.select is not None:
                try:
                    det.sample(parsed, 1)
                except DeterminizeError as e:
                    fails += 1
                    sel = parsed.select
                    me = parsed.current.me
                    print(f"FAIL g{g}: {e}")
                    print(f"  ctx={sel.context.name} activeLen={len(me.active)} "
                          f"active0={'set' if (me.active and me.active[0]) else 'None/empty'} "
                          f"contextCard={sel.contextCard.id if sel.contextCard else None} "
                          f"effect={sel.effect.id if sel.effect else None} "
                          f"looking={[c.id if c else None for c in (parsed.current.looking or [])]}")
                    if fails >= 6:
                        env.close(); return
            raw = env.select(agents[env.acting_player()](raw))
        env.close()
    print("done, fails =", fails)


if __name__ == "__main__":
    main()
