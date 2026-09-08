"""M6-0 Probe A: can search_begin start from a MID-EFFECT sub-select?

The all-contexts receding-horizon architecture (plan M6) assumes search can
begin at any live decision, not just MAIN. This drives greedy-vs-greedy games
(greedy plays search Items -> TO_HAND selects occur) and, at every distinct
select context encountered, attempts search_begin + one search_step + a short
random continuation. Reports per-context success/failure.
"""

from __future__ import annotations

import collections
import dataclasses
import random
import sys
from pathlib import Path
from typing import Any

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.decision.safety import SafePolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.observation.parser import ObservationParser

SAMPLE = "pokemon-tcg-ai-battle/sample_submission/sample_submission/deck.csv"


def filler_hidden_info(raw: dict[str, Any], deck_ids: list[int], cards: CardDatabase):
    current = raw["current"]
    me = current["yourIndex"]
    mine = current["players"][me]
    opp = current["players"][1 - me]
    ids = deck_ids

    your_deck = ids[: max(mine["deckCount"], 1)]
    your_prize = ids[: len(mine["prize"])]
    opponent_deck = ids[: max(opp["deckCount"], 1)]
    opponent_prize = ids[: len(opp["prize"])]
    opponent_hand = ids[: opp["handCount"]]
    active = opp.get("active") or []
    opponent_active: list[int] = []
    if active and active[0] is None:
        basics = [c for c in ids if (i := cards.get_card(c)) and i.is_basic_pokemon]
        opponent_active = [basics[0]]
    return (your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active)


def probe_begin(api, raw: dict[str, Any], deck_ids: list[int], cards: CardDatabase,
                rng: random.Random) -> tuple[str, str]:
    """Try begin+step+5 random continuation steps. Returns (status, detail)."""
    try:
        agent_obs = api.to_observation_class(raw)
        fillers = filler_hidden_info(raw, deck_ids, cards)
        root = api.search_begin(agent_obs, *fillers, manual_coin=False)
    except Exception as e:  # noqa: BLE001 - probe wants every failure mode
        return ("BEGIN-FAIL", f"{type(e).__name__}: {e}")
    try:
        node = root
        contexts = []
        for _ in range(6):
            sel = node.observation.select
            cur = node.observation.current
            if sel is None or (cur is not None and cur.result != -1):
                contexts.append("TERMINAL")
                break
            contexts.append(f"{sel.type}/{sel.context}n{len(sel.option)}")
            k = rng.randint(sel.minCount, sel.maxCount)
            choice = rng.sample(range(len(sel.option)), k)
            node = api.search_step(node.searchId, choice)
        return ("OK", " -> ".join(contexts))
    except Exception as e:  # noqa: BLE001
        return ("STEP-FAIL", f"{type(e).__name__}: {e}")
    finally:
        api.search_end()


def main() -> None:
    config = load_config(profile="benchmark")
    sdk = load_sdk(config.paths.sdk_dir)
    api = sdk.api
    cards = CardDatabase.from_sdk(sdk)
    deck = load_deck(Path(SAMPLE))
    deck_ids = deck.as_list()
    parser = ObservationParser()
    rng = random.Random(11)

    # context signature -> Counter of probe status
    results: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    details: dict[str, str] = {}
    probed_main = 0

    for game in range(3):
        pols = [SafePolicy(GreedyPolicy(deck=deck_ids), deck=deck_ids, seed=100 + game),
                SafePolicy(GreedyPolicy(deck=deck_ids), deck=deck_ids, seed=200 + game)]
        agents = [PTCGAgent(pols[0], deck=deck, cards=cards),
                  PTCGAgent(pols[1], deck=deck, cards=cards)]
        env = BattleEnvironment(config, sdk=sdk)
        raw = env.start(deck_ids, deck_ids)
        for _ in range(400):
            if env.result() is not None:
                break
            parsed = parser.parse(raw)
            sel = parsed.select
            if sel is not None:
                sig = f"{sel.type.name}/{sel.context.name}"
                has_deck_view = raw.get("select", {}).get("deck") is not None
                if has_deck_view:
                    sig += "+deckview"
                is_main = sig.startswith("MAIN/")
                # probe every non-MAIN context each time we see it (max 3 per sig),
                # and MAIN twice as a control
                want = (not is_main and results[sig].total() < 3) or (is_main and probed_main < 2)
                if want:
                    status, detail = probe_begin(api, raw, deck_ids, cards, rng)
                    results[sig][status] += 1
                    details.setdefault(f"{sig}|{status}", detail)
                    if is_main:
                        probed_main += 1
            raw = env.select(agents[env.acting_player()](raw))
        env.close()

    print("=== Probe A: search_begin at each live select context ===")
    for sig in sorted(results):
        counts = dict(results[sig])
        print(f"{sig:45s} {counts}")
        for status in counts:
            print(f"    [{status}] {details.get(f'{sig}|{status}', '')}")


if __name__ == "__main__":
    main()
