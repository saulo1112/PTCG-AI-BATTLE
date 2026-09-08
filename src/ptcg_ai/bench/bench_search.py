"""SDK Search API throughput — THE measurement gating ADR-0010 (ADR-0006).

Method: play a real battle to a mid-game decision point, then determinize
with *filler* hidden information (slices of the known deck lists — validity
matters here, truth does not) and random-walk the search tree, timing
``search_begin`` and ``search_step``. The walk restarts from the root when a
line reaches a terminal state; the persistent node tree makes that free.

Caveats recorded with the result:
- Native tree memory is not visible to Python; observe process RSS
  externally for memory limits (research question Q8).
- Random-walk lines are longer than a real search's mostly-shallow
  expansions; treat nodes/s as an optimistic upper bound for deep lines and
  a lower bound for wide-and-shallow expansion patterns.
"""

from __future__ import annotations

import time
from typing import Any

from ptcg_ai.agent.deck import Deck, load_deck
from ptcg_ai.bench.harness import BenchResult
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config.schema import AppConfig
from ptcg_ai.decision.random_policy import RandomPolicy
from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.environment.adapter import BattleEnvironment, RawObs
from ptcg_ai.environment.sdk import SdkModules, load_sdk
from ptcg_ai.utils.seeding import make_rng

#: Decisions to play before benchmarking, to reach a representative mid-game.
_WARMUP_DECISIONS = 30


def run_search_bench(config: AppConfig, n_steps: int | None = None) -> list[BenchResult]:
    """Measure ``search_begin`` latency and ``search_step`` throughput."""
    n_steps = n_steps if n_steps is not None else config.bench.search_steps
    sdk = load_sdk(config.paths.sdk_dir)
    deck = load_deck(config.paths.deck_path)
    cards = CardDatabase.from_sdk(sdk)
    rng = make_rng(7)

    env = BattleEnvironment(config, sdk=sdk)
    raw = _advance_to_midgame(env, deck, seed=7)
    try:
        agent_obs = sdk.api.to_observation_class(raw)
        fillers = _filler_hidden_info(raw, deck, cards)

        # -- search_begin latency (fresh determinization each time) --------
        begin_durations: list[float] = []
        root = None
        for i in range(10):
            if i > 0:
                sdk.api.search_end()  # reset node pool between begins
            start = time.perf_counter()
            root = sdk.api.search_begin(agent_obs, *fillers, manual_coin=False)
            begin_durations.append(time.perf_counter() - start)
        assert root is not None

        # -- search_step throughput (random walk, restart at terminal) -----
        step_durations: list[float] = []
        terminals = 0
        node = root
        for _ in range(n_steps):
            select = node.observation.select
            current = node.observation.current
            if select is None or (current is not None and current.result != -1):
                terminals += 1
                node = root
                continue
            k = rng.randint(select.minCount, select.maxCount)
            choice = rng.sample(range(len(select.option)), k)
            start = time.perf_counter()
            node = sdk.api.search_step(node.searchId, choice)
            step_durations.append(time.perf_counter() - start)
        sdk.api.search_end()
    finally:
        env.close()

    return [
        BenchResult.from_durations(
            "search_begin (mid-game determinization)", begin_durations
        ),
        BenchResult.from_durations(
            "search_step (random walk)",
            step_durations,
            notes={"terminal restarts": terminals, "warmup decisions": _WARMUP_DECISIONS},
        ),
    ]


def _advance_to_midgame(env: BattleEnvironment, deck: Deck, seed: int) -> RawObs:
    """Play random-vs-random for a fixed number of decisions; return the
    observation at the resulting decision point (battle left open).

    Random games occasionally end inside the warmup window, so retry with
    fresh seeds (each attempt is a new battle).
    """
    for attempt in range(10):
        base = seed + 1000 * attempt
        agents = (PTCGAgent(RandomPolicy(seed=base)), PTCGAgent(RandomPolicy(seed=base + 1)))
        obs = env.start(deck.as_list(), deck.as_list())
        for _ in range(_WARMUP_DECISIONS):
            if env.result() is not None:
                break
            obs = env.select(agents[env.acting_player()](obs))
        if env.result() is None:
            return obs
        env.close()  # battle ended early; try another seed
    raise RuntimeError("No warmup battle survived long enough; check the deck/engine.")


def _filler_hidden_info(
    raw: RawObs, deck: Deck, cards: CardDatabase
) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
    """Mechanically valid (not truthful) hidden-info guesses for search_begin.

    Both players use the sample deck here, so slices of the deck list satisfy
    the engine's ID validation and count requirements; a real Determinizer
    (Phase 2+) will replace this with belief-based sampling.
    """
    current: dict[str, Any] = raw["current"]
    me = current["yourIndex"]
    opp = current["players"][1 - me]
    mine = current["players"][me]
    ids = deck.as_list()

    your_deck = ids[: max(mine["deckCount"], 1)]
    your_prize = ids[: len(mine["prize"])]
    opponent_deck = ids[: max(opp["deckCount"], 1)]
    opponent_prize = ids[: len(opp["prize"])]
    opponent_hand = ids[: opp["handCount"]]
    active = opp.get("active") or []
    opponent_active: list[int] = []
    if active and active[0] is None:  # face-down active must be a Pokémon guess
        basics = [cid for cid in ids if (info := cards.get_card(cid)) and info.is_basic_pokemon]
        if not basics:
            raise RuntimeError("Deck has no Basic Pokémon to guess as opponent active.")
        opponent_active = [basics[0]]
    return (your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active)
