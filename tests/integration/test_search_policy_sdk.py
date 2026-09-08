"""SDK-gated end-to-end SearchPolicy: real engine, real determinization.

Auto-skipped when the native engine is unavailable (ADR-0007).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.decision.safety import SafePolicy
from ptcg_ai.decision.search import SearchConfig, SearchPolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.sdk import load_sdk

pytestmark = pytest.mark.sdk

SAMPLE = "pokemon-tcg-ai-battle/sample_submission/sample_submission/deck.csv"


def _fast_cfg() -> SearchConfig:
    # small budget shape so the test is quick
    return SearchConfig(determinizations=2, max_candidates=8, rollout_step_cap=20)


def test_search_returns_legal_choices_in_a_real_battle() -> None:
    config = load_config(profile="benchmark")
    sdk = load_sdk(config.paths.sdk_dir)
    cards = CardDatabase.from_sdk(sdk)
    deck = load_deck(Path(SAMPLE))
    ids = deck.as_list()

    search = SafePolicy(
        SearchPolicy(deck=ids, cards=cards, api=sdk.api, search_cfg=_fast_cfg(), rng_seed=0),
        deck=ids, seed=0,
    )
    greedy = SafePolicy(GreedyPolicy(deck=ids), deck=ids, seed=1)
    a = PTCGAgent(search, deck=deck, cards=cards)
    b = PTCGAgent(greedy, deck=deck, cards=cards)

    env = BattleEnvironment(config, sdk=sdk)
    raw = env.start(ids, ids)
    search.on_battle_start()
    greedy.on_battle_start()
    decisions = 0
    try:
        for _ in range(120):
            if env.result() is not None:
                break
            raw = env.select(a(raw) if env.acting_player() == 0 else b(raw))
            decisions += 1
    finally:
        env.close()

    assert decisions > 0
    # SafePolicy must never have had to intervene on a legal-index violation.
    assert search.interventions == 0
    inner = search._inner  # the wrapped SearchPolicy
    assert isinstance(inner, SearchPolicy)
    # It actually searched at least some decisions (not all fallbacks).
    assert inner.searched > 0
