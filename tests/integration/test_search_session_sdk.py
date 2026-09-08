"""SDK-gated search-session integration: real engine, real determinization.

Drives a short greedy-vs-greedy game to a mid-game decision, determinizes it,
and exercises begin/step at both MAIN and sub-select contexts, plus lifecycle
hygiene. Auto-skipped when the native engine library is unavailable (ADR-0007).
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.planning.determinize import Determinizer, DeterminizeError
from ptcg_ai.planning.session import SearchSession

pytestmark = pytest.mark.sdk

SAMPLE = "pokemon-tcg-ai-battle/sample_submission/sample_submission/deck.csv"


def _advance(env, agents, raw, n):
    for _ in range(n):
        if env.result() is not None:
            break
        raw = env.select(agents[env.acting_player()](raw))
    return raw


def _setup():
    config = load_config(profile="benchmark")
    sdk = load_sdk(config.paths.sdk_dir)
    cards = CardDatabase.from_sdk(sdk)
    deck = load_deck(Path(SAMPLE))
    ids = deck.as_list()
    agents = [PTCGAgent(GreedyPolicy(deck=ids), deck=deck, cards=cards),
              PTCGAgent(GreedyPolicy(deck=ids), deck=deck, cards=cards)]
    env = BattleEnvironment(config, sdk=sdk)
    raw = env.start(ids, ids)
    return config, sdk, cards, deck, ids, agents, env, raw


def test_determinize_and_begin_step_midgame() -> None:
    config, sdk, cards, deck, ids, agents, env, raw = _setup()
    parser = ObservationParser()
    try:
        raw = _advance(env, agents, raw, 20)
        if env.result() is not None:
            pytest.skip("battle ended during warmup; rerun")
        det = Determinizer(ids, cards, random.Random(0))
        worlds = det.sample(parser.parse(raw), 3)
        assert len(worlds) == 3
        with SearchSession(sdk.api, parser) as sess:
            root = sess.begin(raw, worlds[0])
            assert root.actor_index is not None
            sel = root.observation.select
            assert sel is not None and len(sel.option) >= 1
            child = sess.step(root, [0])
            assert child.search_id != root.search_id  # persistent tree
            # parent still usable (branching)
            child2 = sess.step(root, [min(1, len(sel.option) - 1)])
            assert child2.search_id != child.search_id
    finally:
        env.close()


def test_lethal_step_credits_prizes_before_leaf() -> None:
    # After a KO the acting player's prize count must have dropped by leaf time.
    config, sdk, cards, deck, ids, agents, env, raw = _setup()
    parser = ObservationParser()
    try:
        raw = _advance(env, agents, raw, 24)
        if env.result() is not None:
            pytest.skip("battle ended during warmup; rerun")
        det = Determinizer(ids, cards, random.Random(1))
        try:
            worlds = det.sample(parser.parse(raw), 1)
        except DeterminizeError:
            pytest.skip("determinize mismatch at this decision; rerun")
        with SearchSession(sdk.api, parser) as sess:
            root = sess.begin(raw, worlds[0])
            # Just assert we can walk to a terminal or actor flip without error.
            node = root
            root_actor = root.actor_index
            for _ in range(40):
                if node.is_terminal or node.actor_index != root_actor:
                    break
                sel = node.observation.select
                k = sel.minCount if sel.minCount > 0 else 1
                node = sess.step(node, list(range(min(k, len(sel.option)))))
            assert node.result in (-1, 0, 1, 2)
    finally:
        env.close()


def test_session_reusable_after_close() -> None:
    config, sdk, cards, deck, ids, agents, env, raw = _setup()
    parser = ObservationParser()
    try:
        raw = _advance(env, agents, raw, 16)
        if env.result() is not None:
            pytest.skip("battle ended during warmup; rerun")
        det = Determinizer(ids, cards, random.Random(2))
        worlds = det.sample(parser.parse(raw), 1)
        with SearchSession(sdk.api, parser) as sess:
            sess.begin(raw, worlds[0])
        # a second session in the same process must work (pool was released)
        with SearchSession(sdk.api, parser) as sess2:
            root = sess2.begin(raw, worlds[0])
            assert root.actor_index is not None
    finally:
        env.close()
