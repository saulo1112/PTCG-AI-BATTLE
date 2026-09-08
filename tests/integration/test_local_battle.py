"""Full random-vs-random battles through the whole stack (needs the engine)."""

import pytest

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.config import AppConfig
from ptcg_ai.debug.trace import DecisionTracer, TracingPolicy
from ptcg_ai.decision.random_policy import RandomPolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.runner import BattleRunner
from ptcg_ai.replay.recorder import EpisodeRecorder

pytestmark = pytest.mark.sdk


def test_full_battle_terminates_with_valid_outcome(app_config: AppConfig) -> None:
    deck = load_deck(app_config.paths.deck_path)
    tracer = DecisionTracer()
    agent0 = PTCGAgent(TracingPolicy(RandomPolicy(seed=11), tracer))
    agent1 = PTCGAgent(RandomPolicy(seed=22))

    env = BattleEnvironment(app_config)
    runner = BattleRunner(env, max_decisions=app_config.battle.max_decisions)
    recorder = EpisodeRecorder({"test": "integration"})
    record = runner.run(agent0, agent1, deck.as_list(), deck.as_list(), recorder)

    assert record.outcome.result in (0, 1, 2)
    assert record.outcome.winner in (0, 1, None)
    assert record.outcome.reason in (1, 2, 3, 4, None)
    assert record.decisions > 0

    # Episode captured every decision of both players.
    assert record.episode is not None
    assert len(record.episode.steps) == record.decisions
    assert record.episode.outcome is not None

    # Tracing captured player 0's decisions with latency data.
    assert 0 < len(tracer.decisions) <= record.decisions
    assert all(d.elapsed_ms >= 0 for d in tracer.decisions)


def test_environment_is_reusable_between_battles(app_config: AppConfig) -> None:
    deck = load_deck(app_config.paths.deck_path)
    env = BattleEnvironment(app_config)
    runner = BattleRunner(env)
    for seed in (1, 2):
        record = runner.run(
            PTCGAgent(RandomPolicy(seed=seed)),
            PTCGAgent(RandomPolicy(seed=seed + 100)),
            deck.as_list(),
            deck.as_list(),
        )
        assert record.outcome.result in (0, 1, 2)
