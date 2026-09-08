"""Validate the event-counting convention on full recorded games.

Methodology guard (docs/methodology.md): the engine delivers each event to
BOTH viewers (public events duplicated; hidden ones as ``*_REVERSE``), so
global event counts must come from ONE viewer's stream. The complete,
non-overlapping stream is: logs of the steps where ``player == v`` plus the
terminal observation's logs, where ``v`` is the terminal observation's
``current.yourIndex``.

If this test starts failing after an SDK update, the whole analytics
methodology needs re-validation before trusting any report.
"""

import pytest

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.config import AppConfig
from ptcg_ai.decision.random_policy import RandomPolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.runner import BattleRunner
from ptcg_ai.replay.episode import Episode
from ptcg_ai.replay.recorder import EpisodeRecorder

pytestmark = pytest.mark.sdk

_TURN_START = 2  # LogType.TURN_START


def _record_full_game(app_config: AppConfig, seed: int) -> Episode:
    deck = load_deck(app_config.paths.deck_path)
    env = BattleEnvironment(app_config)
    runner = BattleRunner(env)
    record = runner.run(
        PTCGAgent(RandomPolicy(seed=seed)),
        PTCGAgent(RandomPolicy(seed=seed + 1)),
        deck.as_list(),
        deck.as_list(),
        recorder=EpisodeRecorder(),
    )
    assert record.episode is not None
    return record.episode


def _viewer_stream_counts(episode: Episode, viewer: int, log_type: int) -> int:
    count = 0
    for step in episode.steps:
        if step.player == viewer:
            count += sum(1 for log in step.raw_obs.get("logs", []) if log.get("type") == log_type)
    final = episode.final_obs
    assert final is not None
    if final["current"]["yourIndex"] == viewer:
        count += sum(1 for log in final.get("logs", []) if log.get("type") == log_type)
    return count


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_turn_start_parity_between_viewer_streams(app_config: AppConfig, seed: int) -> None:
    """Both viewers see (nearly) every public event exactly once.

    The tail tolerance exists because events after a player's last decision
    reach them only if they receive the terminal observation.
    """
    episode = _record_full_game(app_config, seed)
    c0 = _viewer_stream_counts(episode, 0, _TURN_START)
    c1 = _viewer_stream_counts(episode, 1, _TURN_START)
    assert abs(c0 - c1) <= 2, f"viewer streams disagree wildly: P0={c0}, P1={c1}"

    # The counting viewer's stream (terminal viewer) must be the complete one.
    assert episode.final_obs is not None
    terminal_viewer = episode.final_obs["current"]["yourIndex"]
    counting = c0 if terminal_viewer == 0 else c1
    other = c1 if terminal_viewer == 0 else c0
    assert counting >= other, "terminal viewer's stream should never miss tail events"

    # Turn count sanity: counting-stream TURN_STARTs match the final turn number.
    final_turn = episode.final_obs["current"]["turn"]
    assert counting == final_turn, (
        f"TURN_START count ({counting}) should equal final turn ({final_turn})"
    )


def test_terminal_observation_recorded(app_config: AppConfig) -> None:
    episode = _record_full_game(app_config, seed=404)
    assert episode.final_obs is not None
    assert episode.final_obs["current"]["result"] != -1
    assert episode.outcome is not None
    assert episode.outcome["duration_s"] > 0
    assert all(step.elapsed_ms is not None for step in episode.steps)
