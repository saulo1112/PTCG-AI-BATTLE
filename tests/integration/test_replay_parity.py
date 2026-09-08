"""Replay-vs-fixture parity: Kaggle replay observations parse through the same
pipeline as live observations (M7-0 de-risk for imitation learning).

The imitation dataset is built by replaying the 650-elo player's logged POV
observations through ``ObservationParser`` → ``GameState`` → ``resolve_option``.
If a replay observation did not have the exact live-obs shape, the featurizer
would silently train on garbage. These tests assert that (a) every captured
live fixture still parses and builds state, and (b) a sample of real replay
observations parse, build state from both seats, and resolve their options at
the same low unresolved-rate we measured (face-down prizes / hidden zones only).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.observation.resolve import resolve_option
from ptcg_ai.state.game_state import GameState
from tests.conftest import OBSERVATIONS_DIR

pytestmark = pytest.mark.sdk

REPO_ROOT = Path(__file__).parents[2]
_LOGS = REPO_ROOT / "Logs" / "Higher ranking logs"

#: (folder, player) pairs whose replays feed the imitation datasets. Each must
#: parse through the live pipeline identically — the training-serving guarantee.
REPLAY_SOURCES = [
    (_LOGS / "650 elo", "greengreenpurple"),
    (_LOGS / "800 elo", "[RU] Nikita Kuznetsov"),
]


@pytest.fixture(scope="module")
def cards() -> CardDatabase:
    from ptcg_ai.utils.paths import default_sdk_dir

    return CardDatabase.from_sdk(load_sdk(default_sdk_dir()))


@pytest.fixture(scope="module")
def parser() -> ObservationParser:
    return ObservationParser()


def _player_seats(data: dict, name: str) -> list[int]:
    agents = data.get("info", {}).get("Agents", [])
    return [i for i, a in enumerate(agents) if a.get("Name", "").lower() == name.lower()]


def test_all_live_fixtures_parse_and_build(
    observation_fixtures: dict[str, dict], parser: ObservationParser, cards: CardDatabase
) -> None:
    """Every captured live observation still parses and builds GameState."""
    for stem, raw in observation_fixtures.items():
        obs = parser.parse(raw)
        gs = GameState.build(obs, cards)
        assert gs is not None
        # Non-deck, non-terminal fixtures carry a decision to resolve.
        if obs.select is not None and obs.current is not None:
            for opt in obs.select.option:
                resolve_option(opt, obs)  # must not raise


@pytest.mark.parametrize("replay_dir, player", REPLAY_SOURCES)
def test_replay_observations_match_live_pipeline(
    replay_dir: Path, player: str, parser: ObservationParser, cards: CardDatabase
) -> None:
    """Sampled replay decisions parse, build both-seat state, and resolve their
    options with only the expected hidden-zone unresolved rate — for every
    imitation data source (650 and 800), the training-serving parity guarantee."""
    if not replay_dir.is_dir():
        pytest.skip(f"replay dir absent: {replay_dir}")
    files = sorted(replay_dir.glob("*.json"))[:5]
    if not files:
        pytest.skip("no replay files present")

    active_decisions = opt_total = opt_unresolved = 0
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        steps = data.get("steps", [])
        for seat in _player_seats(data, player):
            for i in range(len(steps) - 1):
                cell = steps[i][seat] if seat < len(steps[i]) else None
                if not cell or cell.get("status") != "ACTIVE":
                    continue
                raw = cell.get("observation") or {}
                sel = raw.get("select")
                if not sel or not sel.get("option"):
                    continue
                nxt = steps[i + 1][seat].get("action") if seat < len(steps[i + 1]) else None
                if not isinstance(nxt, list) or len(nxt) == 60:
                    continue  # not an option-index action (deck submission / idle)

                active_decisions += 1
                obs = parser.parse(raw)  # must not raise
                assert obs.select is not None and obs.current is not None
                # yourIndex in the replay POV must equal the acting seat.
                assert obs.current.yourIndex == seat
                GameState.build(obs, cards)
                GameState.build_for(obs, cards, seat)  # both perspectives build
                for opt in obs.select.option:
                    opt_total += 1
                    if not resolve_option(opt, obs).resolved:
                        opt_unresolved += 1

    assert active_decisions > 100, f"too few decisions sampled: {active_decisions}"
    unresolved_rate = opt_unresolved / max(opt_total, 1)
    # Only face-down prizes and hidden opponent zones should fail to resolve.
    assert unresolved_rate < 0.10, f"unresolved rate too high: {unresolved_rate:.2%}"
