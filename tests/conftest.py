"""Shared test plumbing (ADR-0007).

- ``@pytest.mark.sdk`` tests auto-skip when the native engine library is
  unavailable (checked once per session).
- Observation fixtures are raw dicts captured from a real battle by
  ``ptcg capture-fixtures``; ``make_raw_obs`` builds a minimal synthetic
  observation for tests that shouldn't depend on captured files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ptcg_ai.config import AppConfig
from ptcg_ai.config.schema import BenchConfig, PathsConfig

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
OBSERVATIONS_DIR = FIXTURES_DIR / "observations"
DECKS_DIR = FIXTURES_DIR / "decks"


def _sdk_is_available() -> bool:
    from ptcg_ai.environment.sdk import sdk_available
    from ptcg_ai.utils.paths import default_sdk_dir

    return sdk_available(default_sdk_dir())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _sdk_is_available():
        return
    skip = pytest.mark.skip(reason="native cabt engine library not available")
    for item in items:
        if "sdk" in item.keywords:
            item.add_marker(skip)


@pytest.fixture()
def app_config(tmp_path: Path) -> AppConfig:
    """Default config with disposable output dirs (SDK paths untouched)."""
    defaults = PathsConfig()
    return AppConfig(
        paths=PathsConfig(
            sdk_dir=defaults.sdk_dir,
            deck_path=defaults.deck_path,
            opponent_deck_path=defaults.opponent_deck_path,
            build_dir=tmp_path / "build",
            replay_dir=tmp_path / "replays",
        ),
        bench=BenchConfig(output_dir=tmp_path / "bench"),
    )


@pytest.fixture()
def observation_fixtures() -> dict[str, dict[str, Any]]:
    """All captured raw observations, keyed by fixture filename stem."""
    if not OBSERVATIONS_DIR.is_dir():
        pytest.skip("observation fixtures not captured yet (run `ptcg capture-fixtures`)")
    fixtures = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(OBSERVATIONS_DIR.glob("*.json"))
    }
    if not fixtures:
        pytest.skip("observation fixtures directory is empty")
    return fixtures


def make_raw_obs(
    *,
    n_options: int = 4,
    min_count: int = 1,
    max_count: int = 1,
    select_type: int = 0,
    context: int = 0,
    extra_select_keys: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A minimal synthetic main-phase-shaped observation."""
    options: list[dict[str, Any]] = [
        {"type": 7, "index": i} for i in range(max(0, n_options - 1))  # PLAY
    ]
    options.append({"type": 14})  # END
    select: dict[str, Any] = {
        "type": select_type,
        "context": context,
        "minCount": min_count,
        "maxCount": max_count,
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "option": options[:n_options],
        "deck": None,
        "contextCard": None,
        "effect": None,
    }
    if extra_select_keys:
        select.update(extra_select_keys)
    player = {
        "active": [
            {
                "id": 721, "serial": 1, "hp": 60, "maxHp": 60, "appearThisTurn": False,
                "energies": [3], "energyCards": [], "tools": [], "preEvolution": [],
            }
        ],
        "bench": [],
        "benchMax": 5,
        "deckCount": 40,
        "discard": [],
        "prize": [None] * 6,
        "handCount": 5,
        "hand": [{"id": 3, "serial": 2, "playerIndex": 0}] * 5,
        "poisoned": False, "burned": False, "asleep": False,
        "paralyzed": False, "confused": False,
    }
    opponent = dict(player, hand=None, handCount=4)
    return {
        "select": select,
        "logs": [{"type": 2, "playerIndex": 0}],
        "current": {
            "turn": 3,
            "turnActionCount": 0,
            "yourIndex": 0,
            "firstPlayer": 0,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "result": -1,
            "stadium": [],
            "looking": None,
            "players": [player, opponent],
        },
        "search_begin_input": "AAAA",
    }
