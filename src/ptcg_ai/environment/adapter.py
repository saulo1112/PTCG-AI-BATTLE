"""Typed wrapper around the SDK's battle host loop.

``BattleEnvironment`` owns one live battle (the SDK allows only one per
process) and exposes the ``start → select → ... → result`` cycle with typed
errors and outcome extraction. Consumers exchange raw observation dicts;
parsing is the job of :mod:`ptcg_ai.observation`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from ptcg_ai.config.schema import AppConfig, DECK_SIZE
from ptcg_ai.environment.sdk import SdkModules, load_sdk

logger = logging.getLogger(__name__)

#: A raw observation exactly as produced by the engine (JSON-decoded dict).
RawObs = dict[str, Any]

#: ``LogType.RESULT`` — the engine log entry carrying the finish reason.
_LOG_TYPE_RESULT = 23

#: Engine deck-validation errors (from the engine's BattleStart).
_START_ERRORS = {
    1: "deck contains an unknown card ID",
    2: "more than 4 copies of a same-name card (non basic Energy)",
    3: "deck contains no Basic Pokémon",
    4: "more than one ACE SPEC card",
}

#: ``State.result`` value meaning the battle is still running.
RESULT_ONGOING = -1
#: ``State.result`` value meaning a draw.
RESULT_DRAW = 2


class BattleStartError(RuntimeError):
    """The engine rejected a deck at battle start."""

    def __init__(self, error_player: int, error_type: int) -> None:
        reason = _START_ERRORS.get(error_type, f"unknown error type {error_type}")
        super().__init__(f"BattleStart failed for player {error_player}: {reason}")
        self.error_player = error_player
        self.error_type = error_type


@dataclass(frozen=True)
class BattleOutcome:
    """Terminal result of a battle.

    Attributes:
        result: Raw ``State.result`` (0/1 = winner index, 2 = draw).
        winner: Winning player index, or ``None`` on a draw.
        reason: Engine finish reason if reported (1 = prizes taken,
            2 = deck-out, 3 = no Active Pokémon, 4 = card effect).
    """

    result: int
    winner: int | None
    reason: int | None

    @property
    def is_draw(self) -> bool:
        return self.winner is None


class BattleEnvironment:
    """One local battle hosted through the vendored SDK.

    Usage::

        with BattleEnvironment(config) as env:
            obs = env.start(deck0, deck1)
            while env.result() is None:
                obs = env.select(agent(obs))
            outcome = env.result()
    """

    def __init__(self, config: AppConfig | None = None, sdk: SdkModules | None = None) -> None:
        self._config = config if config is not None else AppConfig()
        self._sdk = sdk if sdk is not None else load_sdk(self._config.paths.sdk_dir)
        self._active = False
        self._last_obs: RawObs | None = None

    def start(self, deck0: Sequence[int], deck1: Sequence[int]) -> RawObs:
        """Start a battle and return the first observation.

        Note: in local hosting the decks are supplied here directly — the
        Kaggle-side "``select is None`` → return your deck" call never occurs.
        """
        if self._active:
            raise RuntimeError("A battle is already active; close() it first.")
        for i, deck in enumerate((deck0, deck1)):
            if len(deck) != DECK_SIZE:
                raise ValueError(f"deck{i} has {len(deck)} cards; must be {DECK_SIZE}")

        obs, start_data = self._sdk.game.battle_start(list(deck0), list(deck1))
        if obs is None:
            raise BattleStartError(start_data.errorPlayer, start_data.errorType)
        self._active = True
        self._last_obs = obs
        return obs

    def select(self, action: Sequence[int]) -> RawObs:
        """Submit the acting player's chosen option indices; return next obs."""
        self._require_active()
        obs = self._sdk.game.battle_select(list(action))
        self._last_obs = obs
        return obs

    @property
    def last_obs(self) -> RawObs:
        self._require_active()
        assert self._last_obs is not None
        return self._last_obs

    def acting_player(self) -> int:
        """Player index (0/1) whose decision the current observation requests."""
        return int(self.last_obs["current"]["yourIndex"])

    def result(self) -> BattleOutcome | None:
        """Terminal outcome, or ``None`` while the battle is still running."""
        current = self.last_obs.get("current")
        if current is None:
            return None
        result = int(current.get("result", RESULT_ONGOING))
        if result == RESULT_ONGOING:
            return None
        winner = result if result in (0, 1) else None
        return BattleOutcome(result=result, winner=winner, reason=self._finish_reason())

    def visualize_json(self) -> str:
        """The engine's accumulated visualizer data (JSON string)."""
        self._require_active()
        return self._sdk.game.visualize_data()

    def close(self) -> None:
        """Release the engine-side battle memory (idempotent)."""
        if self._active:
            self._sdk.game.battle_finish()
            self._active = False
            self._last_obs = None

    def _finish_reason(self) -> int | None:
        assert self._last_obs is not None
        for log in reversed(self._last_obs.get("logs", [])):
            if log.get("type") == _LOG_TYPE_RESULT:
                return log.get("reason")
        return None

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("No active battle; call start() first.")

    def __enter__(self) -> "BattleEnvironment":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
