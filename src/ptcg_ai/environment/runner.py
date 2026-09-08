"""Drive one full local battle between two agent callables.

The engine returns ONE observation per step, built for whichever player must
act next (``current.yourIndex``); the runner routes it to that agent. Note
that in local hosting decks are passed to :meth:`BattleEnvironment.start`
directly — agents' deck-submission path is exercised only under the Kaggle
harness (see docs/battle_flow.md).

Per ADR-0011 the runner is also the instrumentation point: it measures each
agent call (both players), records the terminal observation, and stamps the
battle's wall-clock duration into the outcome.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from ptcg_ai.environment.adapter import BattleEnvironment, BattleOutcome, RawObs
from ptcg_ai.replay.episode import Episode
from ptcg_ai.replay.recorder import EpisodeRecorder

logger = logging.getLogger(__name__)

#: The Kaggle-shaped agent contract: raw observation → option indices.
AgentCallable = Callable[[RawObs], "list[int]"]

#: Optional per-decision hook: (raw_obs, action, player, elapsed_ms).
DecisionHook = Callable[[RawObs, "list[int]", int, float], None]


@dataclass(frozen=True)
class BattleRecord:
    """Result of one hosted battle."""

    outcome: BattleOutcome
    decisions: int
    duration_s: float
    episode: Episode | None

    def outcome_dict(self) -> Mapping[str, object]:
        data = dict(dataclasses.asdict(self.outcome))
        data["duration_s"] = self.duration_s
        return data


class BattleRunner:
    """Runs battles on a :class:`BattleEnvironment` (one at a time)."""

    def __init__(self, env: BattleEnvironment, max_decisions: int = 20_000) -> None:
        self._env = env
        self._max_decisions = max_decisions

    def run(
        self,
        agent0: AgentCallable,
        agent1: AgentCallable,
        deck0: Sequence[int],
        deck1: Sequence[int],
        recorder: EpisodeRecorder | None = None,
        on_decision: DecisionHook | None = None,
    ) -> BattleRecord:
        """Play one battle to termination and return its record."""
        agents = (agent0, agent1)
        decisions = 0
        started = time.perf_counter()

        obs = self._env.start(deck0, deck1)
        try:
            while (outcome := self._env.result()) is None:
                if decisions >= self._max_decisions:
                    raise RuntimeError(
                        f"Battle exceeded {self._max_decisions} decisions; aborting."
                    )
                player = self._env.acting_player()
                t0 = time.perf_counter()
                action = agents[player](obs)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if recorder is not None:
                    recorder.on_step(obs, action, player, elapsed_ms=elapsed_ms)
                if on_decision is not None:
                    on_decision(obs, action, player, elapsed_ms)
                obs = self._env.select(action)
                decisions += 1
            # Loop exited: `obs` is the terminal observation (result != -1).
            if recorder is not None:
                recorder.set_final(obs)
        finally:
            self._env.close()

        duration_s = time.perf_counter() - started
        logger.debug(
            "Battle finished after %d decisions in %.2fs: winner=%s reason=%s",
            decisions, duration_s, outcome.winner, outcome.reason,
        )
        episode = None
        if recorder is not None:
            outcome_data = dict(dataclasses.asdict(outcome))
            outcome_data["duration_s"] = duration_s
            episode = recorder.finalize(outcome_data)
        return BattleRecord(
            outcome=outcome, decisions=decisions, duration_s=duration_s, episode=episode
        )
