"""Incremental episode recording during a battle (format v2, ADR-0011)."""

from __future__ import annotations

from typing import Any, Mapping

from ptcg_ai.replay.episode import Episode, Step


class EpisodeRecorder:
    """Collects steps during a battle; ``finalize`` yields the Episode."""

    def __init__(self, metadata: Mapping[str, Any] | None = None) -> None:
        self._episode = Episode(metadata=dict(metadata or {}))
        self._finalized = False

    def on_step(
        self,
        raw_obs: Mapping[str, Any],
        action: list[int],
        player: int,
        elapsed_ms: float | None = None,
    ) -> None:
        self._require_open()
        self._episode.steps.append(
            Step(raw_obs=raw_obs, action=list(action), player=player, elapsed_ms=elapsed_ms)
        )

    def set_final(self, raw_obs: Mapping[str, Any]) -> None:
        """Record the terminal observation (not a decision — stale select)."""
        self._require_open()
        self._episode.final_obs = raw_obs

    def finalize(self, outcome: Mapping[str, Any] | None) -> Episode:
        self._finalized = True
        self._episode.outcome = outcome
        return self._episode

    def _require_open(self) -> None:
        if self._finalized:
            raise RuntimeError("Recorder already finalized.")
