"""Uniform-random baseline (rung 1 of the ladder).

Intentionally the only "intelligence" in Phase 0: it exercises every
decision kind, provides the arena floor to beat, and doubles as the safety
fallback inside :class:`~ptcg_ai.decision.safety.SafePolicy`.
"""

from __future__ import annotations

from ptcg_ai.decision.base import BasePolicy, DecisionContext
from ptcg_ai.utils.seeding import make_rng


class RandomPolicy(BasePolicy):
    """Picks a uniformly random legal selection at every decision."""

    name = "random"

    def __init__(self, deck: list[int] | None = None, seed: int | None = None) -> None:
        super().__init__(deck=deck)
        self._rng = make_rng(seed)

    def choose(self, ctx: DecisionContext) -> list[int]:
        select = ctx.observation.select
        if select is None:
            raise ValueError("choose() called on a deck-submission observation")
        n = len(select.option)
        # Clamp: terminal observations carry a stale select whose counts can
        # exceed the (empty) option list — see docs/battle_flow.md.
        k = self._rng.randint(min(select.minCount, n), min(select.maxCount, n))
        chosen = self._rng.sample(range(n), k)
        self.last_note = f"uniform random: {k} of {n} options"
        return chosen
