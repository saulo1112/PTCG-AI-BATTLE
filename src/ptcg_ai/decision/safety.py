"""Crash insurance around any policy.

A submitted agent that raises or returns an illegal selection loses the game
(and rating is win/loss only), so every submitted or long-running policy is
wrapped: exceptions and invalid outputs degrade to a uniform-random legal
choice, and every intervention is logged loudly.
"""

from __future__ import annotations

import logging

from ptcg_ai.decision.base import DecisionContext, Policy
from ptcg_ai.decision.random_policy import RandomPolicy
from ptcg_ai.observation.models import ParsedSelect

logger = logging.getLogger(__name__)


class SafePolicy:
    """Wrap ``inner``; guarantee a legal selection or deck, never an exception."""

    def __init__(
        self,
        inner: Policy,
        deck: list[int] | None = None,
        seed: int | None = None,
    ) -> None:
        self._inner = inner
        self._fallback = RandomPolicy(deck=deck, seed=seed)
        self.name = f"safe({inner.name})"
        self.last_note: str | None = None
        self.interventions = 0

    def choose(self, ctx: DecisionContext) -> list[int]:
        select = ctx.observation.select
        try:
            action = self._inner.choose(ctx)
            self.last_note = self._inner.last_note
            problem = _validate(action, select) if select is not None else None
            if problem is None:
                return action
        except Exception:
            logger.exception("Policy %s raised; falling back to random.", self._inner.name)
            problem = "exception"
        self.interventions += 1
        logger.warning(
            "SafePolicy intervention #%d for %s: %s", self.interventions, self._inner.name, problem
        )
        action = self._fallback.choose(ctx)
        self.last_note = f"SAFETY FALLBACK ({problem}); {self._fallback.last_note}"
        return action

    def choose_deck(self, ctx: DecisionContext) -> list[int]:
        try:
            deck = self._inner.choose_deck(ctx)
            if isinstance(deck, list) and len(deck) == 60 and all(isinstance(c, int) for c in deck):
                return deck
            problem = f"invalid deck ({type(deck).__name__}, len {len(deck) if isinstance(deck, list) else 'n/a'})"
        except Exception:
            logger.exception("Policy %s raised in choose_deck; using fallback deck.", self._inner.name)
            problem = "exception"
        self.interventions += 1
        logger.warning("SafePolicy deck intervention: %s", problem)
        return self._fallback.choose_deck(ctx)

    def on_battle_start(self) -> None:
        self.last_note = None
        self._inner.on_battle_start()

    def on_battle_end(self, outcome: object) -> None:
        self._inner.on_battle_end(outcome)  # type: ignore[arg-type]


def _validate(action: object, select: ParsedSelect) -> str | None:
    """Return a problem description, or ``None`` if ``action`` is legal."""
    if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
        return f"not a list[int]: {action!r}"
    if not (select.minCount <= len(action) <= select.maxCount):
        return f"count {len(action)} outside [{select.minCount}, {select.maxCount}]"
    if len(set(action)) != len(action):
        return f"duplicate indices: {action}"
    n = len(select.option)
    if any(not (0 <= i < n) for i in action):
        return f"index out of range [0, {n}): {action}"
    return None
