"""Policy interface and decision context (ADR-0004).

Every decision the engine can ask for — deck submission, mulligan, setup,
main phase, mid-effect targets, coin calls — flows through the same
``choose`` entry point. Specialization by decision kind happens *inside*
policies, never in the agent loop.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

from ptcg_ai.observation.models import ParsedObservation

if TYPE_CHECKING:
    from ptcg_ai.cards.database import CardDatabase
    from ptcg_ai.environment.adapter import BattleOutcome


@dataclass(frozen=True)
class DecisionContext:
    """Everything a policy may consult for one decision.

    Attributes:
        raw: The observation exactly as delivered (needed e.g. for the SDK
            search API, which requires the untouched dict).
        observation: The parsed, forward-compatible view.
        cards: Static card knowledge; ``None`` when running without the SDK.
    """

    raw: Mapping[str, Any]
    observation: ParsedObservation
    cards: "CardDatabase | None" = None

    @property
    def is_deck_submission(self) -> bool:
        return self.observation.select is None


@runtime_checkable
class Policy(Protocol):
    """A decision engine. Implementations must be safe to reuse across games.

    ``last_note`` is the observability hook (ADR-0009): after ``choose``, a
    policy may leave a short human-readable explanation of *why* it chose.
    """

    name: str
    last_note: str | None

    def choose(self, ctx: DecisionContext) -> list[int]:
        """Return option indices into ``ctx.observation.select.option``.

        The list length must lie in ``[minCount, maxCount]`` with unique
        elements in ``[0, len(option))``.
        """
        ...

    def choose_deck(self, ctx: DecisionContext) -> list[int]:
        """Return the 60 card IDs for the initial deck submission."""
        ...

    def on_battle_start(self) -> None:
        """Reset any per-battle state."""
        ...

    def on_battle_end(self, outcome: "BattleOutcome") -> None:
        """Observe the terminal outcome (for stats or future learning)."""
        ...


class BasePolicy(abc.ABC):
    """Convenience base with no-op hooks and deck plumbing.

    Policies are not required to inherit from this (the Protocol is the
    contract), but most will.
    """

    name: str = "base"

    def __init__(self, deck: list[int] | None = None) -> None:
        self._deck = deck
        self.last_note: str | None = None

    @abc.abstractmethod
    def choose(self, ctx: DecisionContext) -> list[int]: ...

    def choose_deck(self, ctx: DecisionContext) -> list[int]:
        if self._deck is None:
            raise RuntimeError(f"Policy {self.name!r} was not given a deck.")
        return list(self._deck)

    def on_battle_start(self) -> None:
        self.last_note = None

    def on_battle_end(self, outcome: "BattleOutcome") -> None:
        pass
