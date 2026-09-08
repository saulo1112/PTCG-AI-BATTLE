"""The agent façade implementing the competition contract.

``PTCGAgent`` is a callable matching Kaggle's expectation:
``agent(obs_dict) -> list[int]`` — a deck (card IDs) on the initial call,
option indices afterwards. It is pure glue: parse, build context, delegate
to the policy.
"""

from __future__ import annotations

from typing import Any, Mapping

from ptcg_ai.agent.deck import Deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.decision.base import DecisionContext, Policy
from ptcg_ai.observation.options import is_first_call
from ptcg_ai.observation.parser import ObservationParser


class PTCGAgent:
    """Bind a policy (and optional card knowledge) into a Kaggle-shaped callable."""

    def __init__(
        self,
        policy: Policy,
        deck: Deck | None = None,
        cards: CardDatabase | None = None,
        parser: ObservationParser | None = None,
    ) -> None:
        self._policy = policy
        self._deck = deck
        self._cards = cards
        self._parser = parser if parser is not None else ObservationParser()

    @property
    def policy(self) -> Policy:
        return self._policy

    def __call__(self, raw: Mapping[str, Any]) -> list[int]:
        ctx = DecisionContext(
            raw=raw,
            observation=self._parser.parse(raw),
            cards=self._cards,
        )
        if is_first_call(raw):
            if self._deck is not None:
                return self._deck.as_list()
            return self._policy.choose_deck(ctx)
        return self._policy.choose(ctx)
