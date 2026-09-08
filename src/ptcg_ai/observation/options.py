"""Small helpers for working with decision requests."""

from __future__ import annotations

from typing import Any, Mapping

from ptcg_ai.observation.models import OptionKind, ParsedSelect


def is_first_call(raw: Mapping[str, Any]) -> bool:
    """True for the Kaggle-side initial call where the agent must return its
    60-card deck instead of option indices (``select`` is ``None``)."""
    return raw.get("select") is None


def legal_action_counts(select: ParsedSelect) -> tuple[int, int]:
    """The inclusive ``(minCount, maxCount)`` bounds for a valid selection."""
    return select.minCount, select.maxCount


def group_options(select: ParsedSelect) -> dict[OptionKind, list[int]]:
    """Map each option kind to the option indices of that kind.

    Handy for policies ("is END the only choice?") and for the fixture
    capture's decision-signature sampling.
    """
    groups: dict[OptionKind, list[int]] = {}
    for i, option in enumerate(select.option):
        groups.setdefault(option.type, []).append(i)
    return groups


def decision_signature(select: ParsedSelect) -> str:
    """A short stable key describing the *shape* of a decision.

    Used to sample one fixture per distinct decision kind
    (e.g. ``"MAIN/MAIN:ATTACH+END+PLAY"``).
    """
    kinds = "+".join(sorted({o.type.name for o in select.option}))
    return f"{select.type.name}/{select.context.name}:{kinds}"
