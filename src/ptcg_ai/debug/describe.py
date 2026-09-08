"""Human sentences for resolved actions (ADR-0009).

Consumes the *structured* resolution from
:mod:`ptcg_ai.observation.resolve` — analytics never parses these strings;
they exist purely for humans (``ptcg battle --verbose``, ``ptcg
show-episode``).
"""

from __future__ import annotations

from typing import Any, Mapping

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.debug.inspect import format_observation
from ptcg_ai.observation.models import OptionKind, ParsedObservation
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.observation.resolve import ResolvedAction, resolve_option


def describe_action(resolved: ResolvedAction, cards: CardDatabase | None = None) -> str:
    """One action as a short sentence; never raises."""
    name = _name(cards, resolved.card_id)
    target = _name(cards, resolved.target_card_id)
    kind = resolved.kind

    if kind is OptionKind.END:
        return "End turn"
    if kind is OptionKind.YES:
        return "Yes"
    if kind is OptionKind.NO:
        return "No"
    if kind is OptionKind.NUMBER:
        return f"Choose {resolved.number}"
    if kind is OptionKind.ATTACK:
        return _describe_attack(resolved.attack_id, cards)
    if kind is OptionKind.RETREAT:
        return f"Retreat {name}"
    if kind is OptionKind.PLAY:
        return f"Play {name} from hand"
    if kind is OptionKind.ATTACH:
        return f"Attach {name} to {target}"
    if kind is OptionKind.EVOLVE:
        return f"Evolve {target} into {name}"
    if kind is OptionKind.ABILITY:
        return f"Use ability of {name}"
    if kind is OptionKind.DISCARD:
        return f"Discard {name}"
    if kind is OptionKind.CARD:
        area = resolved.source_area.name.lower() if resolved.source_area else "?"
        return f"Select {name} ({area})"
    if kind is OptionKind.ENERGY_CARD or kind is OptionKind.ENERGY:
        return f"Select energy {name} on {target}"
    if kind is OptionKind.TOOL_CARD:
        return f"Select tool {name} on {target}"
    if kind is OptionKind.SKILL:
        return f"Resolve effect of {name}"
    if kind is OptionKind.SPECIAL_CONDITION:
        condition = resolved.special_condition
        return f"Condition: {condition.name if condition is not None else '?'}"
    return f"{kind.name} (unresolved)"


def render_decision(
    raw_obs: Mapping[str, Any],
    action: list[int] | None,
    elapsed_ms: float | None = None,
    cards: CardDatabase | None = None,
    parser: ObservationParser | None = None,
    decision_index: int | None = None,
) -> str:
    """The full per-decision block: board, legal actions, choice, timing.

    Used by ``ptcg battle --verbose`` (live) and ``ptcg show-episode --step``
    (post-hoc) so both render identically.
    """
    parser = parser if parser is not None else ObservationParser()
    obs = parser.parse(raw_obs)
    bar = "=" * 60
    header = _header(obs, decision_index)
    lines = [bar, header, bar, format_observation(obs, cards)]

    if obs.select is not None and obs.select.option:
        lines.append("")
        lines.append("Legal actions:")
        described = [
            describe_action(resolve_option(option, obs), cards)
            for option in obs.select.option
        ]
        lines.extend(f"  [{i}] {text}" for i, text in enumerate(described))
        if action is not None:
            chosen = [described[i] for i in action if 0 <= i < len(described)]
            lines.append("")
            lines.append("Chosen: " + ("; ".join(chosen) if chosen else "(nothing — empty selection)"))
    if elapsed_ms is not None:
        lines.append(f"Decision time: {elapsed_ms:.2f} ms")
    lines.append(bar)
    return "\n".join(lines)


def _header(obs: ParsedObservation, decision_index: int | None) -> str:
    parts = []
    if decision_index is not None:
        parts.append(f"Decision {decision_index}")
    if obs.current is not None:
        parts.append(f"Turn {obs.current.turn}")
        parts.append(f"P{obs.current.yourIndex} to act")
    return " | ".join(parts) if parts else "Decision"


def _describe_attack(attack_id: int | None, cards: CardDatabase | None) -> str:
    if attack_id is None:
        return "Attack: ?"
    attack = cards.get_attack(attack_id) if cards is not None else None
    if attack is None:
        return f"Attack: #{attack_id}"
    damage = f" ({attack.damage} dmg)" if attack.damage else ""
    return f"Attack: {attack.name}{damage}"


def _name(cards: CardDatabase | None, card_id: int | None) -> str:
    if card_id is None:
        return "(face-down/unknown)"
    if cards is None:
        return f"#{card_id}"
    return cards.name(card_id)
