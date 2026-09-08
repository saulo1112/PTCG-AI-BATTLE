"""Human-readable rendering of observations, options, and logs (ADR-0009).

All formatters accept an optional :class:`CardDatabase` — with it, card IDs
become names; without it (e.g. no SDK loaded), IDs are shown raw. Formatters
never raise on odd input; they are debugging tools.
"""

from __future__ import annotations

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.observation.models import (
    ParsedLog,
    ParsedObservation,
    ParsedOption,
    ParsedPlayer,
    ParsedPokemon,
    ParsedSelect,
)

_STATUS_FLAGS = ("poisoned", "burned", "asleep", "paralyzed", "confused")


def _name(cards: CardDatabase | None, card_id: int | None) -> str:
    if card_id is None:
        return "?"
    if cards is None:
        return f"#{card_id}"
    return cards.name(card_id)


def format_observation(obs: ParsedObservation, cards: CardDatabase | None = None) -> str:
    """Render a full observation as an indented text block."""
    lines: list[str] = []
    state = obs.current
    if state is None:
        lines.append("== initial deck submission (no board state) ==")
    else:
        first = {0: "P0", 1: "P1", -1: "?"}.get(state.firstPlayer, "?")
        lines.append(
            f"== turn {state.turn} | acting: P{state.yourIndex} | first: {first} "
            f"| result: {state.result} =="
        )
        flags = [
            name
            for name, on in (
                ("supporterPlayed", state.supporterPlayed),
                ("stadiumPlayed", state.stadiumPlayed),
                ("energyAttached", state.energyAttached),
                ("retreated", state.retreated),
            )
            if on
        ]
        if flags:
            lines.append(f"turn flags: {', '.join(flags)}")
        if state.stadium:
            lines.append(f"stadium: {_name(cards, state.stadium[0].id)}")
        for idx, player in enumerate(state.players):
            you = " (you)" if idx == state.yourIndex else ""
            lines.append(f"-- P{idx}{you} --")
            lines.extend("  " + ln for ln in _player_lines(player, cards))
    if obs.select is not None:
        lines.append(format_select(obs.select, cards))
    return "\n".join(lines)


def _player_lines(player: ParsedPlayer, cards: CardDatabase | None) -> list[str]:
    lines: list[str] = []
    if player.active:
        active = player.active[0]
        lines.append(f"active: {_pokemon_str(active, cards)}")
    else:
        lines.append("active: (none)")
    statuses = [flag for flag in _STATUS_FLAGS if getattr(player, flag)]
    if statuses:
        lines[-1] += f"  [{', '.join(statuses)}]"
    for pokemon in player.bench:
        lines.append(f"bench:  {_pokemon_str(pokemon, cards)}")
    if player.hand is not None:
        hand = ", ".join(_name(cards, c.id) for c in player.hand) or "(empty)"
        lines.append(f"hand ({player.handCount}): {hand}")
    else:
        lines.append(f"hand: {player.handCount} cards (hidden)")
    face_up = [c for c in player.prize if c is not None]
    lines.append(
        f"deck: {player.deckCount} | prizes: {len(player.prize)}"
        + (f" ({len(face_up)} face-up)" if face_up else "")
        + f" | discard: {len(player.discard)}"
    )
    return lines


def _pokemon_str(pokemon: ParsedPokemon | None, cards: CardDatabase | None) -> str:
    if pokemon is None:
        return "(face-down)"
    energy = "".join(e.name[0] for e in pokemon.energies) or "-"
    tools = f" tools:{len(pokemon.tools)}" if pokemon.tools else ""
    return (
        f"{_name(cards, pokemon.id)} {pokemon.hp}/{pokemon.maxHp}HP "
        f"energy:{energy}{tools} (serial {pokemon.serial})"
    )


def format_select(select: ParsedSelect, cards: CardDatabase | None = None) -> str:
    """Render a decision request with numbered options."""
    header = (
        f"select {select.type.name}/{select.context.name} "
        f"pick {select.minCount}..{select.maxCount} of {len(select.option)}"
    )
    if select.effect is not None:
        header += f" | effect: {_name(cards, select.effect.id)}"
    if select.contextCard is not None:
        header += f" | about: {_name(cards, select.contextCard.id)}"
    lines = [header]
    lines.extend(
        f"  [{i}] {format_option(option, cards)}" for i, option in enumerate(select.option)
    )
    if select.deck is not None:
        lines.append(f"  (choosing from your deck: {len(select.deck)} cards)")
    return "\n".join(lines)


def format_option(option: ParsedOption, cards: CardDatabase | None = None) -> str:
    """One option on one line, decoded per its kind."""
    parts = [option.type.name]
    if option.cardId is not None:
        parts.append(_name(cards, option.cardId))
    if option.attackId is not None:
        attack = cards.get_attack(option.attackId) if cards is not None else None
        parts.append(attack.name if attack is not None else f"attack#{option.attackId}")
        if attack is not None and attack.damage:
            parts.append(f"{attack.damage}dmg")
    if option.area is not None:
        loc = f"{option.area.name}[{option.index}]"
        if option.playerIndex is not None:
            loc = f"P{option.playerIndex}.{loc}"
        parts.append(loc)
    if option.inPlayArea is not None:
        parts.append(f"-> {option.inPlayArea.name}[{option.inPlayIndex}]")
    if option.number is not None:
        parts.append(f"n={option.number}")
    if option.count is not None:
        parts.append(f"count={option.count}")
    if option.energyIndex is not None:
        parts.append(f"energy[{option.energyIndex}]")
    if option.toolIndex is not None:
        parts.append(f"tool[{option.toolIndex}]")
    if option.specialConditionType is not None:
        parts.append(option.specialConditionType.name)
    return " ".join(parts)


def summarize_logs(logs: tuple[ParsedLog, ...], cards: CardDatabase | None = None) -> str:
    """One line per engine event since the previous decision."""
    lines = []
    for log in logs:
        bits = [log.type.name]
        data = log.data
        if "playerIndex" in data:
            bits.append(f"P{data['playerIndex']}")
        if data.get("cardId") is not None:
            bits.append(_name(cards, data["cardId"]))
        for key in ("value", "attackId", "head", "result", "reason", "isRecover"):
            if data.get(key) is not None:
                bits.append(f"{key}={data[key]}")
        lines.append(" ".join(bits))
    return "\n".join(lines)
