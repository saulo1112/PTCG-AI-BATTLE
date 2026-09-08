"""Structured resolution of options against the board state.

An :class:`~ptcg_ai.observation.models.ParsedOption` mostly carries *indices*
("hand[3]", "bench[1]"); this module resolves them into concrete card IDs so
that analytics can count semantics ("attached card 3 to card 721") and the
debug renderer can name things — without either of them re-implementing zone
lookups.

Resolution is defensive: hidden zones (opponent hand, face-down cards) and
out-of-range indices yield ``resolved=False`` with ``None`` fields, never an
exception. Analytics must treat unresolved actions as their own category,
not drop them silently.
"""

from __future__ import annotations

from dataclasses import dataclass

from ptcg_ai.observation.models import (
    AreaKind,
    OptionKind,
    ParsedObservation,
    ParsedOption,
    ParsedPokemon,
    ParsedSelect,
    ParsedState,
    SpecialConditionKind,
)


@dataclass(frozen=True)
class ResolvedAction:
    """One option with its indices resolved to card identities.

    ``card_id`` is the primary card acted with/on (the played card, the
    selected target, the attached energy, the retreating Pokémon…);
    ``target_card_id`` is the secondary card for two-sided actions
    (ATTACH/EVOLVE: the Pokémon receiving the card).
    """

    kind: OptionKind
    card_id: int | None = None
    target_card_id: int | None = None
    attack_id: int | None = None
    number: int | None = None
    special_condition: SpecialConditionKind | None = None
    source_area: AreaKind | None = None
    target_area: AreaKind | None = None
    resolved: bool = True


def resolve_option(option: ParsedOption, observation: ParsedObservation) -> ResolvedAction:
    """Resolve one legal option against the observation it came with."""
    state = observation.current
    select = observation.select
    kind = option.type

    if kind is OptionKind.NUMBER:
        return ResolvedAction(kind=kind, number=option.number)
    if kind in (OptionKind.YES, OptionKind.NO, OptionKind.END):
        return ResolvedAction(kind=kind)
    if kind is OptionKind.ATTACK:
        return ResolvedAction(kind=kind, attack_id=option.attackId)
    if kind is OptionKind.SKILL:
        return ResolvedAction(kind=kind, card_id=option.cardId)
    if kind is OptionKind.SPECIAL_CONDITION:
        return ResolvedAction(kind=kind, special_condition=option.specialConditionType)
    if state is None:
        return ResolvedAction(kind=kind, resolved=False)

    if kind is OptionKind.RETREAT:
        active = state.me.active[0] if state.me.active else None
        return ResolvedAction(
            kind=kind,
            card_id=active.id if active is not None else None,
            source_area=AreaKind.ACTIVE,
            resolved=active is not None,
        )

    if kind is OptionKind.PLAY:
        card_id = _card_id_at(state, select, state.yourIndex, AreaKind.HAND, option.index)
        return ResolvedAction(
            kind=kind, card_id=card_id, source_area=AreaKind.HAND, resolved=card_id is not None
        )

    if kind is OptionKind.CARD:
        player = option.playerIndex if option.playerIndex is not None else state.yourIndex
        card_id = _card_id_at(state, select, player, option.area, option.index)
        return ResolvedAction(
            kind=kind, card_id=card_id, source_area=option.area, resolved=card_id is not None
        )

    if kind in (OptionKind.TOOL_CARD, OptionKind.ENERGY_CARD, OptionKind.ENERGY):
        player = option.playerIndex if option.playerIndex is not None else state.yourIndex
        pokemon = _pokemon_at(state, player, option.area, option.index)
        card_id = None
        if pokemon is not None:
            attached = pokemon.tools if kind is OptionKind.TOOL_CARD else pokemon.energyCards
            attached_index = (
                option.toolIndex if kind is OptionKind.TOOL_CARD else option.energyIndex
            )
            card_id = _id_at_index(attached, attached_index)
        return ResolvedAction(
            kind=kind,
            card_id=card_id,
            target_card_id=pokemon.id if pokemon is not None else None,
            source_area=option.area,
            resolved=card_id is not None,
        )

    if kind in (OptionKind.ATTACH, OptionKind.EVOLVE):
        card_id = _card_id_at(state, select, state.yourIndex, option.area, option.index)
        target = _pokemon_at(state, state.yourIndex, option.inPlayArea, option.inPlayIndex)
        return ResolvedAction(
            kind=kind,
            card_id=card_id,
            target_card_id=target.id if target is not None else None,
            source_area=option.area,
            target_area=option.inPlayArea,
            resolved=card_id is not None and target is not None,
        )

    if kind in (OptionKind.ABILITY, OptionKind.DISCARD):
        card_id = _card_id_at(state, select, state.yourIndex, option.area, option.index)
        return ResolvedAction(
            kind=kind, card_id=card_id, source_area=option.area, resolved=card_id is not None
        )

    # Unknown/new option kinds degrade gracefully (ADR-0005).
    return ResolvedAction(kind=kind, resolved=False)


def _card_id_at(
    state: ParsedState,
    select: ParsedSelect | None,
    player_index: int,
    area: AreaKind | None,
    index: int | None,
) -> int | None:
    """Card ID in a zone, or ``None`` for hidden/unknown/out-of-range."""
    if area is None or index is None or not (0 <= player_index < len(state.players)):
        return None
    player = state.players[player_index]
    if area is AreaKind.DECK:
        return _id_at_index(select.deck, index) if select is not None and select.deck else None
    if area is AreaKind.HAND:
        return _id_at_index(player.hand, index) if player.hand is not None else None
    if area is AreaKind.DISCARD:
        return _id_at_index(player.discard, index)
    if area is AreaKind.PRIZE:
        return _id_at_index(player.prize, index)
    if area is AreaKind.STADIUM:
        return _id_at_index(state.stadium, index)
    if area is AreaKind.LOOKING:
        return _id_at_index(state.looking, index) if state.looking is not None else None
    if area in (AreaKind.ACTIVE, AreaKind.BENCH):
        pokemon = _pokemon_at(state, player_index, area, index)
        return pokemon.id if pokemon is not None else None
    return None


def _pokemon_at(
    state: ParsedState, player_index: int, area: AreaKind | None, index: int | None
) -> ParsedPokemon | None:
    if area is None or index is None or not (0 <= player_index < len(state.players)):
        return None
    player = state.players[player_index]
    zone = player.active if area is AreaKind.ACTIVE else (
        player.bench if area is AreaKind.BENCH else None
    )
    if zone is None:
        return None
    return _at_index(zone, index)


def _at_index(seq, index):  # type: ignore[no-untyped-def]
    if index is None or not (0 <= index < len(seq)):
        return None
    return seq[index]


def _id_at_index(seq, index) -> int | None:  # type: ignore[no-untyped-def]
    item = _at_index(seq, index) if seq is not None else None
    return getattr(item, "id", None) if item is not None else None
