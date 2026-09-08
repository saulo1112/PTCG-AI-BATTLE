"""Our mirror of the engine's observation schema (ADR-0005).

Design rules:

- **Field names are kept verbatim from the SDK** (camelCase), so the official
  ``cg/api.py`` docstrings and raw JSON remain the single reference. This is
  a deliberate PEP 8 deviation.
- Every model carries ``extras: dict`` holding any keys the parser did not
  recognize — additive schema changes flow through instead of crashing.
- Enum fields use :class:`TolerantEnum`: unknown integers become
  ``UNKNOWN_<n>`` pseudo-members instead of raising.

Semantics of each field are documented in the vendored ``cg/api.py``; only
non-obvious competition-relevant facts are repeated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Mapping


class TolerantEnum(IntEnum):
    """IntEnum that fabricates ``UNKNOWN_<n>`` members for unlisted values.

    The SDK warns that enum members may be appended during the competition;
    an unknown value must never crash the agent (ADR-0005).
    """

    @classmethod
    def _missing_(cls, value: object) -> "TolerantEnum | None":
        if isinstance(value, int):
            pseudo = int.__new__(cls, value)
            pseudo._name_ = f"UNKNOWN_{value}"
            pseudo._value_ = value
            return pseudo
        return None

    @property
    def is_unknown(self) -> bool:
        return self._name_.startswith("UNKNOWN_")


class AreaKind(TolerantEnum):
    DECK = 1
    HAND = 2
    DISCARD = 3
    ACTIVE = 4
    BENCH = 5
    PRIZE = 6
    STADIUM = 7
    ENERGY = 8
    TOOL = 9
    PRE_EVOLUTION = 10
    PLAYER = 11
    LOOKING = 12


class EnergyKind(TolerantEnum):
    COLORLESS = 0
    GRASS = 1
    FIRE = 2
    WATER = 3
    LIGHTNING = 4
    PSYCHIC = 5
    FIGHTING = 6
    DARKNESS = 7
    METAL = 8
    DRAGON = 9
    RAINBOW = 10
    TEAM_ROCKET = 11


class CardKind(TolerantEnum):
    POKEMON = 0
    ITEM = 1
    TOOL = 2
    SUPPORTER = 3
    STADIUM = 4
    BASIC_ENERGY = 5
    SPECIAL_ENERGY = 6


class SpecialConditionKind(TolerantEnum):
    POISON = 0
    BURN = 1
    SLEEP = 2
    PARALYZE = 3
    CONFUSE = 4


class SelectKind(TolerantEnum):
    MAIN = 0
    CARD = 1
    ATTACHED_CARD = 2
    CARD_OR_ATTACHED_CARD = 3
    ENERGY = 4
    SKILL = 5
    ATTACK = 6
    EVOLVE = 7
    COUNT = 8
    YES_NO = 9
    SPECIAL_CONDITION = 10


class SelectContextKind(TolerantEnum):
    MAIN = 0
    SETUP_ACTIVE_POKEMON = 1
    SETUP_BENCH_POKEMON = 2
    SWITCH = 3
    TO_ACTIVE = 4
    TO_BENCH = 5
    TO_FIELD = 6
    TO_HAND = 7
    DISCARD = 8
    TO_DECK = 9
    TO_DECK_BOTTOM = 10
    TO_PRIZE = 11
    NOT_MOVE = 12
    DAMAGE_COUNTER = 13
    DAMAGE_COUNTER_ANY = 14
    DAMAGE = 15
    REMOVE_DAMAGE_COUNTER = 16
    HEAL = 17
    EVOLVES_FROM = 18
    EVOLVES_TO = 19
    DEVOLVE = 20
    ATTACH_FROM = 21
    ATTACH_TO = 22
    DETACH_FROM = 23
    LOOK = 24
    EFFECT_TARGET = 25
    DISCARD_ENERGY_CARD = 26
    DISCARD_TOOL_CARD = 27
    SWITCH_ENERGY_CARD = 28
    DISCARD_CARD_OR_ATTACHED_CARD = 29
    DISCARD_ENERGY = 30
    TO_HAND_ENERGY = 31
    TO_DECK_ENERGY = 32
    SWITCH_ENERGY = 33
    SKILL_ORDER = 34
    ATTACK = 35
    DISABLE_ATTACK = 36
    EVOLVE = 37
    DRAW_COUNT = 38
    DAMAGE_COUNTER_COUNT = 39
    REMOVE_DAMAGE_COUNTER_COUNT = 40
    IS_FIRST = 41
    MULLIGAN = 42
    ACTIVATE = 43
    FIRST_EFFECT = 44
    MORE_DEVOLVE = 45
    COIN_HEAD = 46
    AFFECT_SPECIAL_CONDITION = 47
    RECOVER_SPECIAL_CONDITION = 48


class OptionKind(TolerantEnum):
    NUMBER = 0
    YES = 1
    NO = 2
    CARD = 3
    TOOL_CARD = 4
    ENERGY_CARD = 5
    ENERGY = 6
    PLAY = 7
    ATTACH = 8
    EVOLVE = 9
    ABILITY = 10
    DISCARD = 11
    RETREAT = 12
    ATTACK = 13
    END = 14
    SKILL = 15
    SPECIAL_CONDITION = 16


class LogKind(TolerantEnum):
    SHUFFLE = 0
    HAS_BASIC_POKEMON = 1
    TURN_START = 2
    TURN_END = 3
    DRAW = 4
    DRAW_REVERSE = 5
    MOVE_CARD = 6
    MOVE_CARD_REVERSE = 7
    SWITCH = 8
    CHANGE = 9
    PLAY = 10
    ATTACH = 11
    EVOLVE = 12
    DEVOLVE = 13
    MOVE_ATTACHED = 14
    ATTACK = 15
    HP_CHANGE = 16
    POISONED = 17
    BURNED = 18
    ASLEEP = 19
    PARALYZED = 20
    CONFUSED = 21
    COIN = 22
    RESULT = 23


@dataclass(frozen=True)
class ParsedCard:
    """A face-up card reference (identity within the match is ``serial``)."""

    id: int
    serial: int
    playerIndex: int
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedPokemon:
    """A Pokémon in play with its attachments."""

    id: int
    serial: int
    hp: int
    maxHp: int
    appearThisTurn: bool
    energies: tuple[EnergyKind, ...]
    energyCards: tuple[ParsedCard, ...]
    tools: tuple[ParsedCard, ...]
    preEvolution: tuple[ParsedCard, ...]
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedPlayer:
    """One player's visible state.

    ``hand`` is ``None`` for the opponent (hidden information — only
    ``handCount`` is known). Face-down cards in ``active``/``prize`` are
    ``None``.
    """

    active: tuple[ParsedPokemon | None, ...]
    bench: tuple[ParsedPokemon, ...]
    benchMax: int
    deckCount: int
    discard: tuple[ParsedCard, ...]
    prize: tuple[ParsedCard | None, ...]
    handCount: int
    hand: tuple[ParsedCard, ...] | None
    poisoned: bool
    burned: bool
    asleep: bool
    paralyzed: bool
    confused: bool
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedState:
    """Board state. ``yourIndex`` is the player this observation was built for."""

    turn: int
    turnActionCount: int
    yourIndex: int
    firstPlayer: int
    supporterPlayed: bool
    stadiumPlayed: bool
    energyAttached: bool
    retreated: bool
    result: int
    stadium: tuple[ParsedCard, ...]
    looking: tuple[ParsedCard | None, ...] | None
    players: tuple[ParsedPlayer, ...]
    extras: Mapping[str, Any] = field(default_factory=dict)

    @property
    def me(self) -> ParsedPlayer:
        return self.players[self.yourIndex]

    @property
    def opponent(self) -> ParsedPlayer:
        return self.players[1 - self.yourIndex]


@dataclass(frozen=True)
class ParsedOption:
    """One legal choice. Which fields are populated depends on ``type``
    (see the field tables in the vendored ``cg/api.py``)."""

    type: OptionKind
    number: int | None = None
    area: AreaKind | None = None
    index: int | None = None
    playerIndex: int | None = None
    toolIndex: int | None = None
    energyIndex: int | None = None
    count: int | None = None
    inPlayArea: AreaKind | None = None
    inPlayIndex: int | None = None
    attackId: int | None = None
    cardId: int | None = None
    serial: int | None = None
    specialConditionType: SpecialConditionKind | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedSelect:
    """A decision request: pick ``minCount..maxCount`` indices into ``option``."""

    type: SelectKind
    context: SelectContextKind
    minCount: int
    maxCount: int
    remainDamageCounter: int
    remainEnergyCost: int
    option: tuple[ParsedOption, ...]
    deck: tuple[ParsedCard, ...] | None
    contextCard: ParsedCard | None
    effect: ParsedCard | None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedLog:
    """One engine event.

    Log entries have ~20 optional fields that vary by ``type``; they are kept
    as the raw ``data`` mapping (minus ``type``) rather than 20 mostly-None
    attributes. Use ``log.data.get("cardId")`` etc.; field meanings per type
    are documented in the vendored ``cg/api.py``.
    """

    type: LogKind
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedObservation:
    """A full parsed observation.

    ``select`` and ``current`` are ``None`` only for the Kaggle-side initial
    deck-submission call. ``search_begin_input`` is the opaque serialized
    state consumed by the SDK search API.
    """

    select: ParsedSelect | None
    logs: tuple[ParsedLog, ...]
    current: ParsedState | None
    search_begin_input: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)
