"""Tolerant raw-dict → parsed-model conversion (ADR-0005).

The parser never raises on *additive* schema changes: unknown dict keys land
in the model's ``extras`` mapping and unknown enum integers become
``UNKNOWN_<n>`` pseudo-members. Structural changes (a required key
disappearing) still raise — that is a real incompatibility we must notice.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, TypeVar

from ptcg_ai.observation.models import (
    AreaKind,
    EnergyKind,
    LogKind,
    OptionKind,
    ParsedCard,
    ParsedLog,
    ParsedObservation,
    ParsedOption,
    ParsedPlayer,
    ParsedPokemon,
    ParsedSelect,
    ParsedState,
    SelectContextKind,
    SelectKind,
    SpecialConditionKind,
)

_T = TypeVar("_T")


class ObservationParser:
    """Stateless parser; safe to share across battles and threads."""

    def parse(self, raw: Mapping[str, Any]) -> ParsedObservation:
        """Parse a raw observation dict as delivered by the engine/Kaggle."""
        known = {"select", "logs", "current", "search_begin_input"}
        return ParsedObservation(
            select=_opt(raw.get("select"), self._select),
            logs=tuple(self._log(entry) for entry in raw.get("logs") or ()),
            current=_opt(raw.get("current"), self._state),
            search_begin_input=raw.get("search_begin_input"),
            extras=_extras(raw, known),
        )

    def _select(self, raw: Mapping[str, Any]) -> ParsedSelect:
        known = {
            "type", "context", "minCount", "maxCount", "remainDamageCounter",
            "remainEnergyCost", "option", "deck", "contextCard", "effect",
        }
        return ParsedSelect(
            type=SelectKind(raw["type"]),
            context=SelectContextKind(raw["context"]),
            minCount=int(raw["minCount"]),
            maxCount=int(raw["maxCount"]),
            remainDamageCounter=int(raw.get("remainDamageCounter", 0)),
            remainEnergyCost=int(raw.get("remainEnergyCost", 0)),
            option=tuple(self._option(o) for o in raw.get("option") or ()),
            deck=_opt_tuple(raw.get("deck"), self._card),
            contextCard=_opt(raw.get("contextCard"), self._card),
            effect=_opt(raw.get("effect"), self._card),
            extras=_extras(raw, known),
        )

    def _option(self, raw: Mapping[str, Any]) -> ParsedOption:
        known = {
            "type", "number", "area", "index", "playerIndex", "toolIndex",
            "energyIndex", "count", "inPlayArea", "inPlayIndex", "attackId",
            "cardId", "serial", "specialConditionType",
        }
        return ParsedOption(
            type=OptionKind(raw["type"]),
            number=raw.get("number"),
            area=_opt(raw.get("area"), AreaKind),
            index=raw.get("index"),
            playerIndex=raw.get("playerIndex"),
            toolIndex=raw.get("toolIndex"),
            energyIndex=raw.get("energyIndex"),
            count=raw.get("count"),
            inPlayArea=_opt(raw.get("inPlayArea"), AreaKind),
            inPlayIndex=raw.get("inPlayIndex"),
            attackId=raw.get("attackId"),
            cardId=raw.get("cardId"),
            serial=raw.get("serial"),
            specialConditionType=_opt(raw.get("specialConditionType"), SpecialConditionKind),
            extras=_extras(raw, known),
        )

    def _state(self, raw: Mapping[str, Any]) -> ParsedState:
        known = {
            "turn", "turnActionCount", "yourIndex", "firstPlayer",
            "supporterPlayed", "stadiumPlayed", "energyAttached", "retreated",
            "result", "stadium", "looking", "players",
        }
        return ParsedState(
            turn=int(raw["turn"]),
            turnActionCount=int(raw.get("turnActionCount", 0)),
            yourIndex=int(raw["yourIndex"]),
            firstPlayer=int(raw.get("firstPlayer", -1)),
            supporterPlayed=bool(raw.get("supporterPlayed", False)),
            stadiumPlayed=bool(raw.get("stadiumPlayed", False)),
            energyAttached=bool(raw.get("energyAttached", False)),
            retreated=bool(raw.get("retreated", False)),
            result=int(raw.get("result", -1)),
            stadium=tuple(self._card(c) for c in raw.get("stadium") or ()),
            looking=_opt_tuple_nullable(raw.get("looking"), self._card),
            players=tuple(self._player(p) for p in raw.get("players") or ()),
            extras=_extras(raw, known),
        )

    def _player(self, raw: Mapping[str, Any]) -> ParsedPlayer:
        known = {
            "active", "bench", "benchMax", "deckCount", "discard", "prize",
            "handCount", "hand", "poisoned", "burned", "asleep", "paralyzed",
            "confused",
        }
        return ParsedPlayer(
            active=tuple(_opt(p, self._pokemon) for p in raw.get("active") or ()),
            bench=tuple(self._pokemon(p) for p in raw.get("bench") or ()),
            benchMax=int(raw.get("benchMax", 5)),
            deckCount=int(raw.get("deckCount", 0)),
            discard=tuple(self._card(c) for c in raw.get("discard") or ()),
            prize=tuple(_opt(c, self._card) for c in raw.get("prize") or ()),
            handCount=int(raw.get("handCount", 0)),
            hand=_opt_tuple(raw.get("hand"), self._card),
            poisoned=bool(raw.get("poisoned", False)),
            burned=bool(raw.get("burned", False)),
            asleep=bool(raw.get("asleep", False)),
            paralyzed=bool(raw.get("paralyzed", False)),
            confused=bool(raw.get("confused", False)),
            extras=_extras(raw, known),
        )

    def _pokemon(self, raw: Mapping[str, Any]) -> ParsedPokemon:
        known = {
            "id", "serial", "hp", "maxHp", "appearThisTurn", "energies",
            "energyCards", "tools", "preEvolution", "playerIndex",
        }
        return ParsedPokemon(
            id=int(raw["id"]),
            serial=int(raw["serial"]),
            hp=int(raw.get("hp", 0)),
            maxHp=int(raw.get("maxHp", 0)),
            appearThisTurn=bool(raw.get("appearThisTurn", False)),
            energies=tuple(EnergyKind(e) for e in raw.get("energies") or ()),
            energyCards=tuple(self._card(c) for c in raw.get("energyCards") or ()),
            tools=tuple(self._card(c) for c in raw.get("tools") or ()),
            preEvolution=tuple(self._card(c) for c in raw.get("preEvolution") or ()),
            extras=_extras(raw, known),
        )

    def _card(self, raw: Mapping[str, Any]) -> ParsedCard:
        known = {"id", "serial", "playerIndex"}
        return ParsedCard(
            id=int(raw["id"]),
            serial=int(raw.get("serial", -1)),
            playerIndex=int(raw.get("playerIndex", -1)),
            extras=_extras(raw, known),
        )

    def _log(self, raw: Mapping[str, Any]) -> ParsedLog:
        data = {k: v for k, v in raw.items() if k != "type"}
        return ParsedLog(type=LogKind(raw["type"]), data=data)


def _extras(raw: Mapping[str, Any], known: set[str]) -> dict[str, Any]:
    """Keys the current schema does not know about (schema-drift signal)."""
    return {k: v for k, v in raw.items() if k not in known}


def _opt(value: Any, convert: Callable[[Any], _T]) -> _T | None:
    return None if value is None else convert(value)


def _opt_tuple(value: Any, convert: Callable[[Any], _T]) -> tuple[_T, ...] | None:
    return None if value is None else tuple(convert(v) for v in value)


def _opt_tuple_nullable(
    value: Any, convert: Callable[[Any], _T]
) -> tuple[_T | None, ...] | None:
    """A nullable list whose *elements* may also be null (e.g. ``looking``)."""
    return None if value is None else tuple(_opt(v, convert) for v in value)
