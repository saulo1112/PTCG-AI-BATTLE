"""Mirror of the engine's static card metadata.

The authoritative source is the engine itself (``all_card_data()`` /
``all_attack()``); the CSVs under the vendored folder are a human reference
only. Vendored dataclass instances are converted to our own frozen types at
load time so no ``cg`` types leak past the environment boundary (ADR-0002).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ptcg_ai.observation.models import CardKind, EnergyKind

if TYPE_CHECKING:
    from ptcg_ai.environment.sdk import SdkModules

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillInfo:
    """An ability/effect blurb attached to a card."""

    name: str
    text: str


@dataclass(frozen=True)
class AttackInfo:
    """One attack, referenced from cards by ``attackId``."""

    attackId: int
    name: str
    text: str
    damage: int
    energies: tuple[EnergyKind, ...]


@dataclass(frozen=True)
class CardInfo:
    """Static definition of one card (field names verbatim from the SDK)."""

    cardId: int
    name: str
    cardType: CardKind
    retreatCost: int
    hp: int
    weakness: EnergyKind | None
    resistance: EnergyKind | None
    energyType: EnergyKind
    basic: bool
    stage1: bool
    stage2: bool
    ex: bool
    megaEx: bool
    tera: bool
    aceSpec: bool
    evolvesFrom: str | None
    skills: tuple[SkillInfo, ...]
    attacks: tuple[int, ...]

    @property
    def is_basic_pokemon(self) -> bool:
        return self.cardType is CardKind.POKEMON and self.basic

    @property
    def is_basic_energy(self) -> bool:
        return self.cardType is CardKind.BASIC_ENERGY


class CardDatabase:
    """Id-keyed lookup over all cards and attacks in the competition pool."""

    def __init__(self, cards: dict[int, CardInfo], attacks: dict[int, AttackInfo]) -> None:
        self._cards = cards
        self._attacks = attacks

    @classmethod
    def from_sdk(cls, sdk: "SdkModules") -> "CardDatabase":
        """Load and convert the engine's card/attack tables (one-time cost)."""
        cards = {cd.cardId: _convert_card(cd) for cd in sdk.api.all_card_data()}
        attacks = {a.attackId: _convert_attack(a) for a in sdk.api.all_attack()}
        logger.info("Card database loaded: %d cards, %d attacks", len(cards), len(attacks))
        return cls(cards, attacks)

    def to_records(self) -> dict[str, list[dict[str, Any]]]:
        """Serialize to plain JSON-safe records (enums → ints).

        Lets the Kaggle submission carry static card knowledge as a stdlib
        ``json`` file instead of loading the native engine at runtime
        (ADR-0014); the inverse is :meth:`from_records`.
        """
        return {
            "cards": [_card_to_record(c) for c in self._cards.values()],
            "attacks": [_attack_to_record(a) for a in self._attacks.values()],
        }

    @classmethod
    def from_records(cls, data: dict[str, list[dict[str, Any]]]) -> "CardDatabase":
        """Rebuild from :meth:`to_records` output (no SDK/native lib needed)."""
        cards = {r["cardId"]: _card_from_record(r) for r in data.get("cards", ())}
        attacks = {r["attackId"]: _attack_from_record(r) for r in data.get("attacks", ())}
        return cls(cards, attacks)

    def get_card(self, card_id: int) -> CardInfo | None:
        return self._cards.get(card_id)

    def get_attack(self, attack_id: int) -> AttackInfo | None:
        return self._attacks.get(attack_id)

    def name(self, card_id: int) -> str:
        """Card name, or a placeholder for unknown ids (never raises)."""
        card = self._cards.get(card_id)
        return card.name if card is not None else f"<card {card_id}>"

    def __len__(self) -> int:
        return len(self._cards)

    def __contains__(self, card_id: int) -> bool:
        return card_id in self._cards


def _convert_card(cd: Any) -> CardInfo:
    """Vendored ``CardData`` → :class:`CardInfo` (tolerant of new fields)."""
    return CardInfo(
        cardId=cd.cardId,
        name=cd.name,
        cardType=CardKind(cd.cardType),
        retreatCost=cd.retreatCost,
        hp=cd.hp,
        weakness=None if cd.weakness is None else EnergyKind(cd.weakness),
        resistance=None if cd.resistance is None else EnergyKind(cd.resistance),
        energyType=EnergyKind(cd.energyType),
        basic=cd.basic,
        stage1=cd.stage1,
        stage2=cd.stage2,
        ex=cd.ex,
        megaEx=getattr(cd, "megaEx", False),
        tera=getattr(cd, "tera", False),
        aceSpec=getattr(cd, "aceSpec", False),
        evolvesFrom=cd.evolvesFrom,
        skills=tuple(SkillInfo(name=s.name, text=s.text) for s in cd.skills),
        attacks=tuple(cd.attacks),
    )


def _convert_attack(a: Any) -> AttackInfo:
    return AttackInfo(
        attackId=a.attackId,
        name=a.name,
        text=a.text,
        damage=a.damage,
        energies=tuple(EnergyKind(e) for e in a.energies),
    )


def _enum_val(e: Any) -> int | None:
    return None if e is None else int(e)


def _card_to_record(c: CardInfo) -> dict[str, Any]:
    return {
        "cardId": c.cardId, "name": c.name, "cardType": int(c.cardType),
        "retreatCost": c.retreatCost, "hp": c.hp,
        "weakness": _enum_val(c.weakness), "resistance": _enum_val(c.resistance),
        "energyType": int(c.energyType),
        "basic": c.basic, "stage1": c.stage1, "stage2": c.stage2,
        "ex": c.ex, "megaEx": c.megaEx, "tera": c.tera, "aceSpec": c.aceSpec,
        "evolvesFrom": c.evolvesFrom,
        "skills": [{"name": s.name, "text": s.text} for s in c.skills],
        "attacks": list(c.attacks),
    }


def _card_from_record(r: dict[str, Any]) -> CardInfo:
    return CardInfo(
        cardId=r["cardId"], name=r["name"], cardType=CardKind(r["cardType"]),
        retreatCost=r["retreatCost"], hp=r["hp"],
        weakness=None if r["weakness"] is None else EnergyKind(r["weakness"]),
        resistance=None if r["resistance"] is None else EnergyKind(r["resistance"]),
        energyType=EnergyKind(r["energyType"]),
        basic=r["basic"], stage1=r["stage1"], stage2=r["stage2"],
        ex=r["ex"], megaEx=r["megaEx"], tera=r["tera"], aceSpec=r["aceSpec"],
        evolvesFrom=r["evolvesFrom"],
        skills=tuple(SkillInfo(name=s["name"], text=s["text"]) for s in r["skills"]),
        attacks=tuple(r["attacks"]),
    )


def _attack_to_record(a: AttackInfo) -> dict[str, Any]:
    return {
        "attackId": a.attackId, "name": a.name, "text": a.text,
        "damage": a.damage, "energies": [int(e) for e in a.energies],
    }


def _attack_from_record(r: dict[str, Any]) -> AttackInfo:
    return AttackInfo(
        attackId=r["attackId"], name=r["name"], text=r["text"],
        damage=r["damage"], energies=tuple(EnergyKind(e) for e in r["energies"]),
    )
