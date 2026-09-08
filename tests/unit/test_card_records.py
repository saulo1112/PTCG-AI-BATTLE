"""CardDatabase JSON records round-trip (stdlib-only, no SDK)."""

from __future__ import annotations

import json

from ptcg_ai.cards.database import AttackInfo, CardDatabase, CardInfo, SkillInfo
from ptcg_ai.observation.models import CardKind, EnergyKind


def _db() -> CardDatabase:
    cards = {
        721: CardInfo(
            cardId=721, name="Kyogre", cardType=CardKind.POKEMON, retreatCost=2,
            hp=60, weakness=EnergyKind.LIGHTNING, resistance=None,
            energyType=EnergyKind.WATER, basic=True, stage1=False, stage2=False,
            ex=False, megaEx=False, tera=False, aceSpec=False, evolvesFrom=None,
            skills=(SkillInfo(name="Downpour", text="do a thing"),), attacks=(900,),
        ),
        723: CardInfo(
            cardId=723, name="Mega Abomasnow ex", cardType=CardKind.POKEMON,
            retreatCost=3, hp=330, weakness=EnergyKind.FIRE,
            resistance=EnergyKind.WATER, energyType=EnergyKind.WATER, basic=False,
            stage1=True, stage2=False, ex=False, megaEx=True, tera=False,
            aceSpec=False, evolvesFrom="Snover", skills=(), attacks=(),
        ),
    }
    attacks = {900: AttackInfo(attackId=900, name="Surf", text="30 damage.",
                               damage=30, energies=(EnergyKind.WATER, EnergyKind.COLORLESS))}
    return CardDatabase(cards, attacks)


def test_records_round_trip_through_json() -> None:
    db = _db()
    rebuilt = CardDatabase.from_records(json.loads(json.dumps(db.to_records())))

    for cid in (721, 723):
        assert rebuilt.get_card(cid) == db.get_card(cid)
    assert rebuilt.get_attack(900) == db.get_attack(900)
    assert len(rebuilt) == len(db)


def test_enum_and_none_fields_preserved() -> None:
    rebuilt = CardDatabase.from_records(_db().to_records())
    kyogre = rebuilt.get_card(721)
    assert kyogre.weakness is EnergyKind.LIGHTNING and kyogre.resistance is None
    assert kyogre.energyType is EnergyKind.WATER
    mega = rebuilt.get_card(723)
    assert mega.megaEx and mega.evolvesFrom == "Snover"
    assert rebuilt.get_attack(900).energies == (EnergyKind.WATER, EnergyKind.COLORLESS)
