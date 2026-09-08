"""GameState derivation and KO math (SDK-free: cards built from literals)."""

from __future__ import annotations

from ptcg_ai.cards.database import AttackInfo, CardDatabase, CardInfo
from ptcg_ai.observation.models import CardKind, EnergyKind, ParsedPokemon
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState
from tests.conftest import make_raw_obs

parser = ObservationParser()


def _card(cid: int, **kw: object) -> CardInfo:
    base = dict(
        cardId=cid, name=f"card{cid}", cardType=CardKind.POKEMON, retreatCost=1,
        hp=100, weakness=None, resistance=None, energyType=EnergyKind.WATER,
        basic=True, stage1=False, stage2=False, ex=False, megaEx=False,
        tera=False, aceSpec=False, evolvesFrom=None, skills=(), attacks=(),
    )
    base.update(kw)
    return CardInfo(**base)  # type: ignore[arg-type]


def _db() -> CardDatabase:
    cards = {
        721: _card(721, energyType=EnergyKind.WATER, hp=60, attacks=(900,)),
        800: _card(800, hp=90, weakness=EnergyKind.WATER),
        801: _card(801, hp=40, resistance=EnergyKind.WATER),
        802: _card(802, hp=200, ex=True),
        803: _card(803, hp=330, megaEx=True),
    }
    attacks = {900: AttackInfo(attackId=900, name="Surf", text="", damage=50,
                               energies=(EnergyKind.WATER, EnergyKind.WATER))}
    return CardDatabase(cards, attacks)


def _pokemon(cid: int, hp: int) -> ParsedPokemon:
    return ParsedPokemon(
        id=cid, serial=cid, hp=hp, maxHp=hp, appearThisTurn=False,
        energies=(), energyCards=(), tools=(), preEvolution=(),
    )


def test_build_derives_perspective_and_prizes() -> None:
    raw = make_raw_obs()
    gs = GameState.build(parser.parse(raw), _db())
    assert gs.my_active is not None and gs.my_active.id == 721
    assert gs.my_prizes_left == 6 and gs.opp_prizes_left == 6
    assert gs.bench_room == 5 and not gs.has_bench_insurance
    assert not gs.energy_attached


def test_weakness_doubles_damage_and_is_lethal() -> None:
    gs = GameState.build(parser.parse(make_raw_obs()), _db())
    attacker = _pokemon(721, 60)
    defender = _pokemon(800, 90)  # weak to WATER
    assert gs.attack_damage(gs.cards.get_attack(900), attacker, defender) == 100
    assert gs.is_lethal(gs.cards.get_attack(900), attacker, defender)


def test_resistance_reduces_and_blocks_lethal() -> None:
    gs = GameState.build(parser.parse(make_raw_obs()), _db())
    attacker = _pokemon(721, 60)
    defender = _pokemon(801, 40)  # resists WATER (−30)
    assert gs.attack_damage(gs.cards.get_attack(900), attacker, defender) == 20
    assert not gs.is_lethal(gs.cards.get_attack(900), attacker, defender)


def test_prize_values_for_ex_and_mega() -> None:
    gs = GameState.build(parser.parse(make_raw_obs()), _db())
    assert gs._prize_value(_pokemon(802, 200), gs.cards) == 2
    assert gs._prize_value(_pokemon(803, 330), gs.cards) == 3
    assert gs._prize_value(_pokemon(721, 60), gs.cards) == 1


def test_build_without_cards_degrades_safely() -> None:
    gs = GameState.build(parser.parse(make_raw_obs()), None)
    assert gs.my_active is not None
    assert gs.opp_active_prize_value == 1  # no card data → default value
