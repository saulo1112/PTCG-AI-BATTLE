"""Leaf evaluator + GameState threat/reserve features (SDK-free)."""

from __future__ import annotations

from ptcg_ai.cards.database import AttackInfo, CardDatabase, CardInfo
from ptcg_ai.decision.evaluator import EvalWeights, Evaluator
from ptcg_ai.observation.models import CardKind, EnergyKind, ParsedPokemon
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

parser = ObservationParser()


def _card(cid: int, **kw) -> CardInfo:
    base = dict(
        cardId=cid, name=f"c{cid}", cardType=CardKind.POKEMON, retreatCost=1,
        hp=100, weakness=None, resistance=None, energyType=EnergyKind.WATER,
        basic=True, stage1=False, stage2=False, ex=False, megaEx=False,
        tera=False, aceSpec=False, evolvesFrom=None, skills=(), attacks=(),
    )
    base.update(kw)
    return CardInfo(**base)  # type: ignore[arg-type]


def _db() -> CardDatabase:
    cards = {
        700: _card(700, hp=120, attacks=(900,)),   # a real attacker
        701: _card(701, hp=70, attacks=(901,)),    # weak hitter
        3: _card(3, cardType=CardKind.BASIC_ENERGY, basic=False),
    }
    attacks = {
        900: AttackInfo(attackId=900, name="Big", text="", damage=120,
                        energies=(EnergyKind.WATER, EnergyKind.WATER)),
        901: AttackInfo(attackId=901, name="Small", text="", damage=30,
                        energies=(EnergyKind.WATER,)),
    }
    return CardDatabase(cards, attacks)


def _pkmn(cid: int, hp: int, energy: int) -> dict:
    return {
        "id": cid, "serial": 1000 + cid, "hp": hp, "maxHp": hp,
        "appearThisTurn": False, "energies": [3] * energy,
        "energyCards": [], "tools": [], "preEvolution": [],
    }


def _player(*, active, bench, prizes, deck=30, hand=5):
    return {
        "active": [active] if active is not None else [],
        "bench": bench,
        "benchMax": 5, "deckCount": deck,
        "discard": [], "prize": [None] * prizes, "handCount": hand, "hand": None,
        "poisoned": False, "burned": False, "asleep": False,
        "paralyzed": False, "confused": False,
    }


def _obs(p0, p1, result=-1, your_index=0):
    return {
        "select": None, "logs": [],
        "current": {
            "turn": 5, "turnActionCount": 0, "yourIndex": your_index, "firstPlayer": 0,
            "supporterPlayed": False, "stadiumPlayed": False,
            "energyAttached": False, "retreated": False, "result": result,
            "stadium": [], "looking": None, "players": [p0, p1],
        },
        "search_begin_input": None,
    }


def _ev() -> Evaluator:
    return Evaluator(EvalWeights(), _db())


# -- GameState features -----------------------------------------------------

def test_reserve_attackers_counts_ready_benchers() -> None:
    p0 = _player(active=_pkmn(700, 120, 2),
                 bench=[_pkmn(700, 120, 2), _pkmn(701, 70, 0), _pkmn(3, 10, 0)],
                 prizes=6)
    gs = GameState.build(parser.parse(_obs(p0, _player(active=_pkmn(700, 120, 0), bench=[], prizes=6))), _db())
    # bench: 700 with 2 energy (payable now), 701 with 0 (needs 1, within 2),
    # 3 is basic energy (no attack) -> 2 reserves.
    assert gs.reserve_attackers(gs.me) == 2


def test_max_threat_effective_damage_and_payability() -> None:
    p0 = _player(active=_pkmn(700, 120, 2), bench=[], prizes=6)
    p1 = _player(active=_pkmn(701, 70, 0), bench=[], prizes=6)
    gs = GameState.build(parser.parse(_obs(p0, p1)), _db())
    # my 700 has 2 energy -> attack 900 (120 dmg) is payable now.
    assert gs.max_threat(gs.my_active, gs.opp_active) == 120
    # their 701 has 0 energy -> not payable now (threat 0); +1 energy -> 30.
    assert gs.max_threat(gs.opp_active, gs.my_active, extra_energy=0) == 0
    assert gs.max_threat(gs.opp_active, gs.my_active, extra_energy=1) == 30


def test_max_threat_zero_when_active_missing() -> None:
    gs = GameState.build(parser.parse(_obs(_player(active=None, bench=[], prizes=6),
                                           _player(active=_pkmn(700, 120, 2), bench=[], prizes=6))), _db())
    assert gs.max_threat(gs.my_active, gs.opp_active) == 0


# -- Evaluator --------------------------------------------------------------

def test_antisymmetry_nonterminal() -> None:
    # player0 clearly ahead: fewer own prizes left, more bench, more energy.
    p0 = _player(active=_pkmn(700, 120, 2), bench=[_pkmn(700, 120, 2)], prizes=2, deck=30, hand=6)
    p1 = _player(active=_pkmn(701, 70, 1), bench=[], prizes=5, deck=20, hand=3)
    obs0 = parser.parse(_obs(p0, p1, your_index=0))
    ev = _ev()
    v0 = ev.value(obs0, 0)
    v1 = ev.value(obs0, 1)
    assert v0 > 0                      # ahead from seat 0
    assert abs(v0 + v1) < 1e-9         # perfectly antisymmetric


def test_terminal_dominates() -> None:
    p0 = _player(active=_pkmn(700, 120, 2), bench=[], prizes=1, deck=30)
    p1 = _player(active=_pkmn(701, 70, 0), bench=[], prizes=5, deck=30)
    ev = _ev()
    win = parser.parse(_obs(p0, p1, result=0))
    assert ev.value(win, 0) == 1.0
    assert ev.value(win, 1) == -1.0
    draw = parser.parse(_obs(p0, p1, result=2))
    assert ev.value(draw, 0) == 0.0 and ev.value(draw, 1) == 0.0


def test_prize_lead_raises_value() -> None:
    ev = _ev()
    base_p0 = _player(active=_pkmn(700, 120, 2), bench=[], prizes=4)
    opp = _player(active=_pkmn(701, 70, 1), bench=[], prizes=4)
    v_even = ev.value(parser.parse(_obs(base_p0, opp)), 0)
    ahead = _player(active=_pkmn(700, 120, 2), bench=[], prizes=1)  # took 3 prizes
    v_ahead = ev.value(parser.parse(_obs(ahead, opp)), 0)
    assert v_ahead > v_even


def test_extra_bencher_raises_value() -> None:
    ev = _ev()
    opp = _player(active=_pkmn(701, 70, 1), bench=[], prizes=5)
    thin = ev.value(parser.parse(_obs(_player(active=_pkmn(700, 120, 2), bench=[], prizes=5), opp)), 0)
    thick = ev.value(parser.parse(_obs(
        _player(active=_pkmn(700, 120, 2), bench=[_pkmn(700, 120, 2)], prizes=5), opp)), 0)
    assert thick > thin


def test_incoming_ko_threat_lowers_value() -> None:
    ev = _ev()
    # my active has 70 hp; opponent 700 with 2 energy hits for 120 (lethal).
    opp_threat = _player(active=_pkmn(700, 120, 2), bench=[], prizes=5)
    safe = _player(active=_pkmn(701, 70, 1), bench=[], prizes=5)  # can't be reached
    me = _player(active=_pkmn(701, 70, 1), bench=[], prizes=5)
    v_threatened = ev.value(parser.parse(_obs(me, opp_threat)), 0)
    v_safe = ev.value(parser.parse(_obs(me, safe)), 0)
    assert v_threatened < v_safe


def test_none_active_does_not_crash() -> None:
    ev = _ev()
    p0 = _player(active=None, bench=[_pkmn(700, 120, 0)], prizes=5)
    p1 = _player(active=_pkmn(701, 70, 1), bench=[], prizes=5)
    v = ev.value(parser.parse(_obs(p0, p1)), 0)
    assert -1.0 <= v <= 1.0


def test_no_cards_degrades() -> None:
    ev = Evaluator(EvalWeights(), None)
    p0 = _player(active=_pkmn(700, 120, 2), bench=[], prizes=3)
    p1 = _player(active=_pkmn(701, 70, 0), bench=[], prizes=5)
    v = ev.value(parser.parse(_obs(p0, p1)), 0)
    assert -1.0 <= v <= 1.0  # prize lead still scored without card data
