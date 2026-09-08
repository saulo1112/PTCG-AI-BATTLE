"""RuleBasedPolicy (rung 4): bench-first energy + retreat, and the GameState
energy-cost math it relies on."""

from __future__ import annotations

from typing import Any

from ptcg_ai.cards.database import AttackInfo, CardDatabase, CardInfo
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.decision.rule_based import RuleBasedPolicy
from ptcg_ai.observation.models import CardKind, EnergyKind, ParsedPokemon
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

parser = ObservationParser()

W, C = EnergyKind.WATER, EnergyKind.COLORLESS


def _card(cid: int, **kw: Any) -> CardInfo:
    base: dict[str, Any] = dict(
        cardId=cid, name=f"card{cid}", cardType=CardKind.POKEMON, retreatCost=1,
        hp=120, weakness=None, resistance=None, energyType=EnergyKind.WATER,
        basic=True, stage1=False, stage2=False, ex=False, megaEx=False,
        tera=False, aceSpec=False, evolvesFrom=None, skills=(), attacks=(),
    )
    base.update(kw)
    return CardInfo(**base)


def _db() -> CardDatabase:
    cards = {
        # Active attacker: best attack Surf costs W,W,C=130.
        721: _card(721, hp=150, attacks=(900,)),
        # Bench attacker: Tackle costs W,C,C=90.
        731: _card(731, hp=110, attacks=(901,)),
        800: _card(800, hp=120, weakness=EnergyKind.WATER),
        3: _card(3, cardType=CardKind.BASIC_ENERGY, basic=False),
    }
    attacks = {
        900: AttackInfo(attackId=900, name="Surf", text="", damage=130, energies=(W, W, C)),
        901: AttackInfo(attackId=901, name="Tackle", text="", damage=90, energies=(W, C, C)),
    }
    return CardDatabase(cards, attacks)


# -- GameState energy math -------------------------------------------------

def test_energy_pays_colorless_is_wild() -> None:
    assert GameState.energy_pays((W, W, W), (W, W, C)) is True
    assert GameState.energy_pays((W, W), (W, W, C)) is False   # short one
    assert GameState.energy_pays((W,), (C,)) is True           # any pays colorless
    assert GameState.energy_pays((), (C,)) is False


def test_wants_energy_tracks_best_attack() -> None:
    gs = GameState(state=None, cards=_db(), me=None, opponent=None,
                   my_active=None, opp_active=None, my_prizes_left=6, opp_prizes_left=6,
                   my_active_prize_value=1, opp_active_prize_value=1, my_bench_count=0,
                   bench_room=5, has_bench_insurance=False, hand_size=0, my_deck_count=0,
                   energy_attached=False, supporter_played=False, retreated=False)
    two = ParsedPokemon(id=721, serial=1, hp=150, maxHp=150, appearThisTurn=False,
                        energies=(W, W), energyCards=(), tools=(), preEvolution=())
    three = ParsedPokemon(id=721, serial=1, hp=150, maxHp=150, appearThisTurn=False,
                          energies=(W, W, W), energyCards=(), tools=(), preEvolution=())
    assert gs.wants_energy(two) is True    # WWC=130 not yet payable with WW
    assert gs.wants_energy(three) is False  # now payable -> charge someone else


# -- bench-first attach ----------------------------------------------------

def _pokemon(cid: int, energies: list[int]) -> dict[str, Any]:
    return {"id": cid, "serial": cid, "hp": 120, "maxHp": 120, "appearThisTurn": False,
            "energies": energies, "energyCards": [], "tools": [], "preEvolution": []}


def _raw(active_energies: list[int], bench_energies: list[int],
         options: list[dict[str, Any]], *, energy_attached: bool = False) -> dict[str, Any]:
    me = {"active": [_pokemon(721, active_energies)],
          "bench": [_pokemon(731, bench_energies)],
          "benchMax": 5, "deckCount": 40, "discard": [], "prize": [None] * 6,
          "handCount": 3, "hand": [{"id": 3, "serial": 99, "playerIndex": 0}] * 3,
          "poisoned": False, "burned": False, "asleep": False, "paralyzed": False,
          "confused": False}
    opp = dict(me, hand=None, handCount=4, active=[_pokemon(800, [])], bench=[])
    return {"select": {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
                       "remainDamageCounter": 0, "remainEnergyCost": 0, "option": options,
                       "deck": None, "contextCard": None, "effect": None},
            "logs": [], "current": {"turn": 5, "turnActionCount": 0, "yourIndex": 0,
                                    "firstPlayer": 0, "supporterPlayed": False,
                                    "stadiumPlayed": False, "energyAttached": energy_attached,
                                    "retreated": False, "result": -1, "stadium": [],
                                    "looking": None, "players": [me, opp]},
            "search_begin_input": "AAAA"}


ATTACH_ACTIVE = {"type": 8, "area": 2, "index": 0, "inPlayArea": 4, "inPlayIndex": 0}
ATTACH_BENCH = {"type": 8, "area": 2, "index": 0, "inPlayArea": 5, "inPlayIndex": 0}
END = {"type": 14}


def _choose(raw: dict[str, Any]) -> list[int]:
    ctx = DecisionContext(raw=raw, observation=parser.parse(raw), cards=_db())
    return RuleBasedPolicy().choose(ctx)


def test_charges_active_until_ready() -> None:
    # Active has WW (needs WWC); charge the Active, not the bench.
    raw = _raw([3, 3], [], [ATTACH_BENCH, ATTACH_ACTIVE, END])
    assert _choose(raw) == [1]  # ATTACH_ACTIVE


def test_bench_first_once_active_ready() -> None:
    # Active has WWW (Surf WWC payable) -> pre-charge the empty bench attacker.
    raw = _raw([3, 3, 3], [], [ATTACH_ACTIVE, ATTACH_BENCH, END])
    assert _choose(raw) == [1]  # ATTACH_BENCH


def test_retreat_stranded_active_for_ready_bencher() -> None:
    # Active has no energy (can't attack); bench has WCC (Tackle payable).
    # RETREAT offered -> swap in the ready attacker.
    RETREAT = {"type": 12}
    raw = _raw([], [3, 3, 3], [RETREAT, END], energy_attached=True)
    assert _choose(raw) == [0]  # RETREAT


def test_no_retreat_when_active_can_attack() -> None:
    RETREAT = {"type": 12}
    ATTACK = {"type": 13, "attackId": 900}
    # Active has WWW (can Surf) -> take the attack path, never retreat.
    raw = _raw([3, 3, 3], [3, 3, 3], [RETREAT, ATTACK, END], energy_attached=True)
    assert _choose(raw) == [1]  # ATTACK (lethal/tempo), not retreat


def test_reproduces_greedy_when_no_card_data() -> None:
    # Without a CardDatabase, rung 4 can't compute readiness; must still be legal.
    raw = _raw([3], [], [ATTACH_ACTIVE, END])
    ctx = DecisionContext(raw=raw, observation=parser.parse(raw), cards=None)
    action = RuleBasedPolicy().choose(ctx)
    assert action and all(0 <= i < 2 for i in action)
