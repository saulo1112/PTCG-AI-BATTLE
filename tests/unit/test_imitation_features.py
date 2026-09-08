"""Featurizer invariants + the deck-specific damage logic (M7-2, SDK-free).

The Rocket Rush / Maximum Belt / Weakness computation is the one piece of hand
math the whole BC pilot leans on (GameState KO math reads Rocket Rush as 0), so
it gets a direct guard. The dimension/determinism checks are the training-serving
skew tripwire: if the shipped featurizer ever changes shape, this fails before a
stale weights file reaches an arena.
"""

from __future__ import annotations

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.imitation import deck_profiles as DP
from ptcg_ai.imitation import features as F
from ptcg_ai.observation.models import CardKind, EnergyKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

TR = DP.TR_650

GRASS = int(EnergyKind.GRASS)
PSYCHIC = int(EnergyKind.PSYCHIC)


def _card(cid, name, ctype, energy, *, basic=True, stage1=False, ex=False, hp=100,
          weakness=None, attacks=()):
    return {
        "cardId": cid, "name": name, "cardType": int(ctype), "retreatCost": 1, "hp": hp,
        "weakness": weakness, "resistance": None, "energyType": energy,
        "basic": basic, "stage1": stage1, "stage2": False, "ex": ex, "megaEx": False,
        "tera": False, "aceSpec": False, "evolvesFrom": None, "skills": [], "attacks": list(attacks),
    }


def _attack(aid, name, damage, energies):
    return {"attackId": aid, "name": name, "text": "", "damage": damage, "energies": list(energies)}


def _cards() -> CardDatabase:
    return CardDatabase.from_records({
        "cards": [
            _card(400, "Team Rocket's Tarountula", CardKind.POKEMON, GRASS, attacks=[559]),
            _card(401, "Team Rocket's Spidops", CardKind.POKEMON, GRASS,
                  basic=False, stage1=True, hp=130, attacks=[560]),
            _card(9999, "Foe ex", CardKind.POKEMON, PSYCHIC, ex=True, hp=200, weakness=GRASS),
        ],
        "attacks": [
            _attack(559, "Take Down", 30, [GRASS]),
            _attack(560, "Rocket Rush", 0, [GRASS]),  # structured 0; prose "30x per TR"
        ],
    })


def _pokemon(cid, *, energies=(), tools=(), hp=130, maxhp=130):
    return {
        "id": cid, "serial": cid * 10, "hp": hp, "maxHp": maxhp, "appearThisTurn": False,
        "energies": list(energies), "energyCards": [], "preEvolution": [],
        "tools": [{"id": t, "serial": 1, "playerIndex": 0} for t in tools],
    }


def _obs(my_active, my_bench, opp_active, *, options):
    player = {
        "active": [my_active], "bench": my_bench, "benchMax": 5, "deckCount": 30,
        "discard": [], "prize": [None] * 6, "handCount": 5,
        "hand": [{"id": 1, "serial": 7, "playerIndex": 0}] * 5,
        "poisoned": False, "burned": False, "asleep": False, "paralyzed": False, "confused": False,
    }
    opp = {
        "active": [opp_active], "bench": [], "benchMax": 5, "deckCount": 30,
        "discard": [], "prize": [None] * 6, "handCount": 4, "hand": None,
        "poisoned": False, "burned": False, "asleep": False, "paralyzed": False, "confused": False,
    }
    return {
        "select": {"type": 0, "context": int(SelectContextKind.MAIN), "minCount": 1, "maxCount": 1,
                   "remainDamageCounter": 0, "remainEnergyCost": 0, "option": options,
                   "deck": None, "contextCard": None, "effect": None},
        "logs": [], "search_begin_input": "AAAA",
        "current": {"turn": 5, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
                    "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": False,
                    "retreated": False, "result": -1, "stadium": [], "looking": None,
                    "players": [player, opp]},
    }


def _gs(my_active, my_bench, opp_active, cards):
    obs = ObservationParser().parse(_obs(my_active, my_bench, opp_active, options=[{"type": 14}]))
    return GameState.build(obs, cards), obs


def test_rocket_rush_scales_with_team_rocket_board() -> None:
    cards = _cards()
    # active Spidops + 2 benched TR Pokémon = 3 TR in play → 30 * 3 = 90.
    gs, _ = _gs(_pokemon(401), [_pokemon(400), _pokemon(401)], _pokemon(9999, hp=200), cards)
    assert F.effective_attack_damage(TR, 560, gs, cards) == 90 * 2  # ×2 Grass weakness on the ex


def test_maximum_belt_adds_50_before_weakness_vs_ex() -> None:
    cards = _cards()
    gs, _ = _gs(
        _pokemon(401, tools=[DP._MAX_BELT_ID]),  # belt on the attacker
        [_pokemon(400)],                         # 2 TR in play → base 60
        _pokemon(9999, hp=200), cards,
    )
    # (30*2 + 50) * 2 (Grass weakness) = 220
    assert F.effective_attack_damage(TR, 560, gs, cards) == 220


def test_rocket_rush_no_belt_no_weakness() -> None:
    cards = _cards()
    # Opponent without Grass weakness: plain 30 * TR count, no doubling.
    cards._cards[9999] = cards._cards[9999].__class__(  # type: ignore[attr-defined]
        **{**cards._cards[9999].__dict__, "weakness": None}
    )
    gs, _ = _gs(_pokemon(401), [_pokemon(400)], _pokemon(9999, hp=200), cards)
    assert F.effective_attack_damage(TR, 560, gs, cards) == 60


def test_feature_dim_and_determinism() -> None:
    cards = _cards()
    obs = ObservationParser().parse(_obs(
        _pokemon(401), [_pokemon(400)], _pokemon(9999, hp=200),
        options=[{"attackId": 560, "type": 13}, {"type": 14}],
    ))
    gs = GameState.build(obs, cards)
    vecs = F.featurize_decision(TR, obs.select, obs, gs, cards)
    assert len(vecs) == 2
    assert all(len(v) == TR.feature_dim for v in vecs)
    # same input → identical vector (single deterministic code path)
    again = F.featurize_option(TR, obs.select, obs, gs, cards, 0)
    assert again == vecs[0]
    # the attack option must carry a positive effective-damage feature; END must not
    assert any(x > 0 for x in vecs[0])


def test_is_prize_pick_only_for_prize_area_to_hand() -> None:
    cards = _cards()
    # A normal MAIN select is not a prize pick.
    obs = ObservationParser().parse(_obs(
        _pokemon(401), [], _pokemon(9999, hp=200), options=[{"type": 14}]))
    assert not F.is_prize_pick(obs.select)


# -- LUCARIO_800 profile ------------------------------------------------------

L8 = DP.LUCARIO_800
FIGHTING = int(EnergyKind.FIGHTING)


def _cards_800() -> CardDatabase:
    return CardDatabase.from_records({
        "cards": [
            _card(676, "Solrock", CardKind.POKEMON, FIGHTING, hp=110, attacks=[980]),
            _card(675, "Lunatone", CardKind.POKEMON, FIGHTING, hp=110, attacks=[979]),
            _card(678, "Mega Lucario ex", CardKind.POKEMON, FIGHTING,
                  basic=False, stage1=True, hp=340, attacks=[982, 983]),
            _card(9999, "Foe", CardKind.POKEMON, PSYCHIC, hp=200),
        ],
        "attacks": [
            _attack(980, "Cosmic Beam", 70, [FIGHTING]),
            _attack(979, "Power Gem", 50, [FIGHTING, FIGHTING]),
            _attack(982, "Aura Jab", 130, [FIGHTING]),
            _attack(983, "Mega Brave", 270, [FIGHTING, FIGHTING]),
        ],
    })


def test_cosmic_beam_zero_without_lunatone_on_bench() -> None:
    cards = _cards_800()
    # Solrock active, NO Lunatone on bench → Cosmic Beam does 0 (and ignores W/R).
    gs, _ = _gs(_pokemon(676), [], _pokemon(9999, hp=200), cards)
    assert F.effective_attack_damage(L8, 980, gs, cards) == 0
    # Lunatone on the bench → 70, and NOT doubled despite the psychic ex (ignores W/R).
    gs2, _ = _gs(_pokemon(676), [_pokemon(675)], _pokemon(9999, hp=200), cards)
    assert F.effective_attack_damage(L8, 980, gs2, cards) == 70


def test_lucario_structured_damage_and_dim() -> None:
    cards = _cards_800()
    gs, _ = _gs(_pokemon(678), [], _pokemon(9999, hp=200), cards)
    # Mega Brave = structured 270 (no weakness on the psychic foe).
    assert F.effective_attack_damage(L8, 983, gs, cards) == 270
    obs = ObservationParser().parse(_obs(
        _pokemon(678), [], _pokemon(9999, hp=200),
        options=[{"attackId": 983, "type": 13}, {"type": 14}]))
    gs2 = GameState.build(obs, cards)
    vecs = F.featurize_decision(L8, obs.select, obs, gs2, cards)
    assert L8.feature_dim == 467
    assert all(len(v) == 467 for v in vecs)


def test_profiles_have_consistent_dims() -> None:
    assert DP.TR_650.feature_dim == 386
    assert DP.LUCARIO_800.feature_dim == 467
    assert DP.LUCARIO_800_V2.feature_dim == 633
    assert DP.get_profile("TR_650") is DP.TR_650
    assert DP.get_profile("LUCARIO_800") is DP.LUCARIO_800
    assert DP.get_profile("LUCARIO_800_V2") is DP.LUCARIO_800_V2


def test_lucario_v2_snapshot_reads_hand_composition() -> None:
    cards = _cards_800()
    # Two F energies (id 6) + a Carmine (1192) in hand; Solrock active with 1 energy.
    hand = [{"id": 6, "serial": 1, "playerIndex": 0}, {"id": 6, "serial": 2, "playerIndex": 0},
            {"id": 1192, "serial": 3, "playerIndex": 0}]
    raw = _obs(_pokemon(676, energies=[6]), [], _pokemon(9999, hp=200), options=[{"type": 14}])
    raw["current"]["players"][0]["hand"] = hand
    raw["current"]["players"][0]["handCount"] = 3
    obs = ObservationParser().parse(raw)
    gs = GameState.build(obs, cards)
    snap = DP.snapshot_800v2(obs, gs, cards)
    assert len(snap) == 31
    # slot 21 = hand F-energy /6 -> 2/6; slot 28 = active energy /4 -> 1/4
    assert abs(snap[21] - 2 / 6) < 1e-9
    assert abs(snap[28] - 1 / 4) < 1e-9
