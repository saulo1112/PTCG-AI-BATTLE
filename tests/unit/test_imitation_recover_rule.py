"""M27 forced-recovery pre-emption in ImitationPolicy (SDK-free).

The rule must fire ONLY when the clock Pokémon is absent from play AND hand yet
sits in discard and the recover Item is a legal play — and must be a strict no-op
when unconfigured (every existing agent) or when a clock is still available, so it
never perturbs normal learned play. These tests pin exactly that contract.
"""

from __future__ import annotations

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.imitation import deck_profiles as DP
from ptcg_ai.imitation.policy import ImitationPolicy
from ptcg_ai.observation.models import CardKind, EnergyKind, OptionKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

GRASS = int(EnergyKind.GRASS)
CLOCK = 756       # stand-in for Mega Kangaskhan (the win condition)
RECOVER = 1097    # stand-in for Night Stretcher (the recover Item)
FILLER = 345      # a non-clock Pokémon that can hold the Active spot


def _card(cid, ctype, *, basic=True, hp=120):
    return {
        "cardId": cid, "name": f"c{cid}", "cardType": int(ctype), "retreatCost": 1, "hp": hp,
        "weakness": None, "resistance": None, "energyType": GRASS,
        "basic": basic, "stage1": False, "stage2": False, "ex": False, "megaEx": False,
        "tera": False, "aceSpec": False, "evolvesFrom": None, "skills": [], "attacks": [],
    }


def _cards() -> CardDatabase:
    return CardDatabase.from_records({
        "cards": [
            _card(CLOCK, CardKind.POKEMON, hp=300),
            _card(FILLER, CardKind.POKEMON, hp=150),
            _card(9999, CardKind.POKEMON, hp=200),
            _card(RECOVER, CardKind.ITEM, basic=False),
        ],
        "attacks": [],
    })


def _pokemon(cid, hp=150):
    return {
        "id": cid, "serial": cid * 10, "hp": hp, "maxHp": hp, "appearThisTurn": False,
        "energies": [], "energyCards": [], "preEvolution": [], "tools": [],
    }


def _card_ref(cid, serial):
    return {"id": cid, "serial": serial, "playerIndex": 0}


def _obs(*, active_id, hand_ids, discard_ids, play_recover=True):
    """A MAIN observation. The first option is 'play the hand card at index 0'."""
    hand = [_card_ref(c, 100 + i) for i, c in enumerate(hand_ids)]
    discard = [_card_ref(c, 200 + i) for i, c in enumerate(discard_ids)]
    player = {
        "active": [_pokemon(active_id)] if active_id is not None else [None],
        "bench": [], "benchMax": 5, "deckCount": 30, "discard": discard,
        "prize": [None] * 6, "handCount": len(hand), "hand": hand,
        "poisoned": False, "burned": False, "asleep": False, "paralyzed": False, "confused": False,
    }
    opp = {
        "active": [_pokemon(9999, hp=200)], "bench": [], "benchMax": 5, "deckCount": 30,
        "discard": [], "prize": [None] * 6, "handCount": 4, "hand": None,
        "poisoned": False, "burned": False, "asleep": False, "paralyzed": False, "confused": False,
    }
    # option 0 = play hand[0]; option 1 = END. If play_recover, hand[0] is the recover Item.
    options = [{"type": int(OptionKind.PLAY), "index": 0}, {"type": int(OptionKind.END)}]
    return {
        "select": {"type": 0, "context": int(SelectContextKind.MAIN), "minCount": 1, "maxCount": 1,
                   "remainDamageCounter": 0, "remainEnergyCost": 0, "option": options,
                   "deck": None, "contextCard": None, "effect": None},
        "logs": [], "search_begin_input": "AAAA",
        "current": {"turn": 6, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
                    "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": False,
                    "retreated": False, "result": -1, "stadium": [], "looking": None,
                    "players": [player, opp]},
    }


def _payload(recover_rule=True):
    p = {"profile": "TR_650", "feature_dim": DP.TR_650.feature_dim, "contexts": {}}
    if recover_rule:
        p["recover_rule"] = {"recover_id": RECOVER, "clock_id": CLOCK}
    return p


def _fire_index(pol, raw, cards):
    obs = ObservationParser().parse(raw)
    gs = GameState.build(obs, cards)
    return pol._forced_recovery(obs.select, obs, gs)


def test_fires_when_clock_lost_and_recoverable() -> None:
    cards = _cards()
    pol = ImitationPolicy(_payload())
    raw = _obs(active_id=FILLER, hand_ids=[RECOVER], discard_ids=[CLOCK])
    assert _fire_index(pol, raw, cards) == 0  # play the recover Item
    # end-to-end: choose() returns exactly that option
    obs = ObservationParser().parse(raw)
    assert pol.choose(DecisionContext(raw=raw, observation=obs, cards=cards)) == [0]
    assert pol._recover_used == 1


def test_noop_when_clock_still_in_play() -> None:
    cards = _cards()
    pol = ImitationPolicy(_payload())
    raw = _obs(active_id=CLOCK, hand_ids=[RECOVER], discard_ids=[CLOCK])
    assert _fire_index(pol, raw, cards) is None


def test_noop_when_clock_in_hand() -> None:
    cards = _cards()
    pol = ImitationPolicy(_payload())
    raw = _obs(active_id=FILLER, hand_ids=[RECOVER, CLOCK], discard_ids=[CLOCK])
    assert _fire_index(pol, raw, cards) is None


def test_noop_when_no_clock_in_discard() -> None:
    cards = _cards()
    pol = ImitationPolicy(_payload())
    raw = _obs(active_id=FILLER, hand_ids=[RECOVER], discard_ids=[FILLER])
    assert _fire_index(pol, raw, cards) is None


def test_default_off_without_rule() -> None:
    cards = _cards()
    pol = ImitationPolicy(_payload(recover_rule=False))
    assert pol._recover_id is None
    raw = _obs(active_id=FILLER, hand_ids=[RECOVER], discard_ids=[CLOCK])
    assert _fire_index(pol, raw, cards) is None
