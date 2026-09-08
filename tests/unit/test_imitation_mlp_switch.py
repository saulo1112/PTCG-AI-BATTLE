"""M32 Strategy B: the ``mlp_switch`` scorer in ImitationPolicy (SDK-free).

Routes MAIN scoring to a Grimmsnarl-specific specialist ensemble (its OWN
DeckProfile/dim) when the opponent's board carries a detector card id, else
falls back to the general ensemble under the payload's own profile, UNCHANGED
from the non-switch path. These tests pin: routes correctly both ways, the
general model's weights/dim are never touched by the specialist's dim, spec
validation rejects malformed switch payloads, and a runtime failure inside the
switch path still falls back to greedy (never crashes a game) -- same
exception-safety contract as the M27 forced-recovery rule.
"""

from __future__ import annotations

import pytest

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.imitation import deck_profiles as DP
from ptcg_ai.imitation.policy import ImitationPolicy, _validate_spec
from ptcg_ai.observation.models import CardKind, EnergyKind, OptionKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

GRASS = int(EnergyKind.GRASS)
DARK = int(EnergyKind.DARKNESS)
MUNKIDORI = 112          # the real detector id used by ALAKAZAM_SPECIALIST
MY_ACTIVE = 743          # Alakazam, in the ALAKAZAM deck_ids vocabulary
OPP_FILLER = 9999        # a generic opponent Pokemon (no detector match)

ALAKAZAM = DP.ALAKAZAM
SPECIALIST = DP.ALAKAZAM_SPECIALIST


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
            _card(MY_ACTIVE, CardKind.POKEMON, hp=130),
            _card(MUNKIDORI, CardKind.POKEMON, hp=110),
            _card(OPP_FILLER, CardKind.POKEMON, hp=150),
        ],
        "attacks": [],
    })


def _pokemon(cid, *, hp=150, energies=()):
    return {
        "id": cid, "serial": cid * 10, "hp": hp, "maxHp": hp, "appearThisTurn": False,
        "energies": list(energies), "energyCards": [], "preEvolution": [], "tools": [],
    }


def _obs(*, opp_active, opp_bench=()):
    """A MAIN observation with a trivial PLAY/END option pair -- these tests
    check WHICH scorer fires, not attack selection specifics."""
    me = {
        "active": [_pokemon(MY_ACTIVE, hp=130)], "bench": [], "benchMax": 5, "deckCount": 40,
        "discard": [], "prize": [None] * 6, "handCount": 5,
        "hand": [{"id": 1, "serial": 900 + i, "playerIndex": 0} for i in range(5)],
        "poisoned": False, "burned": False, "asleep": False, "paralyzed": False, "confused": False,
    }
    opp = {
        "active": [opp_active], "bench": list(opp_bench), "benchMax": 5, "deckCount": 40,
        "discard": [], "prize": [None] * 6, "handCount": 4, "hand": None,
        "poisoned": False, "burned": False, "asleep": False, "paralyzed": False, "confused": False,
    }
    options = [{"type": int(OptionKind.PLAY), "index": 0}, {"type": int(OptionKind.END)}]
    return {
        "select": {"type": 0, "context": int(SelectContextKind.MAIN), "minCount": 1, "maxCount": 1,
                   "remainDamageCounter": 0, "remainEnergyCost": 0, "option": options,
                   "deck": None, "contextCard": None, "effect": None},
        "logs": [], "search_begin_input": "AAAA",
        "current": {"turn": 4, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
                    "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": False,
                    "retreated": False, "result": -1, "stadium": [], "looking": None,
                    "players": [me, opp]},
    }


def _linear(dim: int, value: float = 0.01) -> list[float]:
    return [value] * dim


def _switch_payload(*, detector_ids=(MUNKIDORI,), general_dim=None, specialist_dim=None) -> dict:
    return {
        "profile": ALAKAZAM.name,
        "feature_dim": ALAKAZAM.feature_dim,
        "contexts": {
            "MAIN": {
                "kind": "mlp_switch",
                "specialist_profile": SPECIALIST.name,
                "detector_ids": list(detector_ids),
                "general": _linear(general_dim if general_dim is not None else ALAKAZAM.feature_dim),
                "specialist": _linear(specialist_dim if specialist_dim is not None else SPECIALIST.feature_dim),
            },
        },
    }


def _decide(pol: ImitationPolicy, raw: dict, cards: CardDatabase):
    obs = ObservationParser().parse(raw)
    return pol.choose(DecisionContext(raw=raw, observation=obs, cards=cards))


def test_switch_routes_to_specialist_when_detector_present() -> None:
    cards = _cards()
    pol = ImitationPolicy(_switch_payload())
    raw = _obs(opp_active=_pokemon(OPP_FILLER, hp=150),
               opp_bench=[_pokemon(MUNKIDORI, hp=110, energies=[DARK])])
    _decide(pol, raw, cards)
    assert pol._route_used == 1
    assert "switch=specialist" in pol.last_note
    assert pol._bc_used == 1
    assert pol._bc_failures == 0


def test_switch_routes_to_general_when_detector_absent() -> None:
    cards = _cards()
    pol = ImitationPolicy(_switch_payload())
    raw = _obs(opp_active=_pokemon(OPP_FILLER, hp=150))
    _decide(pol, raw, cards)
    assert pol._route_used == 0
    assert "switch=general" in pol.last_note
    assert pol._bc_used == 1
    assert pol._bc_failures == 0


def test_switch_off_board_after_detector_leaves() -> None:
    """Transient by design (Attempt #1's bug was a permanent flag): the SAME
    policy instance must route back to general once Munkidori is no longer
    on the opponent's board, with no memory of having seen it earlier."""
    cards = _cards()
    pol = ImitationPolicy(_switch_payload())
    raw_on = _obs(opp_active=_pokemon(OPP_FILLER, hp=150),
                  opp_bench=[_pokemon(MUNKIDORI, hp=110, energies=[DARK])])
    _decide(pol, raw_on, cards)
    assert pol._route_used == 1

    raw_off = _obs(opp_active=_pokemon(OPP_FILLER, hp=150))
    _decide(pol, raw_off, cards)
    assert pol._route_used == 1  # unchanged -- did NOT fire again
    assert "switch=general" in pol.last_note


def test_non_switch_payload_unaffected() -> None:
    """A normal (pre-M32) payload never touches the switch machinery."""
    payload = {
        "profile": ALAKAZAM.name, "feature_dim": ALAKAZAM.feature_dim,
        "contexts": {"MAIN": _linear(ALAKAZAM.feature_dim)},
    }
    pol = ImitationPolicy(payload)
    assert pol._switch_profiles == {}
    raw = _obs(opp_active=_pokemon(OPP_FILLER, hp=150),
               opp_bench=[_pokemon(MUNKIDORI, hp=110, energies=[DARK])])
    _decide(pol, raw, cards=_cards())
    assert pol._route_used == 0
    assert "switch" not in pol.last_note


def test_validate_spec_rejects_missing_detector_ids() -> None:
    spec = {
        "kind": "mlp_switch", "specialist_profile": SPECIALIST.name,
        "general": _linear(ALAKAZAM.feature_dim), "specialist": _linear(SPECIALIST.feature_dim),
    }
    with pytest.raises(ValueError, match="mlp_switch"):
        _validate_spec("MAIN", spec, ALAKAZAM.feature_dim)


def test_validate_spec_rejects_missing_specialist_profile() -> None:
    spec = {
        "kind": "mlp_switch", "detector_ids": [MUNKIDORI],
        "general": _linear(ALAKAZAM.feature_dim), "specialist": _linear(SPECIALIST.feature_dim),
    }
    with pytest.raises(ValueError, match="mlp_switch"):
        _validate_spec("MAIN", spec, ALAKAZAM.feature_dim)


def test_validate_spec_checks_general_dim() -> None:
    with pytest.raises(ValueError, match="linear length"):
        _validate_spec("MAIN", _switch_payload(general_dim=ALAKAZAM.feature_dim - 1)
                        ["contexts"]["MAIN"], ALAKAZAM.feature_dim)


def test_validate_spec_checks_specialist_dim() -> None:
    with pytest.raises(ValueError, match="linear length"):
        _validate_spec("MAIN", _switch_payload(specialist_dim=SPECIALIST.feature_dim - 1)
                        ["contexts"]["MAIN"], ALAKAZAM.feature_dim)


def test_switch_unknown_detector_id_just_never_fires() -> None:
    """A detector id that never matches anything is a safe no-op (routes
    general), not an error -- garbage-in-detector_ids should never crash."""
    payload = _switch_payload(detector_ids=[-1])
    pol = ImitationPolicy(payload)
    raw = _obs(opp_active=_pokemon(OPP_FILLER, hp=150),
               opp_bench=[_pokemon(MUNKIDORI, hp=110, energies=[DARK])])
    chosen = _decide(pol, raw, cards=_cards())
    assert chosen
    assert pol._route_used == 0
    assert pol._bc_failures == 0


def test_switch_runtime_error_falls_back_to_greedy() -> None:
    """A specialist spec that passes construction/validation but throws at
    actual scoring time (a corrupt member missing ``b2``, which `_validate_spec`
    does not check) must fall back to greedy -- the same exception-safety
    contract every other scoring path in ImitationPolicy already has."""
    payload = _switch_payload()
    payload["contexts"]["MAIN"]["specialist"] = {
        "kind": "mlp",
        "W1": [[0.0] for _ in range(SPECIALIST.feature_dim)],
        "b1": [0.0],
        "w2": [0.0],
        # "b2" deliberately omitted -- only blows up inside _mlp_forward
    }
    pol = ImitationPolicy(payload)
    raw = _obs(opp_active=_pokemon(OPP_FILLER, hp=150),
               opp_bench=[_pokemon(MUNKIDORI, hp=110, energies=[DARK])])
    chosen = _decide(pol, raw, cards=_cards())
    assert chosen  # greedy still returns a legal choice (fell all the way through)
    assert pol._bc_failures == 1  # the switch path DID raise and got caught
