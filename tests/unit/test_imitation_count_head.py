"""M42 count head in ImitationPolicy (SDK-free).

`_top_k` takes min(maxCount, n) — the teacher's rule everywhere except
SETUP_BENCH_POKEMON, where Yushin declines on 37.4% of decisions and the champion
declined 0 of 45 on its own ladder replays. A count head predicts k instead.

The contract pinned here: absent ⇒ strict no-op (every existing agent byte-for-byte
unchanged); present ⇒ k comes from the head but is CLAMPED into the legal band, so a
mispredicting head can never emit an illegal move.
"""

from __future__ import annotations

import pytest

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.imitation import deck_profiles as DP
from ptcg_ai.imitation.policy import _COUNT_EXTRA, ImitationPolicy, load_weights
from ptcg_ai.observation.models import CardKind, EnergyKind, OptionKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser

GRASS = int(EnergyKind.GRASS)
BASIC_A, BASIC_B = 741, 305
CTX = "SETUP_BENCH_POKEMON"
DIM = DP.TR_650.feature_dim


def _card(cid, ctype, *, basic=True, hp=120):
    return {
        "cardId": cid, "name": f"c{cid}", "cardType": int(ctype), "retreatCost": 1, "hp": hp,
        "weakness": None, "resistance": None, "energyType": GRASS,
        "basic": basic, "stage1": False, "stage2": False, "ex": False, "megaEx": False,
        "tera": False, "aceSpec": False, "evolvesFrom": None, "skills": [], "attacks": [],
    }


def _cards() -> CardDatabase:
    return CardDatabase.from_records({
        "cards": [_card(BASIC_A, CardKind.POKEMON), _card(BASIC_B, CardKind.POKEMON),
                  _card(9999, CardKind.POKEMON, hp=200)],
        "attacks": [],
    })


def _pokemon(cid, hp=150):
    return {"id": cid, "serial": cid * 10, "hp": hp, "maxHp": hp, "appearThisTurn": False,
            "energies": [], "energyCards": [], "preEvolution": [], "tools": []}


def _obs(hand_ids, *, min_count=0, max_count=2):
    hand = [{"id": c, "serial": 100 + i, "playerIndex": 0} for i, c in enumerate(hand_ids)]
    player = {
        "active": [_pokemon(9999)], "bench": [], "benchMax": 5, "deckCount": 30, "discard": [],
        "prize": [None] * 6, "handCount": len(hand), "hand": hand,
        "poisoned": False, "burned": False, "asleep": False, "paralyzed": False, "confused": False,
    }
    opp = dict(player, hand=None, handCount=4)
    options = [{"type": int(OptionKind.CARD), "index": i} for i in range(len(hand_ids))]
    return {
        "select": {"type": 1, "context": int(SelectContextKind.SETUP_BENCH_POKEMON),
                   "minCount": min_count, "maxCount": max_count, "remainDamageCounter": 0,
                   "remainEnergyCost": 0, "option": options, "deck": None,
                   "contextCard": None, "effect": None},
        "logs": [], "search_begin_input": "AAAA",
        "current": {"turn": 1, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
                    "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": False,
                    "retreated": False, "result": -1, "stadium": [], "looking": None,
                    "players": [player, opp]},
    }


def _payload(*, count_head=None):
    p = {"profile": "TR_650", "feature_dim": DIM,
         "contexts": {CTX: {"kind": "linear", "w": [0.0] * DIM}}}
    if count_head is not None:
        p["count_heads"] = {CTX: count_head}
    return p


def _head_forcing(k, *, classes=(0, 1, 2)):
    """A head whose argmax is always the class equal to `k` (all-zero W, biased b)."""
    return {"classes": list(classes),
            "W": [[0.0] * (DIM + _COUNT_EXTRA) for _ in classes],
            "b": [10.0 if c == k else 0.0 for c in classes]}


def _choose(pol, raw, cards):
    return pol.choose(DecisionContext(raw=raw, observation=ObservationParser().parse(raw),
                                      cards=cards))


def test_absent_count_head_is_a_strict_noop() -> None:
    """No count_heads key ⇒ the shipped `_top_k` rule, i.e. take the cap."""
    cards = _cards()
    raw = _obs([BASIC_A, BASIC_B], min_count=0, max_count=2)
    pol = ImitationPolicy(_payload())
    assert len(_choose(pol, raw, cards)) == 2
    assert pol._count_used == 0


def test_count_head_can_decline_which_top_k_never_could() -> None:
    """The whole point: k=0 on a minCount=0 select — unreachable via ranking."""
    cards = _cards()
    raw = _obs([BASIC_A, BASIC_B], min_count=0, max_count=2)
    pol = ImitationPolicy(_payload(count_head=_head_forcing(0)))
    assert _choose(pol, raw, cards) == []
    assert pol._count_used == 1


def test_count_head_picks_a_middle_count() -> None:
    cards = _cards()
    raw = _obs([BASIC_A, BASIC_B], min_count=0, max_count=2)
    pol = ImitationPolicy(_payload(count_head=_head_forcing(1)))
    assert len(_choose(pol, raw, cards)) == 1


@pytest.mark.parametrize("forced,min_count,max_count,expected", [
    (0, 1, 2, 1),   # head says 0 but minCount is 1 -> clamped UP
    (2, 0, 1, 1),   # head says 2 but maxCount is 1 -> clamped DOWN
    (9, 0, 2, 2),   # class outside the offered range -> clamped to n
])
def test_prediction_is_clamped_into_the_legal_band(forced, min_count, max_count, expected) -> None:
    """A mispredicting head must never be able to emit an illegal count."""
    cards = _cards()
    raw = _obs([BASIC_A, BASIC_B], min_count=min_count, max_count=max_count)
    pol = ImitationPolicy(_payload(count_head=_head_forcing(forced, classes=(0, 1, 2, 9))))
    assert len(_choose(pol, raw, cards)) == expected


def test_load_weights_rejects_a_wrong_width_head() -> None:
    bad = _head_forcing(0)
    bad["W"] = [row[:-1] for row in bad["W"]]  # one scalar short
    with pytest.raises(ValueError, match="count_head"):
        load_weights(_payload(count_head=bad))


def test_load_weights_rejects_mismatched_class_count() -> None:
    bad = _head_forcing(0)
    bad["b"] = bad["b"][:-1]
    with pytest.raises(ValueError, match="count_head"):
        load_weights(_payload(count_head=bad))
