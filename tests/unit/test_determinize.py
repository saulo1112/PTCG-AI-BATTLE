"""Determinizer bookkeeping (SDK-free: parsed fixtures + literal cards)."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from ptcg_ai.cards.database import CardDatabase, CardInfo
from ptcg_ai.observation.models import CardKind, EnergyKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.planning.determinize import Determinizer, DeterminizeError, HiddenInfo

parser = ObservationParser()


def _card(cid: int, basic: bool = True) -> CardInfo:
    return CardInfo(
        cardId=cid, name=f"card{cid}", cardType=CardKind.POKEMON, retreatCost=1,
        hp=100, weakness=None, resistance=None, energyType=EnergyKind.WATER,
        basic=basic, stage1=False, stage2=False, ex=False, megaEx=False,
        tera=False, aceSpec=False, evolvesFrom=None, skills=(), attacks=(),
    )


def _db() -> CardDatabase:
    # 200 = Basic Pokémon, 3 = basic energy (not a Pokémon), 50 = non-basic
    return CardDatabase(
        {200: _card(200, basic=True), 3: _card(3, basic=False), 50: _card(50, basic=False)},
        {},
    )


def _pokemon(cid: int, *, energy=(), tools=(), pre=()) -> dict:
    return {
        "id": cid, "serial": 1000 + cid, "hp": 100, "maxHp": 100,
        "appearThisTurn": False, "energies": [],
        "energyCards": [{"id": e, "serial": 2000 + e, "playerIndex": 0} for e in energy],
        "tools": [{"id": t, "serial": 3000 + t, "playerIndex": 0} for t in tools],
        "preEvolution": [{"id": p, "serial": 4000 + p, "playerIndex": 0} for p in pre],
    }


def _card_ref(cid: int, player: int = 0) -> dict:
    return {"id": cid, "serial": 5000 + cid, "playerIndex": player}


def _visible_count(active: list[int], bench: list[dict], hand: list[int],
                   discard: list[int]) -> int:
    """Count my cards NOT in the deck and NOT in a prize slot (board incl.
    attachments, hand, discard) — used with the 60-card identity
    ``60 = deck + hand + board + discard + prize_slots``."""
    n = len(active) + len(hand) + len(discard)
    for pk in bench:
        n += 1 + len(pk["energyCards"]) + len(pk["tools"]) + len(pk["preEvolution"])
    return n


def _raw(
    *,
    hand: list[int],
    discard: list[int],
    active: list[int],
    bench: list[dict],
    prize: list,
    deck_count: int | None = None,
    select_deck: list[int] | None = None,
    opp_deck_count: int = 30,
    opp_hand_count: int = 5,
    opp_prize: int = 6,
    opp_active_facedown: bool = False,
) -> dict:
    if deck_count is None:
        # 60 = deck + hand + board + discard + all prize slots (revealed and hidden)
        deck_count = 60 - _visible_count(active, bench, hand, discard) - len(prize)
    me = {
        "active": [_pokemon(active[0]) if active else None] if active else [],
        "bench": bench,
        "benchMax": 5,
        "deckCount": deck_count,
        "discard": [_card_ref(c) for c in discard],
        "prize": prize,
        "handCount": len(hand),
        "hand": [_card_ref(c) for c in hand],
        "poisoned": False, "burned": False, "asleep": False,
        "paralyzed": False, "confused": False,
    }
    opp = {
        "active": [None] if opp_active_facedown else [_pokemon(200)],
        "bench": [],
        "benchMax": 5,
        "deckCount": opp_deck_count,
        "discard": [],
        "prize": [None] * opp_prize,
        "handCount": opp_hand_count,
        "hand": None,
        "poisoned": False, "burned": False, "asleep": False,
        "paralyzed": False, "confused": False,
    }
    select = None
    if select_deck is not None:
        select = {
            "type": 1, "context": 7, "minCount": 1, "maxCount": 1,
            "remainDamageCounter": 0, "remainEnergyCost": 0,
            "option": [{"type": 8, "index": 0}],
            "deck": [_card_ref(c) for c in select_deck],
            "contextCard": None, "effect": None,
        }
    return {
        "select": select,
        "logs": [],
        "current": {
            "turn": 3, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
            "supporterPlayed": False, "stadiumPlayed": False,
            "energyAttached": False, "retreated": False, "result": -1,
            "stadium": [], "looking": None, "players": [me, opp],
        },
        "search_begin_input": "AAAA",
    }


def _my_deck(counts: dict[int, int]) -> list[int]:
    out: list[int] = []
    for cid, n in counts.items():
        out.extend([cid] * n)
    return out


def test_pool_invariant_and_split() -> None:
    # deck: 10x200, 20x3, 30x50 = 60 cards. Visible: 1 active(200), 1 bench(200
    # with 2 energy=3 + pre=50), 3 in hand (3,3,50), 2 discard (50,50) => 10
    # non-deck non-prize cards; deckCount auto = 60 - 10 - 6 = 44.
    deck = _my_deck({200: 10, 3: 20, 50: 30})
    bench = [_pokemon(200, energy=[3, 3], pre=[50])]
    raw = _raw(hand=[3, 3, 50], discard=[50, 50],
               active=[200], bench=bench, prize=[None] * 6)
    det = Determinizer(deck, _db(), random.Random(1))
    worlds = det.sample(parser.parse(raw), 3)
    assert len(worlds) == 3
    for w in worlds:
        assert len(w.your_deck) == 44
        assert len(w.your_prize) == 6  # all hidden
        # visible mine: active 200; bench 200 + 3 + 3 + 50; hand 3,3,50; disc 50,50
        seen = Counter({200: 2, 3: 4, 50: 4})
        pool = Counter(w.your_deck) + Counter(w.your_prize)
        assert pool == Counter(deck) - seen


def test_revealed_prizes_are_kept_and_excluded_from_pool() -> None:
    deck = _my_deck({200: 10, 3: 20, 50: 30})
    # one prize revealed as a 200, five hidden. Visible non-prize = active(1);
    # deckCount auto = 60 - 1 - 6 = 53.
    prize = [_card_ref(200)] + [None] * 5
    raw = _raw(hand=[], discard=[], active=[200], bench=[], prize=prize)
    det = Determinizer(deck, _db(), random.Random(2))
    w = det.sample(parser.parse(raw), 1)[0]
    assert w.your_prize[0] == 200  # revealed slot preserved
    assert len(w.your_deck) == 53
    # unseen = deck - active(200) - revealed prize(200) = 58 = deck(53) + 5 hidden
    seen = Counter({200: 2})
    pool = Counter(w.your_deck) + Counter(w.your_prize[1:])  # slot0 is the known 200
    assert pool == Counter(deck) - seen


def test_select_deck_revealed_pool_is_only_hidden_prizes() -> None:
    deck = _my_deck({200: 10, 3: 20, 50: 30})
    # Whole current deck (53 cards) revealed via select.deck; your_deck ignored.
    # active(1) + deck(53) + prizes(6) = 60. Hidden prizes = deck - active -
    # revealed = {3:2, 50:4}.
    revealed = _my_deck({200: 9, 3: 18, 50: 26})  # 53 cards still in deck
    raw = _raw(hand=[], discard=[], active=[200], bench=[],
               prize=[None] * 6, select_deck=revealed)
    det = Determinizer(deck, _db(), random.Random(3))
    w = det.sample(parser.parse(raw), 1)[0]
    assert w.your_deck == []  # engine ignores it; we emit empty
    assert len(w.your_prize) == 6
    seen = Counter({200: 1 + 9, 3: 18, 50: 26})
    assert Counter(w.your_prize) == Counter(deck) - seen


def test_mismatch_raises_determinize_error() -> None:
    deck = _my_deck({200: 10, 3: 20, 50: 30})
    # claim deckCount 41 but only 59 cards unaccounted -> mismatch
    raw = _raw(deck_count=41, hand=[], discard=[], active=[200], bench=[], prize=[None] * 6)
    det = Determinizer(deck, _db(), random.Random(4))
    with pytest.raises(DeterminizeError):
        det.sample(parser.parse(raw), 1)


def test_unknown_visible_card_raises() -> None:
    deck = _my_deck({200: 10, 3: 20, 50: 30})
    raw = _raw(deck_count=40, hand=[999], discard=[], active=[200], bench=[], prize=[None] * 6)
    det = Determinizer(deck, _db(), random.Random(5))
    with pytest.raises(DeterminizeError):
        det.sample(parser.parse(raw), 1)


def test_opponent_filler_counts_and_basic() -> None:
    deck = _my_deck({200: 10, 3: 20, 50: 30})
    raw = _raw(hand=[], discard=[], active=[200], bench=[],
               prize=[None] * 6, opp_deck_count=33, opp_hand_count=7, opp_prize=4,
               opp_active_facedown=True)
    det = Determinizer(deck, _db(), random.Random(6))
    w = det.sample(parser.parse(raw), 1)[0]
    assert len(w.opponent_deck) == 33
    assert len(w.opponent_hand) == 7
    assert len(w.opponent_prize) == 4
    assert w.opponent_active == [200]  # a Basic guess for the face-down active
    assert w.opponent_deck[0] == 200   # Basic first in the deck guess


def test_face_up_opponent_active_needs_no_guess() -> None:
    deck = _my_deck({200: 10, 3: 20, 50: 30})
    raw = _raw(hand=[], discard=[], active=[200], bench=[],
               prize=[None] * 6, opp_active_facedown=False)
    det = Determinizer(deck, _db(), random.Random(7))
    w = det.sample(parser.parse(raw), 1)[0]
    assert w.opponent_active == []


def test_seeded_determinism() -> None:
    deck = _my_deck({200: 10, 3: 20, 50: 30})
    raw = _raw(hand=[], discard=[], active=[200], bench=[], prize=[None] * 6)
    parsed = parser.parse(raw)
    a = Determinizer(deck, _db(), random.Random(42)).sample(parsed, 2)
    b = Determinizer(deck, _db(), random.Random(42)).sample(parsed, 2)
    assert [w.your_deck for w in a] == [w.your_deck for w in b]


def test_hidden_info_as_args_order() -> None:
    h = HiddenInfo([1], [2], [3], [4], [5], [6])
    assert h.as_args() == ([1], [2], [3], [4], [5], [6])
