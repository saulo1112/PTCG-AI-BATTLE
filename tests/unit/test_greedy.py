"""GreedyPolicy (rung 3) priority order and legality."""

from __future__ import annotations

from typing import Any

from ptcg_ai.cards.database import AttackInfo, CardDatabase, CardInfo
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.observation.models import CardKind, EnergyKind, ParsedSelect
from ptcg_ai.observation.parser import ObservationParser

parser = ObservationParser()


def _card(cid: int, **kw: Any) -> CardInfo:
    base: dict[str, Any] = dict(
        cardId=cid, name=f"card{cid}", cardType=CardKind.POKEMON, retreatCost=1,
        hp=100, weakness=None, resistance=None, energyType=EnergyKind.WATER,
        basic=True, stage1=False, stage2=False, ex=False, megaEx=False,
        tera=False, aceSpec=False, evolvesFrom=None, skills=(), attacks=(),
    )
    base.update(kw)
    return CardInfo(**base)


def _db() -> CardDatabase:
    cards = {
        721: _card(721, energyType=EnergyKind.WATER, hp=60, attacks=(900,)),
        722: _card(722, name="Snover", basic=True, hp=70),
        800: _card(800, hp=90, weakness=EnergyKind.WATER),
        3: _card(3, cardType=CardKind.BASIC_ENERGY, basic=False),
        # Whitelisted Trainers (M3): a search Item and a draw Supporter.
        1152: _card(1152, name="Poké Pad", cardType=CardKind.ITEM, basic=False),
        1227: _card(1227, name="Lillie's Determination",
                    cardType=CardKind.SUPPORTER, basic=False),
        # A non-whitelisted Trainer must NOT be auto-played.
        1182: _card(1182, name="Boss's Orders", cardType=CardKind.SUPPORTER,
                    basic=False),
        # M4/W2 whitelist additions: a search Item and fetch Supporters.
        1145: _card(1145, name="Mega Signal", cardType=CardKind.ITEM, basic=False),
        1205: _card(1205, name="Cyrano", cardType=CardKind.SUPPORTER, basic=False),
        1235: _card(1235, name="Waitress", cardType=CardKind.SUPPORTER, basic=False),
    }
    attacks = {900: AttackInfo(attackId=900, name="Surf", text="", damage=50,
                               energies=(EnergyKind.WATER,))}
    return CardDatabase(cards, attacks)


def _player(active_id: int, active_hp: int, *, bench: int = 0,
            hand_ids: list[int] | None = None) -> dict[str, Any]:
    hand_ids = hand_ids or []
    return {
        "active": [{"id": active_id, "serial": 10, "hp": active_hp, "maxHp": active_hp,
                    "appearThisTurn": False, "energies": [3, 3], "energyCards": [],
                    "tools": [], "preEvolution": []}],
        "bench": [{"id": 722, "serial": 20 + i, "hp": 70, "maxHp": 70,
                   "appearThisTurn": False, "energies": [], "energyCards": [],
                   "tools": [], "preEvolution": []} for i in range(bench)],
        "benchMax": 5, "deckCount": 40, "discard": [],
        "prize": [None] * 6, "handCount": len(hand_ids),
        "hand": [{"id": cid, "serial": 30 + i, "playerIndex": 0}
                 for i, cid in enumerate(hand_ids)],
        "poisoned": False, "burned": False, "asleep": False,
        "paralyzed": False, "confused": False,
    }


def _raw_main(options: list[dict[str, Any]], *, bench: int = 0,
              energy_attached: bool = False, opp_hp: int = 90,
              hand_ids: list[int] | None = None,
              select_type: int = 0, context: int = 0,
              min_count: int = 1, max_count: int = 1) -> dict[str, Any]:
    me = _player(721, 60, bench=bench, hand_ids=hand_ids)
    opp = dict(_player(800, opp_hp), hand=None, handCount=4)
    return {
        "select": {"type": select_type, "context": context, "minCount": min_count,
                   "maxCount": max_count, "remainDamageCounter": 0,
                   "remainEnergyCost": 0, "option": options,
                   "deck": None, "contextCard": None, "effect": None},
        "logs": [], "current": {
            "turn": 3, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
            "supporterPlayed": False, "stadiumPlayed": False,
            "energyAttached": energy_attached, "retreated": False, "result": -1,
            "stadium": [], "looking": None, "players": [me, opp],
        },
        "search_begin_input": "AAAA",
    }


ATTACK = {"type": 13, "attackId": 900}
END = {"type": 14}
PLAY_0 = {"type": 7, "index": 0}
ATTACH_ACTIVE = {"type": 8, "area": 2, "index": 0, "inPlayArea": 4, "inPlayIndex": 0}
ATTACH_BENCH = {"type": 8, "area": 2, "index": 0, "inPlayArea": 5, "inPlayIndex": 0}


def _play_hand(index: int) -> dict[str, Any]:
    """A PLAY option pointing at hand slot ``index``."""
    return {"type": 7, "index": index}


def _card_option(area: int, index: int) -> dict[str, Any]:
    return {"type": 3, "area": area, "index": index}


def _ctx(raw: dict[str, Any]) -> DecisionContext:
    return DecisionContext(raw=raw, observation=parser.parse(raw), cards=_db())


def _choose(raw: dict[str, Any]) -> list[int]:
    return GreedyPolicy().choose(_ctx(raw))


def test_lethal_attack_taken_first() -> None:
    # opp active weak to WATER, 90 HP; attack 50 → 100 doubled = lethal.
    raw = _raw_main([ATTACH_ACTIVE, ATTACK, END], bench=2, opp_hp=90)
    assert _choose(raw) == [1]


def test_develop_bench_before_attach() -> None:
    # no lethal (opp 200 HP); a Basic in hand and an empty bench slot.
    raw = _raw_main([ATTACH_ACTIVE, PLAY_0, END], bench=0, opp_hp=200, hand_ids=[722])
    assert _choose(raw) == [1]  # PLAY the basic


def test_attach_to_active_when_no_development() -> None:
    raw = _raw_main([ATTACH_BENCH, ATTACH_ACTIVE, END], bench=5, opp_hp=200)
    assert _choose(raw) == [1]  # prefers the active-targeted attach


def test_non_lethal_attack_over_end() -> None:
    raw = _raw_main([ATTACK, END], bench=5, energy_attached=True, opp_hp=200)
    assert _choose(raw) == [0]


def test_end_when_only_option() -> None:
    raw = _raw_main([END], bench=5, energy_attached=True)
    assert _choose(raw) == [0]


def test_is_first_goes_first() -> None:
    raw = _raw_main([{"type": 1}, {"type": 2}], select_type=9, context=41)
    assert _choose(raw) == [0]  # YES


def test_setup_bench_develops_all() -> None:
    opts = [{"type": 3, "area": 2, "index": i} for i in range(3)]
    raw = _raw_main(opts, select_type=1, context=2, min_count=0, max_count=3)
    assert _choose(raw) == [0, 1, 2]


def test_output_always_legal() -> None:
    for raw in (
        _raw_main([ATTACH_ACTIVE, ATTACK, END], bench=2),
        _raw_main([END], bench=5, energy_attached=True),
        _raw_main([{"type": 1}, {"type": 2}], select_type=9, context=41),
    ):
        select: ParsedSelect = parser.parse(raw).select  # type: ignore[assignment]
        action = _choose(raw)
        assert len(set(action)) == len(action)
        assert all(0 <= i < len(select.option) for i in action)
        assert select.minCount <= len(action) <= select.maxCount


def test_note_is_set() -> None:
    policy = GreedyPolicy()
    policy.choose(_ctx(_raw_main([ATTACH_ACTIVE, ATTACK, END], bench=2)))
    assert policy.last_note and "lethal" in policy.last_note


# -- M3: whitelisted Trainer play -----------------------------------------

def test_search_item_played_before_develop() -> None:
    # Hand: [Basic(722), Poké Pad(1152)]. Search Item (priority 2) beats
    # develop-bench (priority 3). No lethal (opp 200 HP), bench has room.
    raw = _raw_main(
        [PLAY_0, _play_hand(1), END], bench=0, opp_hp=200, hand_ids=[722, 1152],
    )
    assert _choose(raw) == [1]  # PLAY the Poké Pad (index 1)


def test_draw_supporter_played_after_development_not_before() -> None:
    # Hand: [Basic(722), Lillie's(1227)]. With an empty bench, develop
    # (priority 3) must beat the draw Supporter (priority 6): keep the Basic.
    raw = _raw_main(
        [PLAY_0, _play_hand(1), END], bench=0, opp_hp=200, hand_ids=[722, 1227],
    )
    assert _choose(raw) == [0]  # PLAY the Basic, not Lillie's


def test_draw_supporter_played_when_only_chaff_left() -> None:
    # Bench full, energy attached, no evolve: the draw Supporter is the only
    # progress left before attack/END.
    raw = _raw_main(
        [_play_hand(0), END], bench=5, energy_attached=True, opp_hp=200,
        hand_ids=[1227],
    )
    assert _choose(raw) == [0]  # play Lillie's to refresh


def test_non_whitelisted_trainer_not_auto_played() -> None:
    # Boss's Orders (1182) is NOT whitelisted; with nothing else to do we END
    # rather than fire an effect we cannot reason about.
    raw = _raw_main(
        [_play_hand(0), END], bench=5, energy_attached=True, opp_hp=200,
        hand_ids=[1182],
    )
    assert _choose(raw) == [1]  # END, not the Boss's Orders


def test_to_hand_search_prefers_basic_then_energy() -> None:
    # A TO_HAND deck search offering [energy(3), Basic(722), evolution(800)]:
    # take the Basic first.
    opts = [_card_option(1, 0), _card_option(1, 1), _card_option(1, 2)]
    raw = _raw_main(opts, select_type=1, context=7, min_count=1, max_count=1)
    raw["select"]["deck"] = [
        {"id": 3, "serial": 1, "playerIndex": 0},
        {"id": 722, "serial": 2, "playerIndex": 0},
        {"id": 800, "serial": 3, "playerIndex": 0},
    ]
    assert _choose(raw) == [1]  # the Basic (722)


# -- M4/W2: expanded whitelist + mid-effect attach -------------------------

def test_mega_signal_item_played_in_search_tier() -> None:
    # Mega Signal (1145) is a search Item — fired before develop/attach, like
    # any other whitelisted Item. Bench full so develop can't mask it.
    raw = _raw_main(
        [_play_hand(0), END], bench=5, energy_attached=True, opp_hp=200,
        hand_ids=[1145],
    )
    assert _choose(raw) == [0]  # play Mega Signal


def test_fetch_supporter_played_before_draw_supporter() -> None:
    # Hand: [Waitress(1235), Lillie's(1227)]. Both are Supporters (one/turn);
    # the fetch Supporter (tier 6) beats the draw Supporter (tier 7).
    raw = _raw_main(
        [_play_hand(0), _play_hand(1), END], bench=5, energy_attached=True,
        opp_hp=200, hand_ids=[1235, 1227],
    )
    assert _choose(raw) == [0]  # Waitress before Lillie's


def test_mid_effect_attach_targets_active() -> None:
    # Waitress ATTACH_FROM sub-select: CARD options for [bench, active];
    # attach to the Active.
    opts = [{"type": 3, "area": 5, "index": 0}, {"type": 3, "area": 4, "index": 0}]
    raw = _raw_main(opts, select_type=1, context=21, min_count=1, max_count=1)
    assert _choose(raw) == [1]  # the ACTIVE-area option


def test_whitelist_is_configurable_for_ablation() -> None:
    # With the M3 whitelist (no fetch Supporters, no Mega Signal), the new
    # cards are NOT auto-played — reproduces the pre-W2 policy exactly.
    old = GreedyPolicy(
        search_items=frozenset({1152, 1102, 1142}),
        fetch_supporters=frozenset(),
    )
    raw = _raw_main(
        [_play_hand(0), END], bench=5, energy_attached=True, opp_hp=200,
        hand_ids=[1145],  # Mega Signal — not in the ablated search set
    )
    assert old.choose(_ctx(raw)) == [1]  # END, not Mega Signal
