"""Data-driven context classification (M8-3, SDK-free).

``_classify_contexts`` decides, per select-context, whether to learn a scorer,
emit a degenerate zeros vector, or skip entirely — without any hard-coded
context list. This guards the three verdict paths directly: a context whose
options are all indistinguishable (identical resolved card id) must be flagged
degenerate even when greedy already looks perfect there; a context with real
choice and a beatable greedy baseline must be flagged learnable.
"""

from __future__ import annotations

from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.imitation.kaggle_replay import ReplayDecision
from ptcg_ai.imitation.train import _classify_contexts
from ptcg_ai.observation.parser import ObservationParser


def _obs(context: int, hand_ids: list[int], select_type: int = 0, options=None) -> dict:
    """A select observation; default PLAY options are each hand[i]."""
    n = len(hand_ids)
    player = {
        "active": [{"id": 721, "serial": 1, "hp": 60, "maxHp": 60, "appearThisTurn": False,
                    "energies": [], "energyCards": [], "tools": [], "preEvolution": []}],
        "bench": [], "benchMax": 5, "deckCount": 40, "discard": [],
        "prize": [None] * 6, "handCount": n,
        "hand": [{"id": cid, "serial": 10 + i, "playerIndex": 0} for i, cid in enumerate(hand_ids)],
        "poisoned": False, "burned": False, "asleep": False, "paralyzed": False, "confused": False,
    }
    opp = dict(player, hand=None, handCount=1)
    if options is None:
        options = [{"type": 7, "index": i} for i in range(n)]
    return {
        "select": {
            "type": select_type, "context": context, "minCount": 1, "maxCount": 1,
            "remainDamageCounter": 0, "remainEnergyCost": 0,
            "option": options,
            "deck": None, "contextCard": None, "effect": None,
        },
        "logs": [], "search_begin_input": "AAAA",
        "current": {"turn": 3, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
                    "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": False,
                    "retreated": False, "result": -1, "stadium": [], "looking": None,
                    "players": [player, opp]},
    }


def _decisions(context: int, hand_ids: list[int], action: int, n: int,
               select_type: int = 0, options=None) -> list[ReplayDecision]:
    """``n`` copies of the same (context, action) decision as train rows."""
    n_opt = len(options) if options is not None else len(hand_ids)
    return [
        ReplayDecision(
            game_id=f"g{i}", seat=0, step_index=0,
            raw_observation=_obs(context, hand_ids, select_type, options),
            action=[action], won=True, context=context, select_type=select_type,
            min_count=1, max_count=1, n_options=n_opt,
        )
        for i in range(n)
    ]


def test_degenerate_card_context_flagged_zeros() -> None:
    # A CARD select (type 1, e.g. ATTACH_TO) whose options all resolve to the
    # SAME card id -> degenerate + card-like -> "zeros" (live take-k is correct),
    # even though greedy (which just returns [0]) looks 100% correct.
    rows = _decisions(context=22, hand_ids=[3, 3, 3], action=0, n=300, select_type=1)
    cls = _classify_contexts(
        profile=None, train_rows=rows, val_rows=rows,
        parser=ObservationParser(), cards=None, greedy=GreedyPolicy(),
    )
    v = next(iter(cls.values()))
    assert v["verdict"] == "zeros"
    assert v["degen_frac"] == 1.0


def test_number_select_degenerate_routes_to_skip_not_zeros() -> None:
    # A COUNT select (type 8, DRAW_COUNT) is "degenerate" by the card-id test
    # (NUMBER options resolve to card_id=None) but take-k=index-0 would draw the
    # MINIMUM. It must route to "skip" -> greedy _max_number, not "zeros".
    number_opts = [{"type": 0, "number": k} for k in range(4)]  # NUMBER options 0..3
    rows = _decisions(context=38, hand_ids=[0], action=3, n=300,
                      select_type=8, options=number_opts)
    cls = _classify_contexts(
        profile=None, train_rows=rows, val_rows=rows,
        parser=ObservationParser(), cards=None, greedy=GreedyPolicy(),
    )
    v = next(iter(cls.values()))
    assert v["degen_frac"] == 1.0            # looks degenerate (all card_id None)
    assert v["select_type"] == 8
    assert v["verdict"] == "skip"            # but NOT zeros -> greedy fallback


def test_learnable_context_with_real_choice_and_beatable_greedy() -> None:
    # Distinct card ids per option (real choice) and the teacher always picks
    # the LAST option while greedy (no card DB) takes the first ATTACK/END or
    # falls through to _safe_default -> first option: greedy should be beaten.
    hand_ids = [3, 5, 7]
    rows = _decisions(context=0, hand_ids=hand_ids, action=2, n=300)
    cls = _classify_contexts(
        profile=None, train_rows=rows, val_rows=rows,
        parser=ObservationParser(), cards=None, greedy=GreedyPolicy(),
    )
    d = cls["MAIN"]
    assert d["degen_frac"] < 0.5  # not degenerate: 3 distinct ids
    assert d["n_train"] == 300
    # Greedy (no card knowledge) always picks the first Basic play or falls to
    # index 0 -> systematically wrong against a teacher who always picks index 2.
    assert d["greedy_acc"] < 0.5
    assert d["verdict"] == "learn"


def test_too_few_rows_is_not_learned() -> None:
    rows = _decisions(context=0, hand_ids=[3, 5, 7], action=2, n=10)  # < _MIN_ROWS
    cls = _classify_contexts(
        profile=None, train_rows=rows, val_rows=rows,
        parser=ObservationParser(), cards=None, greedy=GreedyPolicy(),
    )
    assert cls["MAIN"]["verdict"] == "skip"
