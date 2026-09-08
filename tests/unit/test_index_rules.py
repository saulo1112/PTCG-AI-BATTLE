"""M47 — deterministic index rules for contexts whose options carry no card identity.

THE DEFECT THIS FIXES, measured over Yushin's 2330 games (`m47_count_audit.py` plus the
per-context follow-up): in `DRAW_COUNT` the teacher picks the LAST option **488 times out
of 488**, and the shipped agent — which has no model for that context — falls to
`GreedyPolicy._safe_default`, i.e. `list(range(minCount))` = option 0, **without looking
at the state**. A 100% systematic divergence on 0.21 decisions/game that no milestone had
ever examined.

It cannot be fixed with a model. The raw options are `{"number": 0, "type": 0}` /
`{"number": 1, "type": 0}` — no card id, so `featurize_option` returns the same vector for
each and the softmax is symmetric by construction. That is the M34 bottleneck at its
limit, and it is why this is a RULE and not a head.

The tests that matter here are the absence tests: a payload with no `index_rules` must
behave exactly as before, because four milestones' worth of ladder results depend on it.
"""

from __future__ import annotations

import json

import pytest

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.imitation.policy import ImitationPolicy
from ptcg_ai.observation.models import CardKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

from tests.conftest import make_raw_obs


def _payload(index_rules=None):
    p = {"version": 2, "profile": "ALAKAZAM", "feature_dim": 658, "contexts": {}}
    if index_rules is not None:
        p["index_rules"] = index_rules
    return p


def _obs(context: SelectContextKind, n_options: int, min_count=1, max_count=1):
    """Observation whose select carries BARE NUMERIC options, as a real DRAW_COUNT does:
    `{"number": k, "type": 0}` with no card id. Built on the shared `make_raw_obs` so the
    surrounding board is a real parseable one, then the option list is replaced."""
    raw = make_raw_obs(n_options=n_options, min_count=min_count, max_count=max_count,
                       context=int(context))
    raw["select"]["option"] = [{"number": i, "type": 0} for i in range(n_options)]
    return raw


def _cards() -> CardDatabase:
    """Just enough card knowledge for the greedy fallback to run on the synthetic board,
    so the fall-through tests exercise the real path instead of a stub."""
    def card(cid):
        return {"cardId": cid, "name": f"c{cid}", "cardType": int(CardKind.POKEMON),
                "retreatCost": 1, "hp": 120, "weakness": None, "resistance": None,
                "energyType": 1, "basic": True, "stage1": False, "stage2": False,
                "ex": False, "megaEx": False, "tera": False, "aceSpec": False,
                "evolvesFrom": None, "skills": [], "attacks": []}
    return CardDatabase.from_records({"cards": [card(3), card(721)], "attacks": []})


def _decide(policy, raw):
    parser = ObservationParser()
    obs = parser.parse(raw)
    cards = _cards()
    gs = GameState.build(obs, cards)
    return policy._decide(obs.select, obs, gs, cards)


def test_absent_index_rules_leave_the_agent_byte_identical():
    """The load-bearing test. Every shipped agent has no `index_rules` key."""
    plain = ImitationPolicy(_payload(), deck=[1] * 60)
    assert plain._index_rules == {}
    raw = _obs(SelectContextKind.DRAW_COUNT, 2)
    # No rule, no model for the context -> the greedy default, which is option 0.
    assert _decide(plain, raw) == [0]
    assert plain._index_rule_used == 0


def test_last_rule_picks_the_final_option():
    policy = ImitationPolicy(_payload({"DRAW_COUNT": "last"}), deck=[1] * 60)
    for n in (2, 3, 5):
        assert _decide(policy, _obs(SelectContextKind.DRAW_COUNT, n)) == [n - 1]
    assert policy._index_rule_used == 3


def test_first_rule_exists_and_is_the_old_behaviour():
    policy = ImitationPolicy(_payload({"DRAW_COUNT": "first"}), deck=[1] * 60)
    assert _decide(policy, _obs(SelectContextKind.DRAW_COUNT, 4)) == [0]


def test_the_rule_only_touches_its_own_context():
    policy = ImitationPolicy(_payload({"DRAW_COUNT": "last"}), deck=[1] * 60)
    assert _decide(policy, _obs(SelectContextKind.DISCARD_ENERGY, 3)) == [0]
    assert policy._index_rule_used == 0


def test_multi_pick_decisions_are_left_alone():
    """The audit measured WHICH INDEX on single-pick decisions. Applying it where the
    engine asks for several options would be inventing unmeasured behaviour."""
    policy = ImitationPolicy(_payload({"DRAW_COUNT": "last"}), deck=[1] * 60)
    chosen = _decide(policy, _obs(SelectContextKind.DRAW_COUNT, 4, min_count=2, max_count=2))
    assert chosen != [3]
    assert policy._index_rule_used == 0


def test_an_unknown_rule_name_is_rejected_at_load():
    with pytest.raises(ValueError, match="not 'first' or 'last'"):
        ImitationPolicy(_payload({"DRAW_COUNT": "maximum"}), deck=[1] * 60)


def test_rules_survive_a_json_round_trip():
    """The rule ships inside the weights payload, so it must survive serialisation the
    same way `recover_rule` and `count_heads` do."""
    text = json.dumps(_payload({"DRAW_COUNT": "last"}))
    policy = ImitationPolicy(json.loads(text), deck=[1] * 60)
    assert policy._index_rules == {"DRAW_COUNT": "last"}
