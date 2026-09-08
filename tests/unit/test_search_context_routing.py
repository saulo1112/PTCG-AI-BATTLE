"""M48 -- `ImitationSearchPolicy` must engage search ONLY on MAIN decisions.

THE BUG THIS CLOSES, measured directly (`m48_search_arena.py sweep`, n=100/arm against
the real champion mirror): with search unguarded across every context, score against the
champion fell to 0.175 (greedy_bias=0.02) and 0.280 (greedy_bias=0.05) -- confidently
worse, and monotonically WORSE the more search was allowed to override. The mechanism:
`SearchPolicy.choose` (the base) has no context check and was built against the M10-era
TR-650 champion, which had no context model outside MAIN to override. This project's
shipped champion has four specialized, individually-verified context heads (M42/M43:
ACTIVATE, SETUP_BENCH_POKEMON+count, SWITCH, TO_HAND's MLP ensemble; +0.096 to +0.117 in
this exact arena, the largest measured win in the project). Unguarded search would
generate alternative candidates for THOSE contexts too and could discard the specialized
head's answer whenever a 7-feature, MAIN-shaped board evaluator (never fit for "which
card to fetch" or "which bench slot to promote") scored an alternative higher by more
than `greedy_bias` -- which is exactly the monotonic-with-override-rate loss observed.

These tests do not need the native SDK: they verify DISPATCH (does the search machinery
get touched at all), not the search algorithm itself (covered by test_search_policy.py).
"""

from __future__ import annotations

import pytest

from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.decision.search import SearchPolicy
from ptcg_ai.decision.search_bc import ImitationSearchPolicy
from ptcg_ai.observation.models import SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from tests.conftest import make_raw_obs


def _payload():
    return {"version": 2, "profile": "TR_650", "feature_dim": 386, "contexts": {}}


def _critic():
    return {"mean": [0.0] * 7, "std": [1.0] * 7, "weights": [0.1] * 7, "bias": 0.0}


def _policy():
    # bc_rollout=False: the rollout policy is never exercised by these dispatch-only
    # tests, and GreedyPolicy needs no weights payload to construct.
    return ImitationSearchPolicy(_payload(), _critic(), deck=[1] * 60, bc_rollout=False)


def _ctx(context: SelectContextKind, n_options=3):
    raw = make_raw_obs(n_options=n_options, min_count=1, max_count=1, context=int(context))
    obs = ObservationParser().parse(raw)
    return DecisionContext(raw=raw, observation=obs, cards=None)


def test_non_main_context_bypasses_search_entirely(monkeypatch):
    pol = _policy()

    def _boom(*a, **k):
        raise AssertionError("search machinery touched for a non-MAIN decision")

    monkeypatch.setattr(pol, "_search_choose", _boom)
    monkeypatch.setattr(SearchPolicy, "choose", lambda self, ctx: _boom())

    called = {}
    real_greedy_choose = pol._greedy.choose

    def spy(ctx):
        called["yes"] = True
        return real_greedy_choose(ctx)

    monkeypatch.setattr(pol._greedy, "choose", spy)

    result = pol.choose(_ctx(SelectContextKind.TO_HAND))
    assert called.get("yes") is True
    assert result == pol._greedy.choose(_ctx(SelectContextKind.TO_HAND))
    assert pol.searched == 0 and pol.fallbacks == 0


@pytest.mark.parametrize("context", [
    SelectContextKind.SWITCH, SelectContextKind.ACTIVATE,
    SelectContextKind.SETUP_BENCH_POKEMON, SelectContextKind.EVOLVE,
])
def test_every_specialized_head_context_bypasses_search(monkeypatch, context):
    """The four contexts M42/M43 gave the champion specialized heads for -- the exact
    ones the base SearchPolicy would otherwise silently override."""
    pol = _policy()
    monkeypatch.setattr(SearchPolicy, "choose",
                        lambda self, ctx: (_ for _ in ()).throw(
                            AssertionError(f"search ran for {context.name}")))
    pol.choose(_ctx(context))   # must not raise


def test_main_context_still_routes_through_search(monkeypatch):
    """The routing guard must not accidentally swallow MAIN too."""
    pol = _policy()
    calls = []
    monkeypatch.setattr(SearchPolicy, "choose", lambda self, ctx: calls.append(1) or [0])
    result = pol.choose(_ctx(SelectContextKind.MAIN))
    assert calls == [1]
    assert result == [0]


def test_forced_single_option_never_touches_search_or_greedy_spy(monkeypatch):
    """A single-option MAIN decision is a forced move in the base class; confirm the
    routing guard still lets MAIN through to it rather than diverting to `_greedy`."""
    pol = _policy()
    monkeypatch.setattr(SearchPolicy, "choose", lambda self, ctx: [0])
    assert pol.choose(_ctx(SelectContextKind.MAIN, n_options=1)) == [0]
