"""RandomPolicy legality and determinism."""

import pytest

from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.decision.random_policy import RandomPolicy
from ptcg_ai.observation.parser import ObservationParser
from tests.conftest import make_raw_obs

parser = ObservationParser()


def _ctx(raw: dict) -> DecisionContext:
    return DecisionContext(raw=raw, observation=parser.parse(raw))


@pytest.mark.parametrize("min_count,max_count", [(0, 0), (0, 2), (1, 1), (2, 4)])
def test_choice_is_always_legal(min_count: int, max_count: int) -> None:
    policy = RandomPolicy(seed=1)
    ctx = _ctx(make_raw_obs(n_options=4, min_count=min_count, max_count=max_count))
    for _ in range(50):
        action = policy.choose(ctx)
        assert min_count <= len(action) <= max_count
        assert len(set(action)) == len(action)
        assert all(0 <= i < 4 for i in action)


def test_seeded_determinism() -> None:
    ctx = _ctx(make_raw_obs(n_options=6, min_count=1, max_count=3))
    run1 = [RandomPolicy(seed=99).choose(ctx) for _ in range(10)]
    run2 = [RandomPolicy(seed=99).choose(ctx) for _ in range(10)]
    assert run1 == run2


def test_choose_deck_requires_deck() -> None:
    policy = RandomPolicy()
    ctx = _ctx({"select": None, "logs": [], "current": None})
    with pytest.raises(RuntimeError, match="not given a deck"):
        policy.choose_deck(ctx)
    assert RandomPolicy(deck=[3] * 60).choose_deck(ctx) == [3] * 60


def test_note_is_set() -> None:
    policy = RandomPolicy(seed=1)
    policy.choose(_ctx(make_raw_obs()))
    assert policy.last_note is not None and "random" in policy.last_note
