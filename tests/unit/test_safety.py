"""SafePolicy: the crash insurance must actually insure."""

from ptcg_ai.decision.base import BasePolicy, DecisionContext
from ptcg_ai.decision.safety import SafePolicy
from ptcg_ai.observation.parser import ObservationParser
from tests.conftest import make_raw_obs

parser = ObservationParser()


class _Raising(BasePolicy):
    name = "raising"

    def choose(self, ctx: DecisionContext) -> list[int]:
        raise RuntimeError("boom")


class _OutOfRange(BasePolicy):
    name = "out-of-range"

    def choose(self, ctx: DecisionContext) -> list[int]:
        return [999]


class _Fine(BasePolicy):
    name = "fine"

    def choose(self, ctx: DecisionContext) -> list[int]:
        self.last_note = "took option 0"
        return [0]


def _ctx() -> DecisionContext:
    raw = make_raw_obs(n_options=4, min_count=1, max_count=1)
    return DecisionContext(raw=raw, observation=parser.parse(raw))


def test_exception_falls_back_to_random() -> None:
    safe = SafePolicy(_Raising(), seed=1)
    action = safe.choose(_ctx())
    assert len(action) == 1 and 0 <= action[0] < 4
    assert safe.interventions == 1
    assert safe.last_note is not None and "FALLBACK" in safe.last_note


def test_illegal_output_falls_back() -> None:
    safe = SafePolicy(_OutOfRange(), seed=1)
    action = safe.choose(_ctx())
    assert 0 <= action[0] < 4
    assert safe.interventions == 1


def test_valid_policy_passes_through() -> None:
    safe = SafePolicy(_Fine(), seed=1)
    assert safe.choose(_ctx()) == [0]
    assert safe.interventions == 0
    assert safe.last_note == "took option 0"


def test_deck_fallback() -> None:
    safe = SafePolicy(_Raising(), deck=[3] * 60)
    raw = {"select": None, "logs": [], "current": None}
    ctx = DecisionContext(raw=raw, observation=parser.parse(raw))
    assert safe.choose_deck(ctx) == [3] * 60
    assert safe.interventions == 1
