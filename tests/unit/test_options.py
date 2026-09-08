"""Option helpers."""

from ptcg_ai.observation.models import OptionKind
from ptcg_ai.observation.options import (
    decision_signature,
    group_options,
    is_first_call,
    legal_action_counts,
)
from ptcg_ai.observation.parser import ObservationParser
from tests.conftest import make_raw_obs

parser = ObservationParser()


def test_is_first_call() -> None:
    assert is_first_call({"select": None})
    assert not is_first_call(make_raw_obs())


def test_group_options_and_counts() -> None:
    obs = parser.parse(make_raw_obs(n_options=4, min_count=1, max_count=2))
    assert obs.select is not None
    groups = group_options(obs.select)
    assert groups[OptionKind.PLAY] == [0, 1, 2]
    assert groups[OptionKind.END] == [3]
    assert legal_action_counts(obs.select) == (1, 2)


def test_decision_signature_is_stable() -> None:
    obs = parser.parse(make_raw_obs())
    assert obs.select is not None
    assert decision_signature(obs.select) == "MAIN/MAIN:END+PLAY"
