"""Forward-compatible parsing (ADR-0005) on synthetic observations."""

from ptcg_ai.observation.models import LogKind, OptionKind, SelectKind
from ptcg_ai.observation.parser import ObservationParser
from tests.conftest import make_raw_obs

parser = ObservationParser()


def test_parses_synthetic_main_phase() -> None:
    obs = parser.parse(make_raw_obs(n_options=4))
    assert obs.select is not None
    assert obs.select.type is SelectKind.MAIN
    assert len(obs.select.option) == 4
    assert obs.select.option[-1].type is OptionKind.END
    assert obs.current is not None
    assert obs.current.me.hand is not None
    assert obs.current.opponent.hand is None  # hidden information
    assert obs.logs[0].type is LogKind.TURN_START


def test_first_call_observation_parses() -> None:
    obs = parser.parse({"select": None, "logs": [], "current": None})
    assert obs.select is None
    assert obs.current is None
    assert obs.logs == ()


def test_unknown_keys_land_in_extras() -> None:
    raw = make_raw_obs(extra_select_keys={"newCompetitionField": 42})
    raw["someTopLevelAddition"] = {"a": 1}
    obs = parser.parse(raw)
    assert obs.extras == {"someTopLevelAddition": {"a": 1}}
    assert obs.select is not None
    assert obs.select.extras == {"newCompetitionField": 42}


def test_unknown_enum_values_survive() -> None:
    raw = make_raw_obs()
    raw["select"]["type"] = 99
    raw["select"]["context"] = 77
    raw["select"]["option"].append({"type": 55})
    raw["logs"].append({"type": 88, "playerIndex": 1})
    obs = parser.parse(raw)
    assert obs.select is not None
    assert obs.select.type.is_unknown and obs.select.type == 99
    assert obs.select.type.name == "UNKNOWN_99"
    assert obs.select.context == 77
    assert obs.select.option[-1].type == 55
    assert obs.logs[-1].type == 88


def test_perspective_properties() -> None:
    raw = make_raw_obs()
    raw["current"]["yourIndex"] = 1
    obs = parser.parse(raw)
    assert obs.current is not None
    assert obs.current.me is obs.current.players[1]
    assert obs.current.opponent is obs.current.players[0]
