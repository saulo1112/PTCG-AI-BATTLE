"""Pretty-printers must render (and never raise) on any observation."""

from ptcg_ai.debug.inspect import format_observation, format_select, summarize_logs
from ptcg_ai.observation.parser import ObservationParser
from tests.conftest import make_raw_obs

parser = ObservationParser()


def test_format_synthetic_observation() -> None:
    obs = parser.parse(make_raw_obs())
    text = format_observation(obs)
    assert "turn 3" in text
    assert "(you)" in text
    assert "hidden" in text  # opponent hand
    assert "[0]" in text  # numbered options


def test_format_first_call() -> None:
    obs = parser.parse({"select": None, "logs": [], "current": None})
    assert "deck submission" in format_observation(obs)


def test_format_select_shows_bounds() -> None:
    obs = parser.parse(make_raw_obs(n_options=4, min_count=1, max_count=2))
    assert "pick 1..2 of 4" in format_select(obs.select)


def test_summarize_logs() -> None:
    obs = parser.parse(make_raw_obs())
    assert "TURN_START P0" in summarize_logs(obs.logs)


def test_formatters_tolerate_unknown_enums() -> None:
    raw = make_raw_obs()
    raw["select"]["option"].append({"type": 55})
    raw["logs"].append({"type": 88, "playerIndex": 0})
    obs = parser.parse(raw)
    assert "UNKNOWN_55" in format_observation(obs)
    assert "UNKNOWN_88" in summarize_logs(obs.logs)
