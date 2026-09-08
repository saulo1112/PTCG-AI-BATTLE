"""Sentence rendering for resolved actions."""

from ptcg_ai.debug.describe import describe_action, render_decision
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.observation.resolve import resolve_option
from tests.conftest import make_raw_obs

parser = ObservationParser()


def test_core_sentences() -> None:
    raw = make_raw_obs()
    raw["select"]["option"] = [
        {"type": 7, "index": 0},                                            # PLAY
        {"type": 8, "area": 2, "index": 0, "inPlayArea": 4, "inPlayIndex": 0},  # ATTACH
        {"type": 13, "attackId": 42},                                       # ATTACK
        {"type": 12},                                                       # RETREAT
        {"type": 14},                                                       # END
    ]
    obs = parser.parse(raw)
    sentences = [describe_action(resolve_option(o, obs)) for o in obs.select.option]
    assert sentences[0] == "Play #3 from hand"
    assert sentences[1] == "Attach #3 to #721"
    assert sentences[2] == "Attack: #42"
    assert sentences[3] == "Retreat #721"
    assert sentences[4] == "End turn"


def test_describe_never_raises_on_unknown_kind() -> None:
    raw = make_raw_obs()
    raw["select"]["option"] = [{"type": 55}]
    obs = parser.parse(raw)
    text = describe_action(resolve_option(obs.select.option[0], obs))
    assert "UNKNOWN_55" in text


def test_render_decision_block() -> None:
    raw = make_raw_obs(n_options=3)
    block = render_decision(raw, action=[0], elapsed_ms=1.5, decision_index=7)
    assert "Decision 7 | Turn 3 | P0 to act" in block
    assert "Legal actions:" in block
    assert "Chosen: " in block
    assert "Decision time: 1.50 ms" in block


def test_render_decision_over_all_fixtures(observation_fixtures: dict[str, dict]) -> None:
    """The full block must render for every captured decision shape."""
    for name, raw in observation_fixtures.items():
        block = render_decision(raw, action=None)
        assert isinstance(block, str) and block, name
