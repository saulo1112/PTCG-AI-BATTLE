"""SearchSession lifecycle + observation converter (SDK-free with a fake api)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.planning.session import SearchSession, SimNode, observation_to_mapping


# -- a miniature cg-shaped dataclass tree for the converter -----------------

class _Kind(IntEnum):
    A = 0
    B = 5


@dataclass
class _Card:
    id: int
    serial: int
    playerIndex: int


@dataclass
class _Opt:
    type: _Kind
    index: int | None = None


@dataclass
class _Sel:
    type: _Kind
    option: list[_Opt]
    deck: list[_Card] | None


@dataclass
class _Obs:
    select: _Sel | None
    current: dict | None
    search_begin_input: str | None


def test_converter_coerces_enums_and_nesting() -> None:
    obs = _Obs(
        select=_Sel(type=_Kind.B, option=[_Opt(type=_Kind.A, index=2)], deck=None),
        current={"turn": 3},
        search_begin_input=None,
    )
    d = observation_to_mapping(obs)
    assert d["select"]["type"] == 5  # IntEnum -> int
    assert d["select"]["option"][0]["type"] == 0
    assert d["select"]["option"][0]["index"] == 2
    assert d["select"]["deck"] is None
    assert d["current"] == {"turn": 3}
    assert d["search_begin_input"] is None


def test_converter_output_is_parseable() -> None:
    # A converted observation must round-trip through the real parser.
    from tests.conftest import make_raw_obs

    raw = make_raw_obs()
    parsed = ObservationParser().parse(raw)
    assert parsed.select is not None
    assert parsed.current is not None


# -- SearchSession lifecycle with a fake api --------------------------------

class _FakeState:
    def __init__(self, search_id: int, obs):
        self.searchId = search_id
        self.observation = obs


class _FakeApi:
    """Records begin/step/end calls; returns scripted _FakeState nodes."""

    def __init__(self) -> None:
        self.ends = 0
        self.begins = 0
        self.steps: list[tuple[int, list[int]]] = []

    def to_observation_class(self, raw):
        return raw  # identity: begin just needs an object to pass on

    def search_begin(self, agent_obs, *hidden, manual_coin=False):
        self.begins += 1
        return _FakeState(1, _Obs(select=None, current=None, search_begin_input=None))

    def search_step(self, search_id, select):
        self.steps.append((search_id, list(select)))
        return _FakeState(search_id + 1, _Obs(select=None, current=None, search_begin_input=None))

    def search_end(self):
        self.ends += 1


class _Hidden:
    def as_args(self):
        return ([1], [2], [3], [4], [5], [6])


def test_session_calls_search_end_on_normal_exit() -> None:
    api = _FakeApi()
    parser = ObservationParser()
    with SearchSession(api, parser) as sess:
        node = sess.begin({"search_begin_input": "x"}, _Hidden())
        sess.step(node, [0])
    assert api.ends == 1
    assert api.begins == 1
    assert api.steps == [(1, [0])]
    assert sess.begins == 1 and sess.steps == 1


def test_session_calls_search_end_on_exception() -> None:
    api = _FakeApi()
    parser = ObservationParser()
    try:
        with SearchSession(api, parser) as sess:
            sess.begin({"search_begin_input": "x"}, _Hidden())
            raise ValueError("boom")
    except ValueError:
        pass
    assert api.ends == 1  # cleanup happened despite the exception


def test_session_close_is_idempotent_without_begin() -> None:
    api = _FakeApi()
    with SearchSession(api, ObservationParser()):
        pass
    assert api.ends == 0  # never began -> nothing to release


def test_simnode_terminal_and_actor() -> None:
    parser = ObservationParser()
    finished = _FakeState(1, _Obs(select=None, current=type("C", (), {"result": 0, "yourIndex": 1})(),
                                  search_begin_input=None))
    node = SimNode(finished, parser)
    assert node.is_terminal
    assert node.result == 0
    assert node.actor_index == 1
