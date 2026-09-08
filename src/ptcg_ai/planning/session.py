"""Context-managed wrapper over the vendored ``cg`` search API (ADR-0010).

The engine's search state is a process-global, persistent node tree; stepping
copies the parent so branching is free, but the node pool must be released with
``search_end`` or it leaks native memory across the ~600 s episode. This module
guarantees that hygiene: :class:`SearchSession` is a context manager whose
``__exit__`` always calls ``search_end``.

The search API returns ``cg`` dataclass observations, but our decision pipeline
(parser, ``GameState``) consumes raw-dict observations. :func:`observation_to_mapping`
bridges the two, producing a dict indistinguishable from a live Kaggle
observation (enums coerced to their ints), so the sim path reuses the exact
production parser.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from types import ModuleType
from typing import Any, Mapping

from ptcg_ai.observation.models import ParsedObservation
from ptcg_ai.observation.parser import ObservationParser


def observation_to_mapping(obs: Any) -> dict[str, Any]:
    """Convert a ``cg`` ``Observation`` dataclass to a raw-obs dict.

    Recurses dataclasses -> dicts, lists -> lists, and ``IntEnum`` members ->
    their ``int`` values, matching the shape the engine/Kaggle deliver and the
    :class:`ObservationParser` expects.
    """
    return _to_plain(obs)


def _to_plain(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int, float)):
        # bool/int first: IntEnum is an int subclass, handled below explicitly
        if isinstance(obj, Enum):
            return obj.value
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


class SimNode:
    """A node in the search tree: a decision point (or terminal) inside a sim.

    Wraps a ``cg`` ``SearchState``; ``raw`` and ``parsed`` are computed lazily
    and cached (parsing is ~0.1 ms and only needed at leaves/rollout steps).
    """

    __slots__ = ("_ss", "_parser", "_raw", "_parsed")

    def __init__(self, search_state: Any, parser: ObservationParser) -> None:
        self._ss = search_state
        self._parser = parser
        self._raw: dict[str, Any] | None = None
        self._parsed: ParsedObservation | None = None

    @property
    def search_id(self) -> int:
        return self._ss.searchId

    @property
    def observation(self) -> Any:
        """The raw ``cg`` observation dataclass (no conversion)."""
        return self._ss.observation

    @property
    def raw(self) -> dict[str, Any]:
        if self._raw is None:
            self._raw = observation_to_mapping(self._ss.observation)
        return self._raw

    @property
    def parsed(self) -> ParsedObservation:
        if self._parsed is None:
            self._parsed = self._parser.parse(self.raw)
        return self._parsed

    @property
    def actor_index(self) -> int | None:
        """Whose decision this node is (``current.yourIndex``), or None if the
        game is over / there is no state."""
        cur = self._ss.observation.current
        return None if cur is None else cur.yourIndex

    @property
    def is_terminal(self) -> bool:
        cur = self._ss.observation.current
        if cur is not None and cur.result != -1:
            return True
        return self._ss.observation.select is None

    @property
    def result(self) -> int:
        """Winner index (0/1), 2 for a draw, or -1 if not finished."""
        cur = self._ss.observation.current
        return -1 if cur is None else cur.result

    @property
    def n_options(self) -> int:
        sel = self._ss.observation.select
        return 0 if sel is None else len(sel.option)


class SearchSession:
    """One determinized-search lifetime. Use as a context manager::

        with SearchSession(api, parser) as sess:
            root = sess.begin(raw_obs, hidden, manual_coin=False)
            leaf = sess.step(root, [0])
        # search_end() called here, unconditionally

    A session owns the whole process-global node pool for its lifetime, so run
    one session per decision and let all determinized worlds/candidates live
    inside it, then close.
    """

    __slots__ = ("_api", "_parser", "_active", "begins", "steps")

    def __init__(self, api: ModuleType, parser: ObservationParser) -> None:
        self._api = api
        self._parser = parser
        self._active = False
        self.begins = 0
        self.steps = 0

    def __enter__(self) -> "SearchSession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def begin(self, raw_obs: Mapping[str, Any], hidden: Any, manual_coin: bool = False) -> SimNode:
        """Start a determinized world from a live raw observation.

        ``hidden`` is a :class:`~ptcg_ai.planning.determinize.HiddenInfo`
        (anything exposing ``as_args()``).
        """
        agent_obs = self._api.to_observation_class(dict(raw_obs))
        ss = self._api.search_begin(agent_obs, *hidden.as_args(), manual_coin=manual_coin)
        self._active = True
        self.begins += 1
        return SimNode(ss, self._parser)

    def step(self, node: SimNode, select: list[int]) -> SimNode:
        ss = self._api.search_step(node.search_id, list(select))
        self.steps += 1
        return SimNode(ss, self._parser)

    def close(self) -> None:
        if self._active:
            self._api.search_end()
            self._active = False
