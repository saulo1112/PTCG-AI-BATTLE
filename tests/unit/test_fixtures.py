"""Regression tests over captured real observations (ADR-0005 / ADR-0007).

These are the schema-drift detectors: when a new SDK drop changes the
observation shape, re-capturing fixtures and running this module shows
exactly what changed (new extras keys, new enum values).
"""

from typing import Any

from ptcg_ai.debug.inspect import format_observation
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.decision.random_policy import RandomPolicy
from ptcg_ai.observation.models import ParsedObservation
from ptcg_ai.observation.parser import ObservationParser

parser = ObservationParser()


def _all_extras(obs: ParsedObservation) -> dict[str, dict[str, Any]]:
    """Collect every non-empty extras mapping in a parsed observation."""
    found: dict[str, dict[str, Any]] = {}

    def add(label: str, extras: Any) -> None:
        if extras:
            found[label] = dict(extras)

    add("observation", obs.extras)
    if obs.select is not None:
        add("select", obs.select.extras)
        for i, option in enumerate(obs.select.option):
            add(f"select.option[{i}]", option.extras)
    if obs.current is not None:
        add("current", obs.current.extras)
        for pi, player in enumerate(obs.current.players):
            add(f"players[{pi}]", player.extras)
            for pokemon in [*player.active, *player.bench]:
                if pokemon is not None:
                    add(f"players[{pi}].pokemon", pokemon.extras)
    return found


def test_every_fixture_parses(observation_fixtures: dict[str, dict]) -> None:
    for name, raw in observation_fixtures.items():
        obs = parser.parse(raw)
        assert obs is not None, name


def test_no_schema_drift_in_fixtures(observation_fixtures: dict[str, dict]) -> None:
    """Extras must be empty for the SDK version the fixtures came from.

    If this fails after re-capturing, the SDK grew new fields: inspect them,
    extend the models, and note the change in docs/sdk_analysis.md.
    """
    for name, raw in observation_fixtures.items():
        assert _all_extras(parser.parse(raw)) == {}, f"schema drift in {name}"


def test_no_unknown_enum_values_in_fixtures(observation_fixtures: dict[str, dict]) -> None:
    for name, raw in observation_fixtures.items():
        obs = parser.parse(raw)
        if obs.select is not None:
            assert not obs.select.type.is_unknown, name
            assert not obs.select.context.is_unknown, name
            assert not any(o.type.is_unknown for o in obs.select.option), name
        assert not any(log.type.is_unknown for log in obs.logs), name


def test_random_policy_legal_on_every_fixture(observation_fixtures: dict[str, dict]) -> None:
    policy = RandomPolicy(seed=3, deck=[3] * 60)
    for name, raw in observation_fixtures.items():
        ctx = DecisionContext(raw=raw, observation=parser.parse(raw))
        if ctx.observation.select is None:
            assert len(policy.choose_deck(ctx)) == 60
            continue
        if ctx.observation.current is not None and ctx.observation.current.result != -1:
            # Terminal observations carry a stale select (counts can exceed
            # the empty option list); the policy must not crash on them.
            assert policy.choose(ctx) == [], name
            continue
        select = ctx.observation.select
        for _ in range(20):
            action = policy.choose(ctx)
            assert select.minCount <= len(action) <= select.maxCount, name
            assert len(set(action)) == len(action), name
            assert all(0 <= i < len(select.option) for i in action), name


def test_formatter_renders_every_fixture(observation_fixtures: dict[str, dict]) -> None:
    for name, raw in observation_fixtures.items():
        text = format_observation(parser.parse(raw))
        assert isinstance(text, str) and text, name
