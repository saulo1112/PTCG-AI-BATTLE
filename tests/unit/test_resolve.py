"""Structured action resolution."""

from ptcg_ai.observation.models import AreaKind, OptionKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.observation.resolve import resolve_option
from tests.conftest import make_raw_obs

parser = ObservationParser()


def test_play_resolves_hand_card() -> None:
    raw = make_raw_obs()
    raw["select"]["option"] = [{"type": 7, "index": 2}]  # PLAY hand[2]
    obs = parser.parse(raw)
    resolved = resolve_option(obs.select.option[0], obs)
    assert resolved.kind is OptionKind.PLAY
    assert resolved.card_id == 3  # synthetic hand is five Basic Water Energy
    assert resolved.source_area is AreaKind.HAND
    assert resolved.resolved


def test_attach_resolves_source_and_target() -> None:
    raw = make_raw_obs()
    raw["select"]["option"] = [
        {"type": 8, "area": 2, "index": 0, "inPlayArea": 4, "inPlayIndex": 0}
    ]  # ATTACH HAND[0] -> ACTIVE[0]
    obs = parser.parse(raw)
    resolved = resolve_option(obs.select.option[0], obs)
    assert resolved.card_id == 3
    assert resolved.target_card_id == 721  # synthetic active pokemon
    assert resolved.target_area is AreaKind.ACTIVE
    assert resolved.resolved


def test_attack_and_bare_options() -> None:
    raw = make_raw_obs()
    raw["select"]["option"] = [{"type": 13, "attackId": 42}, {"type": 14}, {"type": 12}]
    obs = parser.parse(raw)
    attack, end, retreat = (resolve_option(o, obs) for o in obs.select.option)
    assert attack.attack_id == 42 and attack.resolved
    assert end.kind is OptionKind.END and end.resolved
    assert retreat.card_id == 721  # own active


def test_opponent_hand_is_unresolvable() -> None:
    raw = make_raw_obs()
    # CARD option pointing at opponent's (hidden) hand.
    raw["select"]["option"] = [{"type": 3, "area": 2, "index": 0, "playerIndex": 1}]
    obs = parser.parse(raw)
    resolved = resolve_option(obs.select.option[0], obs)
    assert not resolved.resolved
    assert resolved.card_id is None


def test_out_of_range_never_raises() -> None:
    raw = make_raw_obs()
    raw["select"]["option"] = [
        {"type": 3, "area": 5, "index": 99, "playerIndex": 0},   # BENCH[99]
        {"type": 8, "area": 2, "index": 50, "inPlayArea": 5, "inPlayIndex": 9},
        {"type": 55},  # unknown future kind
    ]
    obs = parser.parse(raw)
    for option in obs.select.option:
        resolved = resolve_option(option, obs)
        assert not resolved.resolved


def test_every_fixture_option_resolves_without_raising(
    observation_fixtures: dict[str, dict]
) -> None:
    """Ground-truth sweep: no option in any captured decision may raise, and
    the acting player's own-zone options must resolve to real card IDs."""
    for name, raw in observation_fixtures.items():
        obs = parser.parse(raw)
        if obs.select is None:
            continue
        for option in obs.select.option:
            resolved = resolve_option(option, obs)
            if option.type in (OptionKind.PLAY, OptionKind.ATTACH) and obs.current is not None:
                assert resolved.card_id is not None, (name, option)
