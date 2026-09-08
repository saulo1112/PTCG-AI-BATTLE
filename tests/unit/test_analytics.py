"""Analytics extractors, aggregation, and report rendering.

Synthetic episodes with known answers — the arithmetic must be exact.
"""

from pathlib import Path

from ptcg_ai.analytics.aggregate import Aggregator
from ptcg_ai.analytics.extractors import extract_episode
from ptcg_ai.analytics.report import render_markdown, write_reports
from ptcg_ai.replay.episode import Episode, Step
from tests.conftest import make_raw_obs


def _obs(turn: int, your_index: int = 0, n_options: int = 4) -> dict:
    raw = make_raw_obs(n_options=n_options)
    raw["current"]["turn"] = turn
    raw["current"]["yourIndex"] = your_index
    return raw


def _episode(final_viewer: int = 0) -> Episode:
    """3 decisions (P0 turn 1 ×2, P1 turn 2), terminal at turn 4."""
    steps = [
        Step(raw_obs=_obs(1, 0, n_options=4), action=[0], player=0, elapsed_ms=1.0),
        Step(raw_obs=_obs(1, 0, n_options=2), action=[1], player=0, elapsed_ms=2.0),
        Step(raw_obs=_obs(2, 1, n_options=6), action=[5], player=1, elapsed_ms=3.0),
    ]
    final = _obs(4, final_viewer)
    final["current"]["result"] = 0
    return Episode(
        steps=steps,
        final_obs=final,
        outcome={"result": 0, "winner": 0, "reason": 1, "duration_s": 0.5},
    )


def test_extract_game_record() -> None:
    extract = extract_episode(_episode())
    assert extract.game.decisions == 3
    assert extract.game.final_turn == 4
    assert extract.game.end_reason == 1
    assert extract.game.winner == 0
    assert extract.game.duration_s == 0.5
    assert not extract.events_partial


def test_extract_decisions_and_kinds() -> None:
    extract = extract_episode(_episode())
    assert len(extract.decisions) == 3
    first = extract.decisions[0]
    assert first.select_kind == "MAIN" and first.select_context == "MAIN"
    assert first.n_options == 4
    assert first.option_kinds == {"PLAY": 3, "END": 1}
    assert first.chosen_kinds == {"PLAY": 1}  # action [0] is a PLAY
    # Step 3 chose index 5 of 6 → the END option.
    assert extract.decisions[2].chosen_kinds == {"END": 1}


def test_terminal_observation_is_not_a_decision() -> None:
    extract = extract_episode(_episode())
    assert len(extract.decisions) == extract.game.decisions == 3


def test_event_counting_viewer_convention() -> None:
    # Each synthetic obs has exactly one TURN_START log. Counting viewer = 0:
    # steps by P0 (2 of them) + terminal (viewer 0) = 3 TURN_STARTs.
    extract = extract_episode(_episode(final_viewer=0))
    assert extract.event_counts["TURN_START"] == 3
    # Counting viewer = 1: P1 steps (1) + terminal (viewer 1) = 2.
    extract = extract_episode(_episode(final_viewer=1))
    assert extract.event_counts["TURN_START"] == 2


def test_zone_sampling_once_per_turn() -> None:
    extract = extract_episode(_episode())
    # Turns 1 and 2 each have MAIN decisions; sampled once each.
    assert extract.zone_samples == 2
    # Synthetic hand: five cards of id 3, visible only in the turn-1 sample
    # (make_raw_obs gives players[1] a hidden hand, so the turn-2 acting
    # player contributes nothing — unlike the real engine).
    assert extract.zone_counts["hand"][3] == 5
    # Both players' actives (id 721) per sample → 4.
    assert extract.zone_counts["active"][721] == 4


def test_aggregator_known_numbers() -> None:
    aggregator = Aggregator()
    aggregator.add(extract_episode(_episode()))
    aggregator.add(extract_episode(_episode()))
    result = aggregator.result()

    assert result["n_games"] == 2
    assert result["n_decisions"] == 6
    assert result["game_length"]["decisions"]["mean"] == 3.0
    assert result["game_length"]["turns"]["mean"] == 4.0
    assert result["end_reasons"]["1"]["count"] == 2
    assert result["end_reasons"]["1"]["share"] == 1.0
    assert result["winners"]["0"]["count"] == 2
    # Branching per game: mean(4,2,6) = 4.0 in both games.
    assert result["branching"]["per_game_mean"]["mean"] == 4.0
    assert result["branching"]["pooled_max"] == 6
    context = result["contexts"]["MAIN/MAIN"]
    assert context["count"] == 6
    assert context["branching_max"] == 6
    main_options = result["options_by_select_kind"]["MAIN"]
    assert main_options["PLAY"]["chosen"] == 2  # step 1's [0], once per game
    assert main_options["END"]["chosen"] == 4   # steps 2 and 3 both pick END
    assert result["latency_ms"]["n"] == 6
    assert result["latency_ms"]["max"] == 3.0


def test_report_rendering_and_files(tmp_path: Path) -> None:
    aggregator = Aggregator()
    aggregator.add(extract_episode(_episode()))
    result = aggregator.result()

    markdown = render_markdown(result, header={"experiment": "unit"})
    assert "# Battle Analysis Report" in markdown
    assert "prizes taken" in markdown  # end-reason naming
    assert "MAIN/MAIN" in markdown
    assert "95% CI" in markdown

    write_reports(result, tmp_path, header={"experiment": "unit"})
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()
    for table in ("contexts", "options_by_select_kind", "end_reasons", "events", "zones"):
        assert (tmp_path / "tables" / f"{table}.csv").is_file()


def test_v1_episode_without_final_obs_is_partial() -> None:
    episode = _episode()
    episode.final_obs = None
    extract = extract_episode(episode)
    assert extract.events_partial
    assert extract.game.final_turn is None
