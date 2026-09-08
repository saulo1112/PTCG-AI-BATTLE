"""M49 — a renamed team must not silently vanish from the corpus.

THE FAILURE THIS PINS, caught live. The imitation corpus is extracted by matching
``info.Agents[].Name`` against one string. A Kaggle team can RENAME itself, and the
project's teacher did: submission 54773249 shows as ``Yushin Ito`` in the 2026-08-02
snapshot and as ``AlphaStarmie`` / ``AlphaTCG`` / ``AlphaTcg`` in later episodes --
verified against the API by ``submissionId``, not guessed.

Matching one name against a corpus that uses three does not raise: it yields zero seats
for the other two eras, so ``build_decision_dataset`` writes a corpus missing a third of
its games and reports success. The context heads would then be retrained believing they
had +32% more data while having none. That is the same silent-wrong-artifact shape as
M37's wrong-deck submission, which passed every structural check it had.
"""

from __future__ import annotations

from ptcg_ai.imitation.kaggle_replay import player_seats


def _replay(name_a: str, name_b: str) -> dict:
    return {"info": {"Agents": [{"Name": name_a}, {"Name": name_b}]}}


def test_a_single_name_behaves_exactly_as_before():
    data = _replay("Yushin Ito", "Someone Else")
    assert player_seats(data, "Yushin Ito") == [0]
    assert player_seats(data, "Someone Else") == [1]
    assert player_seats(data, "Nobody") == []


def test_matching_is_case_insensitive():
    """`AlphaTCG` and `AlphaTcg` are the SAME team on consecutive days."""
    assert player_seats(_replay("AlphaTcg", "x"), "AlphaTCG") == [0]
    assert player_seats(_replay("ALPHATCG", "x"), "alphatcg") == [0]


def test_a_name_set_matches_every_era_of_a_renamed_team():
    names = ["Yushin Ito", "AlphaStarmie", "AlphaTCG"]
    for era_name in names + ["AlphaTcg"]:
        data = _replay(era_name, "Opponent")
        assert player_seats(data, names) == [0], f"{era_name} not matched by the name set"


def test_the_old_single_name_MISSES_the_renamed_eras():
    """The regression itself: this is what would have silently shrunk the corpus."""
    assert player_seats(_replay("AlphaStarmie", "Opponent"), "Yushin Ito") == []
    assert player_seats(_replay("AlphaTCG", "Opponent"), "Yushin Ito") == []


def test_a_self_match_still_returns_both_seats():
    data = _replay("AlphaStarmie", "Yushin Ito")   # same team, two eras of the name
    assert player_seats(data, ["Yushin Ito", "AlphaStarmie"]) == [0, 1]


def test_missing_or_null_names_do_not_crash():
    data = {"info": {"Agents": [{"Name": None}, {}]}}
    assert player_seats(data, ["Yushin Ito"]) == []
    assert player_seats({}, ["Yushin Ito"]) == []
