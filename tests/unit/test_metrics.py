"""Match statistics."""

from ptcg_ai.evaluation.metrics import MatchStats


def test_score_rate_counts_draws_as_half() -> None:
    stats = MatchStats(wins=3, losses=1, draws=2)
    assert stats.n == 6
    assert stats.score_rate == (3 + 1.0) / 6


def test_add_respects_side() -> None:
    stats = MatchStats()
    stats.add(winner=0, a_played_as=0)  # win
    stats.add(winner=0, a_played_as=1)  # loss
    stats.add(winner=None, a_played_as=0)  # draw
    assert (stats.wins, stats.losses, stats.draws) == (1, 1, 1)


def test_wilson_interval_shrinks_with_n() -> None:
    small = MatchStats(wins=6, losses=4)
    large = MatchStats(wins=60, losses=40)
    lo_s, hi_s = small.wilson_interval()
    lo_l, hi_l = large.wilson_interval()
    assert hi_s - lo_s > hi_l - lo_l
    assert 0.0 <= lo_l <= 0.6 <= hi_l <= 1.0


def test_empty_stats() -> None:
    stats = MatchStats()
    assert stats.score_rate == 0.0
    assert stats.wilson_interval() == (0.0, 1.0)
