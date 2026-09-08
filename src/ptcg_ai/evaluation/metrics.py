"""Match statistics.

The competition rating counts only win/loss/draw (no margin), so arena
metrics mirror that: score = wins + draws/2.
"""

from __future__ import annotations

from dataclasses import dataclass

from ptcg_ai.utils.stats import wilson_interval


@dataclass
class MatchStats:
    """Aggregate result of a series of games from player A's perspective."""

    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def n(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score_rate(self) -> float:
        """(wins + draws/2) / n — the rating-relevant success rate."""
        if self.n == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.n

    def add(self, winner: int | None, a_played_as: int) -> None:
        """Record one game given the winner index and which side A played."""
        if winner is None:
            self.draws += 1
        elif winner == a_played_as:
            self.wins += 1
        else:
            self.losses += 1

    def wilson_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """Wilson score interval for the score rate (draws counted as half).

        Use before trusting small-sample win rates — 6/10 is not "60%".
        """
        return wilson_interval(self.wins + 0.5 * self.draws, self.n, confidence)

    def summary(self) -> str:
        low, high = self.wilson_interval()
        return (
            f"{self.wins}W-{self.losses}L-{self.draws}D of {self.n} | "
            f"score rate {self.score_rate:.3f} (95% CI {low:.3f}..{high:.3f})"
        )
