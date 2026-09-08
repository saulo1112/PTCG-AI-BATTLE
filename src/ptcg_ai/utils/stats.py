"""Shared statistical helpers (stdlib only).

Conventions (docs/methodology.md): Wilson intervals for game-level
proportions; normal intervals over per-game means for decision-level
statistics (decisions within a game are correlated — never pool them as if
i.i.d.).
"""

from __future__ import annotations

from statistics import NormalDist, mean, stdev
from typing import Sequence


def wilson_interval(
    successes: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a proportion (draws may count as 0.5)."""
    if n == 0:
        return (0.0, 1.0)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return (max(0.0, center - margin), min(1.0, center + margin))


def mean_interval(
    values: Sequence[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    """(mean, low, high) normal-approximation CI over independent values.

    Feed this *per-game* values, never pooled per-decision values.
    """
    if not values:
        return (0.0, 0.0, 0.0)
    m = mean(values)
    if len(values) < 2:
        return (m, m, m)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    half = z * stdev(values) / (len(values) ** 0.5)
    return (m, m - half, m + half)


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted sequence (q in [0,1])."""
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1))))
    return sorted_values[index]
