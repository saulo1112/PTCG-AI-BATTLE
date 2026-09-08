"""Streaming aggregation of episode extracts.

Feed extracts one at a time (:meth:`Aggregator.add`); memory stays bounded
by the aggregate structures, never by the dataset. CI methodology
(docs/methodology.md): game-level proportions get Wilson intervals;
decision-level statistics are reported as means of per-game means with
normal CIs over games (decisions within a game are correlated).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any

from ptcg_ai.analytics.extractors import EpisodeExtract
from ptcg_ai.utils.stats import mean_interval, percentile, wilson_interval


@dataclass
class _ContextStats:
    """Accumulators for one (select_kind, select_context) pair."""

    count: int = 0
    branching: list[int] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        values = sorted(self.branching)
        return {
            "count": self.count,
            "branching_mean": round(mean(values), 2) if values else 0,
            "branching_median": median(values) if values else 0,
            "branching_max": values[-1] if values else 0,
        }


class Aggregator:
    """Accumulates :class:`EpisodeExtract` records into an analysis dict."""

    def __init__(self) -> None:
        self.n_games = 0
        self.n_decisions = 0
        # Game-level values (one entry per game — CI-safe).
        self._game_decisions: list[float] = []
        self._game_turns: list[float] = []
        self._game_durations: list[float] = []
        self._game_mean_branching: list[float] = []
        self._end_reasons: Counter = Counter()
        self._winners: Counter = Counter()
        # Decision-level accumulators (pooled; used for shapes, not CIs).
        self._contexts: dict[tuple[str, str], _ContextStats] = defaultdict(_ContextStats)
        self._select_kind_counts: Counter = Counter()
        self._available_by_kind: Counter = Counter()  # (select_kind, option_kind)
        self._chosen_by_kind: Counter = Counter()
        self._latency_ms: list[float] = []
        # Events and zones.
        self._events: Counter = Counter()
        self._events_partial_games = 0
        self._zone_counts: dict[str, Counter] = defaultdict(Counter)
        self._zone_samples = 0

    def add(self, extract: EpisodeExtract) -> None:
        game = extract.game
        self.n_games += 1
        self.n_decisions += len(extract.decisions)
        self._game_decisions.append(game.decisions)
        if game.final_turn is not None:
            self._game_turns.append(game.final_turn)
        if game.duration_s is not None:
            self._game_durations.append(game.duration_s)
        self._end_reasons[game.end_reason] += 1
        self._winners[game.winner] += 1

        branchings: list[int] = []
        for decision in extract.decisions:
            key = (decision.select_kind, decision.select_context)
            stats = self._contexts[key]
            stats.count += 1
            stats.branching.append(decision.n_options)
            branchings.append(decision.n_options)
            self._select_kind_counts[decision.select_kind] += 1
            for kind, count in decision.option_kinds.items():
                self._available_by_kind[(decision.select_kind, kind)] += count
            for kind, count in decision.chosen_kinds.items():
                self._chosen_by_kind[(decision.select_kind, kind)] += count
            if decision.elapsed_ms is not None:
                self._latency_ms.append(decision.elapsed_ms)
        if branchings:
            self._game_mean_branching.append(mean(branchings))

        self._events.update(extract.event_counts)
        self._events_partial_games += extract.events_partial
        for zone, counter in extract.zone_counts.items():
            self._zone_counts[zone].update(counter)
        self._zone_samples += extract.zone_samples

    def result(self) -> dict[str, Any]:
        """The full analysis as a JSON-serializable dict."""
        turns_total = sum(self._game_turns)
        latency = sorted(self._latency_ms)
        return {
            "n_games": self.n_games,
            "n_decisions": self.n_decisions,
            "game_length": {
                "decisions": _mean_ci_dict(self._game_decisions),
                "turns": _mean_ci_dict(self._game_turns),
                "duration_s": _mean_ci_dict(self._game_durations),
            },
            "end_reasons": _proportions(self._end_reasons, self.n_games),
            "winners": _proportions(self._winners, self.n_games),
            "branching": {
                "per_game_mean": _mean_ci_dict(self._game_mean_branching),
                "pooled_max": max(
                    (s.summary()["branching_max"] for s in self._contexts.values()), default=0
                ),
            },
            "select_kinds": _proportions(self._select_kind_counts, self.n_decisions),
            "contexts": {
                f"{kind}/{context}": stats.summary()
                for (kind, context), stats in sorted(
                    self._contexts.items(), key=lambda kv: -kv[1].count
                )
            },
            "options_by_select_kind": _kind_table(self._available_by_kind, self._chosen_by_kind),
            "events": {
                "counts": dict(self._events.most_common()),
                "per_turn": {
                    name: round(count / turns_total, 3)
                    for name, count in self._events.most_common()
                }
                if turns_total
                else {},
                "partial_games": self._events_partial_games,
            },
            "latency_ms": {
                "n": len(latency),
                "p50": round(percentile(latency, 0.50), 3),
                "p95": round(percentile(latency, 0.95), 3),
                "max": round(latency[-1], 3) if latency else 0.0,
            },
            "zones": {
                "samples": self._zone_samples,
                "top_cards": {
                    zone: dict(counter.most_common(50))
                    for zone, counter in self._zone_counts.items()
                },
            },
        }


def _mean_ci_dict(values: list[float]) -> dict[str, float]:
    m, low, high = mean_interval(values)
    return {
        "n": len(values),
        "mean": round(m, 2),
        "ci95_low": round(low, 2),
        "ci95_high": round(high, 2),
        "median": round(median(values), 2) if values else 0.0,
        "max": round(max(values), 2) if values else 0.0,
    }


def _proportions(counter: Counter, n: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, count in counter.most_common():
        low, high = wilson_interval(count, n)
        out[str(key)] = {
            "count": count,
            "share": round(count / n, 4) if n else 0.0,
            "ci95_low": round(low, 4),
            "ci95_high": round(high, 4),
        }
    return out


def _kind_table(available: Counter, chosen: Counter) -> dict[str, dict[str, dict[str, int]]]:
    """Per select kind: how often each option kind was available vs chosen."""
    table: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for (select_kind, option_kind), count in available.items():
        table[select_kind][option_kind] = {
            "available": count,
            "chosen": chosen.get((select_kind, option_kind), 0),
        }
    return {
        kind: dict(sorted(kinds.items(), key=lambda kv: -kv[1]["chosen"]))
        for kind, kinds in sorted(table.items())
    }
