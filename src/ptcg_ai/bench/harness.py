"""Timing/statistics plumbing shared by all benchmarks.

Note on memory: ``tracemalloc`` sees Python allocations only. The engine's
native allocations (notably search trees) are invisible to it — measure
those via process RSS deltas, recorded in ``notes`` by the benchmarks that
care.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class BenchResult:
    """Aggregated timing for one benchmarked operation."""

    name: str
    n_ops: int
    total_s: float
    ops_per_s: float
    p50_ms: float
    p95_ms: float
    max_ms: float
    notes: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_durations(
        cls, name: str, durations_s: list[float], notes: Mapping[str, Any] | None = None
    ) -> "BenchResult":
        if not durations_s:
            return cls(name, 0, 0.0, 0.0, 0.0, 0.0, 0.0, notes or {})
        total = sum(durations_s)
        ms = sorted(d * 1000.0 for d in durations_s)
        p95 = ms[min(len(ms) - 1, int(round(0.95 * (len(ms) - 1))))]
        return cls(
            name=name,
            n_ops=len(ms),
            total_s=total,
            ops_per_s=len(ms) / total if total > 0 else 0.0,
            p50_ms=statistics.median(ms),
            p95_ms=p95,
            max_ms=ms[-1],
            notes=notes or {},
        )


def format_report(results: list[BenchResult]) -> str:
    """Markdown table + notes, printable and committable."""
    lines = [
        "| benchmark | ops | ops/s | p50 ms | p95 ms | max ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.n_ops} | {r.ops_per_s:,.1f} | "
            f"{r.p50_ms:.2f} | {r.p95_ms:.2f} | {r.max_ms:.2f} |"
        )
    for r in results:
        if r.notes:
            notes = ", ".join(f"{k}={v}" for k, v in r.notes.items())
            lines.append(f"- **{r.name}**: {notes}")
    return "\n".join(lines)


def save_report(results: list[BenchResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_report(results) + "\n", encoding="utf-8")
