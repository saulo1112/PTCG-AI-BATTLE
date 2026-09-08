"""Structured decision traces (ADR-0009).

``TracingPolicy`` wraps any Policy and records one :class:`TracedDecision`
per call — decision shape, choice, latency, and the policy's own
explanation (``last_note``). Latency data doubles as input to the
benchmark-gated algorithm decision (ADR-0006).
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

from ptcg_ai.decision.base import DecisionContext, Policy


@dataclass(frozen=True)
class TracedDecision:
    """One decision as observed at the policy boundary."""

    index: int
    turn: int | None
    player: int | None
    select_type: str
    select_context: str
    n_options: int
    min_count: int
    max_count: int
    chosen: list[int]
    elapsed_ms: float
    note: str | None
    policy: str


class DecisionTracer:
    """Collects decisions; persists as JSONL; renders a readable report."""

    def __init__(self) -> None:
        self.decisions: list[TracedDecision] = []

    def record(self, decision: TracedDecision) -> None:
        self.decisions.append(decision)

    def save_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for decision in self.decisions:
                fh.write(json.dumps(dataclasses.asdict(decision)) + "\n")

    def report(self) -> str:
        """A compact per-decision table plus latency summary."""
        if not self.decisions:
            return "(no decisions traced)"
        lines = [
            f"{'#':>4} {'turn':>4} {'pl':>2} {'decision':<38} {'opts':>4} "
            f"{'chosen':<14} {'ms':>7}  note"
        ]
        for d in self.decisions:
            decision = f"{d.select_type}/{d.select_context}"
            lines.append(
                f"{d.index:>4} {d.turn if d.turn is not None else '-':>4} "
                f"{d.player if d.player is not None else '-':>2} {decision:<38} "
                f"{d.n_options:>4} {str(d.chosen):<14} {d.elapsed_ms:>7.2f}  {d.note or ''}"
            )
        times = sorted(d.elapsed_ms for d in self.decisions)
        mid = times[len(times) // 2]
        lines.append(
            f"decisions: {len(times)} | latency ms p50={mid:.2f} "
            f"max={times[-1]:.2f} total={sum(times):.1f}"
        )
        return "\n".join(lines)


class TracingPolicy:
    """Transparent Policy wrapper feeding a :class:`DecisionTracer`."""

    def __init__(self, inner: Policy, tracer: DecisionTracer) -> None:
        self._inner = inner
        self._tracer = tracer
        self.name = inner.name
        self.last_note: str | None = None

    def choose(self, ctx: DecisionContext) -> list[int]:
        start = time.perf_counter()
        chosen = self._inner.choose(ctx)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.last_note = self._inner.last_note

        select = ctx.observation.select
        state = ctx.observation.current
        assert select is not None  # choose() is never the deck call
        self._tracer.record(
            TracedDecision(
                index=len(self._tracer.decisions),
                turn=state.turn if state is not None else None,
                player=state.yourIndex if state is not None else None,
                select_type=select.type.name,
                select_context=select.context.name,
                n_options=len(select.option),
                min_count=select.minCount,
                max_count=select.maxCount,
                chosen=list(chosen),
                elapsed_ms=elapsed_ms,
                note=self._inner.last_note,
                policy=self._inner.name,
            )
        )
        return chosen

    def choose_deck(self, ctx: DecisionContext) -> list[int]:
        return self._inner.choose_deck(ctx)

    def on_battle_start(self) -> None:
        self._inner.on_battle_start()

    def on_battle_end(self, outcome: object) -> None:
        self._inner.on_battle_end(outcome)  # type: ignore[arg-type]
