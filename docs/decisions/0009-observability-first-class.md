# ADR-0009: Observability as a first-class concern

- **Status**: Accepted
- **Date**: 2026-07-05

## Context

Debugging decision-making will dominate development time: "why did the agent
attach energy there?" must be answerable in seconds, not by re-running games
under a debugger. Timing per decision also feeds the benchmark-gated
algorithm decision (ADR-0006) and the unknown Kaggle per-move budget.

## Problem

Is observability an ad-hoc sprinkle of prints, or a designed layer?

## Alternatives

1. **Ad-hoc logging where needed** — cheap now; unstructured, inconsistent,
   and always missing when you need it.
2. **A dedicated debug layer** — pretty-printers, structured decision traces,
   a tracing policy wrapper; small upfront cost.

## Decision

`ptcg_ai.debug` ships from day one: `inspect` (human-readable rendering of
observations, options, and logs) and `trace` (a `TracedDecision` record per
decision — select kind/context, option count, chosen indices, elapsed ms,
optional policy note — collected by `DecisionTracer`, persisted as JSONL,
rendered as a report). Tracing wraps any Policy (`TracingPolicy`) and is
zero-cost when not enabled. Policies may expose `last_note` explaining their
choice.

## Justification

Every rung of the baseline ladder benefits; traces make policy comparisons
and regressions concrete ("the new policy stopped attacking on turn 3");
per-decision timing is required data for ADR-0010 anyway.

## Consequences

- The Policy protocol includes the optional `last_note` hook — a one-field
  cost on implementers.
- Trace JSONL shares the raw-first philosophy of ADR-0008.
- CLI exposes `ptcg show-obs` and `ptcg battle --trace`.
