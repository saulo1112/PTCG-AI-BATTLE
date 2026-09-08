# ADR-0006: Benchmark-gated algorithm commitment

- **Status**: Accepted
- **Date**: 2026-07-05

## Context

The SDK offers a determinized search API (fork the game with guessed hidden
information, step through hypothetical lines). Tree search (MCTS-family) is
the intuitive leading candidate, and learning-based approaches are possible.
But the decisive facts are unmeasured: search nodes/second on 2 vCPUs,
per-decision time budget on Kaggle (undocumented), decisions per battle,
memory per search tree. Committing an architecture to an algorithm before
measuring is the classic way to burn a six-week timeline.

## Problem

When may search/learning algorithm implementation begin, and what commits us?

## Alternatives

1. **Commit now to MCTS (or an RL pipeline)** — starts "real" work sooner;
   risks discovering mid-competition that the per-move budget or throughput
   makes it unviable.
2. **Benchmark first, then decide by ADR** — costs ~days of measurement;
   converts the decision from taste to data.

## Decision

No search or learning algorithm is chosen or implemented until the `bench/`
suite has measured, on representative states: Search API throughput
(`search_begin` cost, `search_step` nodes/s, tree memory), parser and battle
throughput, and the empirical Kaggle per-move budget. The viability decision
is then recorded as **ADR-0010**. Until that ADR exists, all algorithm
references in this repository's documentation are candidates, not
commitments. The architecture stays algorithm-agnostic behind the Policy
interface (ADR-0004).

## Justification

Every algorithm family's viability hinges on the same few numbers; measuring
them is cheaper than implementing any one algorithm. The Policy interface
means deferring costs nothing structurally.

## Consequences

- Phase 2 of the roadmap begins with a rule-based baseline (no gate needed)
  in parallel with benchmarking.
- `docs/benchmarking.md` must define the decision framework: which
  measurement answers which architectural question.
- Slight delay before any search code; accepted.

**Update (2026-07-05)**: the "per-decision time budget on Kaggle" unknown
cited in Context is resolved (research question Q1) — there is no per-move
timeout (`actTimeout=0`); the real constraint is a 2000 s whole-episode
budget (`runTimeout`), shared across both agents + the engine. See
`docs/competition_analysis.md`. This removes timeout risk as a factor in
ADR-0010; the remaining gate is Search API throughput/memory under 2 vCPUs
and 12.2 GiB, per the revised framework in `docs/benchmarking.md`.
