# ADR-0004: One universal Policy interface for all decision types

- **Status**: Accepted
- **Date**: 2026-07-05

## Context

The engine surfaces heterogeneous decision points through one mechanism —
`SelectData` with typed options: initial deck submission, going-first choice,
mulligan, setup placements, main-phase actions, mid-effect sub-selections
(targets, counts, discards), coin calls, special-condition picks. The
baseline ladder (Random → Greedy → Rule-Based → Search → Learning) requires
swapping decision engines without touching the agent loop.

## Problem

One decision interface for everything, or per-decision-type handlers?

## Alternatives

1. **Per-SelectContext handler registry** — fine-grained; but ~49 contexts
   (growing mid-competition), and every policy must wire all of them.
2. **Universal `Policy.choose(DecisionContext) -> list[int]`** — one entry
   point; specialization happens *inside* policies that care.

## Decision

Every decision flows through `Policy.choose(ctx) -> list[int]` (option
indices). Deck submission — the one call that returns card IDs instead of
indices — is the separate `Policy.choose_deck(ctx)`. Policies are
algorithm-agnostic plug-ins behind this interface, registered by name in
`decision.registry`.

## Justification

The engine already guarantees legality by enumeration, so uniform index
selection is safe for any decision kind. New SelectContexts added
mid-competition degrade gracefully (a policy that doesn't recognize a
context can still select validly). Every rung of the baseline ladder is a
drop-in replacement.

## Consequences

- `DecisionContext` must carry everything any policy could need (raw dict,
  parsed observation, card database); it grows additively.
- Sophisticated policies dispatch on `SelectKind`/`SelectContext` internally —
  complexity lives in the policy, not the framework.
- The safety wrapper (`SafePolicy`) can wrap any policy uniformly.
