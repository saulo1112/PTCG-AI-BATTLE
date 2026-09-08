# ADR-0013: Layered decision pipeline with a shared evaluator

- **Status**: Accepted
- **Date**: 2026-07-05

## Context

Phase 2 must turn the safe-random submission into a competitive agent. The
evidence is in place: real ladder replays show all recoverable losses are
turn-5–7 bench collapse ([replay_analysis.md](../replay_analysis.md)); the
action space is tiny (MAIN mean branching 7.8, [game_analysis.md](../game_analysis.md));
card-effect semantics are prose-only, so the only forward model we own is the
engine itself ([feature_inventory.md](../feature_inventory.md) §9,
[sdk_analysis.md](../sdk_analysis.md)); and search throughput has ~30× budget
headroom ([benchmarking.md](../benchmarking.md)). The candidate paradigms and
the full comparison live in [phase2_blueprint.md](../phase2_blueprint.md) §2.
ADR-0006 forbids committing to a *search family* before the benchmark-gated
ADR-0010; this ADR decides the surrounding architecture, not the search
algorithm.

## Problem

What decision-making architecture do the Phase 2 ladder rungs (greedy →
rule-based → +search) share, so that each rung is a drop-in improvement rather
than a rewrite?

## Alternatives

1. **Isolated rule sets per rung** — fastest to a first agent; but rung 5
   search would need its own leaf evaluator built from scratch, duplicating
   rung 4. Rejected: throws away the rung-4 work.
2. **Behaviour tree / FSM / GOAP core** — organisation over rules; no new
   capability against hidden info or prose effects; GOAP additionally needs a
   symbolic action-effect model we do not have. Rejected (blueprint §2).
3. **Neural RL policy/value core** — highest ceiling, worst ROI here (no GPU,
   ~5-week runway, single-battle-per-process self-play, overfitting). Rejected
   as the Phase-2/3 core; retained only as optional linear-evaluator tuning.
4. **Shared pipeline + one evaluator reused as the search leaf** — one
   `GameState`, one context router, per-rung Scorer; the rung-4 utility
   evaluator becomes rung 5's leaf value and in-tree policy.

## Decision

Adopt alternative 4. Every decision flows through
`ObservationParser → DecisionContext → GameState.build → ContextRouter →
per-context Handler → Scorer → SafePolicy`. The router, `GameState`, and
handlers are shared across rungs; **only the Scorer changes**: rung 3 = fixed-
priority rules, rung 4 = `argmax V(predict(state, action))`, rung 5 =
determinized engine search (per ADR-0010) with `V` at the leaves and the
rung-4 policy for in-tree sub-decisions. `V(GameState) → [-1,1]` is a linear
sum of bounded, normalised terms (blueprint §4.2).

## Justification

The evaluator that rung 4 needs is exactly the leaf function search needs, so
building it once makes rung 5 an integration, not a rewrite — the single most
leverage-positive structural choice under a one-engineer, ~5-week budget. A
shared `GameState`/router keeps the ADR-0004 universal-Policy contract and the
ADR-0009 tracing surface intact, and lets each rung be promoted through the
existing Wilson-CI arena gate. Prefer the simpler of equally extensible designs
(review principle 8): a linear evaluator + fixed pipeline is debuggable and
tuneable without touching control flow.

## Consequences

- Creates the Growth-Plan module `state/game_state.py` and, per rung,
  `decision/greedy.py` and `decision/rule_based.py`; `planning/` follows only
  after ADR-0010. No search code is written under this ADR.
- The evaluator's weights become a tuning surface (optional coordinate-descent/
  SPSA against arena win-rate) — the only "learning" retained for the
  competition; anything beyond keeps its own ADR (rung 6).
- Adds fields to `GameState` additively as rules consume them; it stays SDK-
  free and unit-testable from parsed fixtures (architecture.md import rules).
- Search-family choice and scope remain deferred to ADR-0010; this ADR fixes
  the interfaces (`Determinizer.sample`, `SearchSession`, `V` leaf) so that
  decision can be implemented without further architectural work.
- Behaviour-tree/FSM/GOAP and neural-RL cores are explicitly out; revisiting
  them requires a new ADR superseding this one.
