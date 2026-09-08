# Research Methodology

The experimental protocol behind every number this project reports. The
conventions here are **normative** — analytics code implements them
(`src/ptcg_ai/analytics/`) and tests enforce them; change them only by
editing this document and the code together.

## The pipeline

```
ptcg collect --name X --games N     # battles → data/experiments/X/ (ADR-0012)
ptcg analyze data/experiments/X     # streaming aggregation → report.{json,md} + tables/*.csv
→ findings summarized by hand into docs/game_analysis.md (citing experiment + git SHA)
```

Every report header names the **generating policy**. This is not decoration:

> **Random-play bias caveat.** Statistics from random-vs-random games
> describe the *decision space* (which contexts exist, branching factors,
> engine behavior) — NOT competent play. Card usage, game length, and end
> reasons will shift under stronger policies. The pipeline is
> policy-agnostic; re-run it at every ladder rung and compare.

## Reproducibility contract (ADR-0012)

The engine's battle RNG is `std::random_device` — **games cannot be
replayed from seeds**. What we guarantee instead:

- Episodes are persisted verbatim (raw observations + actions, format v2,
  ADR-0011) → **analysis is 100% reproducible** from stored data.
- The manifest records everything under our control: git SHA, config
  snapshot, policy names + policy seeds, deck lists, platform, counts.

## Conventions

### Decisions

The terminal observation (`current.result != -1`) carries a **stale
`select`** (counts can exceed its empty option list) — it is **never
counted as a decision**. A "decision" = one step where the engine requested
a selection and an agent answered.

### Event counting (engine events from logs)

The engine delivers each event to BOTH viewers (public events duplicated;
hidden ones as `*_REVERSE`), so naive counting doubles everything. The
convention, validated by `tests/integration/test_log_convention.py`
(TURN_START count exactly equals the final turn number across seeds):

> Per episode, let `v` = the terminal observation's `current.yourIndex`
> (the *counting viewer*). Global event counts = logs of all steps where
> `step.player == v`, plus the terminal observation's logs. This stream is
> complete and non-overlapping; any other single viewer's stream misses the
> tail between their last decision and game end.

**Player-initiated events are counted from chosen actions, not logs**:
attacks, manual attaches, plays, retreats are `OptionKind`s recorded for
both players with no viewer subtleties. Log-derived `ATTACH` additionally
includes ability/trainer-driven attaches — the two measures are reported
separately and labeled ("manual attach" vs "ATTACH events").

### Zone sampling (card presence)

Card-presence-by-zone is sampled at the **first MAIN/MAIN decision of each
turn**: a canonical post-draw state where the acting player's hand is fully
visible and both players' public zones (active/bench/discard) are readable
from the same observation. `State.turn` increments per player-turn, so the
turn number identifies the acting side. Consequence: hand statistics
aggregate over the *acting* player's hands only; public-zone statistics
cover both players every sample.

## Statistics

- **Game-level proportions** (end reasons, winner shares): Wilson 95%
  intervals (`ptcg_ai.utils.stats.wilson_interval`).
- **Decision-level statistics** (branching, latency, context shares):
  decisions within a game are **correlated** — never treat pooled decisions
  as i.i.d. Headline numbers use **per-game means with normal CIs over
  games** (`mean_interval`). Pooled medians/maxima are reported as shape
  descriptors, without intervals.
- Every headline stat prints **n and its CI** — a report answers "do we
  have enough games?" by itself.

### Sample-size guidance (user question 10)

For a game-level proportion near `p`, the games needed for a ±ε CI:
`n ≈ p(1−p)·(1.96/ε)²`. In practice:

| target | games needed |
|---|---|
| end-reason share ±3 pp (p≈0.1) | ~400 |
| end-reason share ±1 pp (p≈0.1) | ~3,500 |
| win-rate difference detection ±3 pp | ~1,000 per matchup |
| rare context stats (e.g. CARD/SWITCH ≈ 0.4/game) | scale n by 1/frequency |

Empirical anchor (random-baseline-1k, N=1000): end-reason shares resolved
to ±1–2 pp; mean decisions/game to ±2.2. **Default protocol: 20-game dry
run to validate the pipeline, then 1000 games; grow only if the CI on the
specific question is still too wide.** Collection costs ~1 min per 1000
random games (14 MB gzipped), so iterating is cheap; slow future policies
are why episodes are persisted rather than regenerated.

## Adding a new metric

1. Extend `analytics/extractors.py` (per-episode extraction) and/or
   `analytics/aggregate.py` (accumulator), honoring the conventions above.
2. Add it to `report.py` with n + CI if it's a headline number.
3. Unit-test the arithmetic on synthetic episodes with known answers
   (`tests/unit/test_analytics.py` pattern).
4. If it needs a new convention, write it into this document first.
