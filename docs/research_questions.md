# Open Research Questions

Prioritized; each with the method that answers it. Answers get recorded
here (with date + evidence link) — this file is the experiment queue.

## Blocking (Phase 1)

- **Q2 — Decision count and shape distribution in real games?**
  Method: decision-count stats from `bench battle` (random) now; re-measure
  with rule-based play in Phase 2 (competent games differ). Feeds time
  budgeting and search planning.
- ~~**Q3 — Search API throughput on 2 vCPUs?**~~ **Resolved 2026-07-05** —
  `search_begin`/`search_step` sub-ms even throttled; see the Answered table
  and benchmarking.md. Latency does not gate search.

## Phase 2

- **Q4 — How sensitive is search quality to determinization accuracy?**
  Method: arena — search with ground-truth hidden info (host knows it) vs
  uniform-random determinization vs belief-tracked. The gap sizes the value
  of investing in beliefs/opponent modeling.
- **Q5 — What does the metagame/card pool support?** Which archetypes in the
  1.2k-card pool are strong under simulator rules? Method: assemble 3–5
  known archetypes from the card list; round-robin arena under the
  rule-based policy; watch top public leaderboard decks if visible.
- **Q6 — How often do draws occur and what causes them?** (engine hard caps,
  mutual deck-out). Method: outcome stats over large arena runs; draws score
  0.5 — if common, tie-breaking play matters.

## Phase 3+

- **Q7 — Schema drift monitoring**: will the organizers actually extend
  enums/attributes mid-competition? Method: fixture drift tests on every SDK
  drop; a submission-side guard that logs unknown values.
- **Q8 — Native memory footprint of large search trees?** Method: process
  RSS while scaling `search_steps`; establishes nodes-per-decision memory
  ceiling under 12.2 GiB.
- **Q9 — Which decisions actually decide games?** (attack choice? energy
  allocation? bench selection?) Method: replay analysis — correlate decision
  categories with outcome flips under search re-evaluation. Focuses rule and
  search effort where it pays.
- **Q10 — Is there exploitable matchmaking structure?** (e.g., meta-decks on
  the ladder to counter-pick against). Method: leaderboard observation;
  likely low leverage, check cheaply. **Partial (2026-07-05)**: the μ≈600 pool
  runs a small set of recipe decks (Mega Lucario ex dominant; Dragapult ex,
  Mega Gardevoir ex, Water mirrors) recoverable in full from every replay —
  see [replay_analysis.md](replay_analysis.md). Counter-picking is possible but
  low priority vs fixing our own play/deck first.
- **Q11 — Does the per-agent ~600 s `remainingOverageTime` budget (and
  `actTimeout=0`) hold on competitive ladder episodes, and as we climb?**
  Method: parse `remainingOverageTime` from every downloaded replay drop.
  **Confirmed on the first 10 competitive episodes** (opponents used ≤13 s);
  re-check each drop — search time-budgeting depends on it.

## Answered

| Q | Answer | Evidence | Date |
|---|---|---|---|
| Q2 (random play) | 51.0 decisions/game mean (median 40, max 213); 14.9 turns; MAIN/MAIN = 70.4% of decisions, mean branching 7.8 (max 50); only 11/49 contexts appear with the sample deck | experiment `random-baseline-1k` (N=1000); docs/game_analysis.md | 2026-07-05 |
| Q6 (partial, random play) | 0 draws in 1000 games; end reasons: 89.8% no-active, 8.7% prizes, 1.5% deck-out — re-measure under competent play | `random-baseline-1k`; game_analysis.md §5 | 2026-07-05 |
| Q10 | ~1000 games → ±1–2 pp on end-reason shares, ±2.2 on mean decisions; formula + protocol in methodology.md; collection costs ~1 min/1000 random games | methodology.md; `random-baseline-1k` CI widths | 2026-07-05 |
| terminal-obs quirk | final observation carries stale select with counts > empty option list; never a decision | fixture `021_terminal.json`; battle_flow.md; extractors guard | 2026-07-05 |
| event-stream convention | events are delivered to BOTH viewers; counting viewer = terminal obs `yourIndex` gives the complete non-overlapping stream (TURN_START count == final turn, exact across seeds) | tests/integration/test_log_convention.py | 2026-07-05 |
| **Q1 — per-move time budget** | **No per-move limit** (`actTimeout = 0`); `runTimeout = 2000 s` whole-episode; **the tighter, per-agent budget is `remainingOverageTime` ≈ 600 s** (per side, counts down). Removes the timeout risk that gated ADR-0010 — search viability hinges on CPU/RAM (2 vCPUs, 12.2 GiB) and the 600 s per-agent budget, not per-decision latency. **Reconfirmed on 10 competitive episodes** (not just validation); see Q11. | validation `Replay.json` (EpisodeId 84143748) + 10 competitive replays; docs/competition_analysis.md, docs/replay_analysis.md | 2026-07-05 |
| Q3 (search throughput @ 2 vCPU) | `search_begin` p50 **0.34 ms**, `search_step` p50 **0.39 ms** (2,345 steps/s) throttled to 2 cores → affordable node-work ≈ 15,000 expansions/decision, ~30× above the "full search on the table" bar. Latency does **not** gate search; memory (Q8) and quality (Q4) do. | `ptcg bench all`, benchmark profile; docs/benchmarking.md baseline table | 2026-07-05 |
