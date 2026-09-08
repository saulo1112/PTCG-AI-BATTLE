# Benchmarking

Measurements gate architecture (ADR-0006). No search or learning algorithm
is selected until the numbers below exist; the selection is then recorded as
**ADR-0010**.

## Running

```bash
uv run ptcg --profile benchmark bench all      # or: battle | parser | search
```

Reports print as Markdown and are saved under `build/bench/`. The benchmark
profile fixes seeds and mutes logging. **Record Kaggle-relevant numbers on
hardware throttled to ~2 vCPUs** (or scale results accordingly) — dev-box
numbers flatter multi-core.

## What we measure, and what each number decides

| Benchmark | Metric | Decides |
|---|---|---|
| `bench search` | `search_begin` latency | cost of each fresh determinization → how many determinized worlds per decision are affordable |
| `bench search` | `search_step` nodes/s | tree-search viability: nodes/decision = budget_ms × nodes/ms. If a few hundred nodes per decision aren't affordable, MCTS-family approaches die here |
| `bench battle` | battles/s, decisions/battle (p50/max) | self-play data generation rate for any learning approach; expected game length for time budgeting |
| `bench parser` | observations/s | confirms parsing is negligible per move (expected ≫ 10k/s) |
| traces (`--trace`) | per-decision policy latency | the live per-move cost of whatever policy is under test |
| external | process RSS during long searches | native search-tree memory vs 12.2 GiB (tracemalloc cannot see it — watch the process) |

## The decision framework for ADR-0010

**Q1 resolved (2026-07-05, docs/competition_analysis.md)**: there is no
per-move timeout (`actTimeout = 0`). The real constraint is a **2000 s
whole-episode** wall-clock (`runTimeout`), shared by both agents + the
engine, across every decision in the match. This removes the timeout risk
that used to gate search — the framework below shifts from "fits under a
per-move deadline" to "fits under the episode-total budget with room to
spare for CPU/RAM".

Let `E` = 2000 s (per-episode budget), `K` = decisions per episode (mean 51,
median 40, max 213 under random play — docs/game_analysis.md; re-measure
under the policy actually shipping), `b` = search_begin ms, `s` =
search_step ms, `D` = determinizations per decision, `N` = nodes per
determinization:

```
per-decision budget  B ≈ (0.5 × E) / K        (half the episode for margin
                                                + the opponent's/engine's share)
affordable when      D × (b + N × s)  ≲  B
```

With `K`≈50 and `E`=2000 s, `B` is on the order of **10–20 s per decision**
— several orders of magnitude above anything `bench search` has measured so
far (sub-millisecond `search_begin`/`search_step` on this hardware). In
practice this means **latency is very unlikely to be the limiting factor**;
CPU (2 vCPUs) and RAM (12.2 GiB, native search-tree memory) become the real
constraints. Decision rules of thumb (now framed by resource cost, not
timeout risk):
- `D × N ≥ ~500` affordable on 2 vCPUs → full determinized tree search is on
  the table.
- Only `D × N ≈ 50–200` viable under CPU/RAM pressure → shallow search
  (1-ply lookahead + evaluation heuristic) or rule-based with targeted
  search on critical decisions.
- If memory (not latency) is the binding constraint → cap tree size, not
  search depth.

Caveats already known (recorded in `bench_search.py`):
- The random-walk step benchmark mixes deep-line and shallow-expansion
  costs; treat it as an order-of-magnitude estimate.
- Determinization *quality* (belief accuracy) is a separate research
  question (Q4) — throughput viability comes first.

## Baseline numbers

Fill this table as Phase 1 measurements land; keep history (date + machine).

| Date | Machine | battle ops/s | decisions/battle p50 | parser obs/s | begin ms p50 | step ms p50 | notes |
|---|---|---|---|---|---|---|---|
| 2026-07-05 | Win11, 2-core affinity (mask 0x3) | 51.0 | 33 (max 170) | 8,976 | 0.34 (p95 2.02) | 0.39 (p95 0.74; 2,345 steps/s) | `bench all`, benchmark profile; report `build/bench/bench_all.md` |

### ADR-0010 verdict inputs (from the 2026-07-05 throttled run)

Plug the measured latencies into the framework above with the corrected
budget (**~600 s per agent** — the `remainingOverageTime` finding, research
question Q1/Q11, competition_analysis.md — not the 2000 s shared episode
cap):

- `b` (search_begin) ≈ **0.34 ms**, `s` (search_step) ≈ **0.39 ms** on 2
  cores. Both are sub-millisecond even throttled.
- Per-decision budget `B ≈ (0.5 × 600 s) / 50 ≈ 6 s` (half the agent's pool,
  ~50 decisions/episode). Affordable node-work per decision:
  `D × N ≈ B / s ≈ 6000 / 0.39 ≈ 15,000` step-expansions — **~30× above the
  `D × N ≥ 500` bar** that puts "full determinized tree search on the table".
- **Conclusion (evidence, not yet the ADR)**: latency/throughput does **not**
  gate search. The binding constraints are (a) native search-tree **memory**
  under 12.2 GiB (Q8 — RSS probe still owed) and (b) determinization/decision
  **quality** (Q4/Q6, R6), not speed. ADR-0010 is written from these numbers
  in Phase 2 milestone M5.

> Caveat carried from `bench_search.py`: the random-walk step benchmark mixes
> deep-line and shallow-expansion costs — treat `s` as an order-of-magnitude
> figure. It only strengthens the verdict here (we have ~30× headroom).
