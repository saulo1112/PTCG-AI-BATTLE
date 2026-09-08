# Observability

"Why did the agent do that?" must be answerable in seconds (ADR-0009).
Three tools, all in `ptcg_ai.debug`, all wired into the CLI.

## 1. Observation pretty-printing — `ptcg show-obs`

```bash
uv run ptcg show-obs tests/fixtures/observations/004_MAIN_MAIN.json --logs
```

Renders the board (actives/bench with HP and energy, hands, prizes, deck
counts, status conditions, turn flags) and the decision request as a
numbered option list — with card *names* when the SDK is loadable, raw IDs
otherwise. Works on any raw observation JSON: fixtures, episode steps, or
an observation dumped from a live run.

Formatters (`debug/inspect.py`): `format_observation`, `format_select`,
`format_option`, `summarize_logs`. They never raise — including on unknown
enum values (`UNKNOWN_<n>`), which is itself a schema-drift signal.

## 2. Decision traces — `ptcg battle --trace`

```bash
uv run ptcg battle --trace build/trace.jsonl
```

`TracingPolicy` wraps any policy transparently and records one
`TracedDecision` per call:

| Field | Meaning |
|---|---|
| `index`, `turn`, `player` | position in the game |
| `select_type`, `select_context` | what kind of decision |
| `n_options`, `min_count`, `max_count` | decision shape |
| `chosen` | the returned option indices |
| `elapsed_ms` | policy latency (feeds the per-episode budget watchdog, ADR-0006/Q1 — no per-move limit exists, but the 2000 s episode total does) |
| `note` | the policy's own explanation (`last_note`) |
| `policy` | which policy produced it |

The CLI prints a per-decision report table after the game and saves JSONL
for offline analysis. **Convention for policy authors**: set `last_note` to
a short "because…" every time you make a non-obvious choice — rule name,
score, or search statistic. That string is the difference between a debug
session of seconds and one of hours.

## 3. Episodes as replayable artifacts — `ptcg battle --record`

Episodes store raw observations + actions (ADR-0008), so any past game can
be re-inspected with *current* tooling: load the JSONL, feed any step's
`raw_obs` to `show-obs`, or re-parse the whole game to test a new policy's
opinion of old decisions.

## Timing information

Every trace row carries `elapsed_ms`; the report footer aggregates p50/max/
total. Q1 is resolved: there is no per-move limit, but episodes share a
2000 s wall-clock budget — a cumulative-budget watchdog in `SafePolicy`
becomes the enforcement point (roadmap Phase 4).

## Engine-side visualizer

The engine accumulates spectator-view frames (`visualize_data()`, exposed as
`BattleEnvironment.visualize_json()`); the official HTML viewer comes with
the kaggle-environments harness (`env.render(mode="html")` on Linux/WSL).
Wiring a local viewer for the raw JSON is deliberately deferred (roadmap).
