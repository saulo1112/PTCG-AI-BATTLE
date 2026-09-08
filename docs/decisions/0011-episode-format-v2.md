# ADR-0011: Episode format v2 — terminal observation, timing, gzip

- **Status**: Accepted
- **Date**: 2026-07-05

## Context

Phase 1 turns episodes into the primary research dataset. The v1 format
(ADR-0008) has measured gaps: the runner exits its loop before the terminal
observation is ever recorded (so final turn count, end-of-game events, and
the RESULT log are absent from stored data); no per-decision timing exists
for the non-traced player; game wall-clock duration is not captured; and at
~1 MB/game raw, large datasets want compression.

A design review also confirmed that `search_begin_input` must be **kept**:
it is ~12% of bytes pre-gzip (negligible after), and stored verbatim it
makes every recorded decision point a ready-made seed for the Phase 2
search-viability benchmarks (ADR-0006/0010).

## Problem

Amend the episode format to support analytics without breaking the
raw-first principle of ADR-0008 or existing v1 files.

## Alternatives

1. **Parallel side-files** (timing.jsonl, final.json per episode) — keeps v1
   untouched but adds join logic everywhere.
2. **Additive v2 in the same JSONL** — one file remains the unit of record.

## Decision

Episode format v2, additive: the `meta` line gains `"version": 2`; each
`step` line gains optional `"elapsed_ms"` (measured by the runner around the
agent call — includes parse overhead, for BOTH players); a new
`{"kind": "final", "raw_obs": ...}` line carries the terminal observation;
the `outcome` line gains `"duration_s"` (battle wall-clock). Files ending in
`.jsonl.gz` are transparently gzip-compressed. `search_begin_input` is
retained in all stored observations. Loaders accept v1 files (missing
fields default to None).

## Justification

Closes every measured gap with one backward-compatible format change;
gzip alone yields ~10× size reduction at negligible CPU cost; the terminal
observation is required by four of the ten Phase 1 research questions.

## Consequences

- Extractors must guard against the terminal observation's stale `select`
  (documented in battle_flow.md) — it is never a decision.
- v1 episodes remain loadable but lack final/turn/duration data; the
  fixture-capture flow is unaffected.
- Recorded corpora double as Phase 2 search-benchmark seeds.
