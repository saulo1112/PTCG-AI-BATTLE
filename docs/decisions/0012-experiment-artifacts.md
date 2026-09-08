# ADR-0012: Experiment artifacts and manifest

- **Status**: Accepted
- **Date**: 2026-07-05

## Context

Phase 1 introduces batch data collection (`ptcg collect`) and analysis
(`ptcg analyze`). Findings are only trustworthy if every dataset can state
exactly how it was produced. The engine's battle RNG is not seedable
(`random_device`), so *game-level* reproducibility is impossible — the
reproducibility contract must be defined honestly.

## Problem

Where do experiment outputs live, and what metadata makes them citable?

## Alternatives

1. Ad-hoc output paths per run — fastest, unauditable.
2. A structured experiment directory with a manifest — small fixed cost.

## Decision

Each collection run writes `data/experiments/<name>/`:

```
manifest.json      # git SHA, package version, profile + config snapshot,
                   # policy names + POLICY seeds (engine RNG not seedable),
                   # deck lists per side, games requested/completed/aborted,
                   # timestamps, host platform
episodes/NNNNN.jsonl.gz   # format v2 (ADR-0011)
report.json / report.md / tables/*.csv   # written by `ptcg analyze`
```

`data/` is gitignored; findings are summarized by hand into committed docs
(`docs/game_analysis.md`) citing the experiment name + git SHA.
Collection is fault-tolerant: a per-game exception is logged and counted in
the manifest (`aborted`), never killing the run.

**Reproducibility contract**: analysis is 100% reproducible from stored
episodes; game *generation* is not (engine RNG). The manifest records
everything under our control so any number in a report is traceable to
code + config + data.

## Justification

Minimal structure that makes every reported statistic auditable and lets
identical pipelines re-run against future policy rungs for comparison.

## Consequences

- Reports must always cite experiment name + generating policy (random-play
  bias caveat, docs/methodology.md).
- Large corpora live outside git; deleting `data/` loses raw games but no
  committed findings.
- `analyze` must stream episodes one at a time (datasets exceed RAM).
