# ADR-0003: Configuration via frozen dataclasses + YAML profiles

- **Status**: Accepted
- **Date**: 2026-07-05

## Context

The project needs configuration for paths (SDK location, decks, output dirs),
battle runs, policies, evaluation, benchmarks, and submission building — and
distinct working modes (day-to-day development, benchmarking, building a
submission). The submission artifact itself is stdlib-only and carries no
config machinery.

## Problem

Which configuration mechanism, and how are working modes expressed?

## Alternatives

1. **pydantic / hydra / omegaconf** — rich validation and composition; heavy
   idioms and dependencies for a codebase this size.
2. **Pure Python config modules** — typed, but config-as-code invites logic in
   configs and complicates diffing/experiment tracking.
3. **Frozen stdlib dataclasses + YAML overlay** — schema and defaults in
   typed code; small hand-written loader; YAML files stay declarative.

## Decision

Configuration schema lives in `ptcg_ai.config.schema` as frozen dataclasses
with defaults in code. Named **profiles** (`configs/development.yaml`,
`configs/benchmark.yaml`, `configs/submission.yaml`) overlay selected fields.
Precedence: code defaults ← profile YAML ← programmatic overrides. Unknown
keys are errors (typo safety).

## Justification

`AppConfig()` works with zero files; profiles switch working modes with one
CLI flag (`--profile`); no new dependency beyond `pyyaml`; the loader is
~60 lines we fully control.

## Consequences

- Adding a config field means editing the schema dataclass (deliberate
  friction; keeps config enumerable).
- No config-group composition à la hydra — revisit only if experiment sweeps
  demand it (would be a new ADR).
- Profiles are data, so experiment configs can be diffed and committed.
