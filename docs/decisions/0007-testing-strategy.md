# ADR-0007: Testing via captured fixtures + SDK-gated integration tests

- **Status**: Accepted
- **Date**: 2026-07-05

## Context

Most of our logic (parsing, option handling, policies, packaging) operates on
observation dicts. Real observations require the native engine library, which
is present on dev machines but should not be a prerequisite for running the
bulk of the test suite (CI, quick iteration, license caution about copying
engine bits around).

## Problem

How do tests get realistic observations without every test loading the
native engine?

## Alternatives

1. **Hand-write mock observations** — no engine needed; but hand-written
   dicts encode our assumptions, not the engine's actual output.
2. **Capture real observations once, replay as fixtures** — ground truth,
   diffable, engine needed only at capture time.

## Decision

A `ptcg capture-fixtures` command runs one real local battle and saves raw
observation dicts as JSON fixtures (one per distinct decision signature)
under `tests/fixtures/observations/`. Unit tests run against fixtures only.
Tests that need the live engine are marked `@pytest.mark.sdk` and auto-skip
when the native library is unavailable.

## Justification

Fixtures are engine ground truth, human-inspectable, and double as schema
snapshots: when a new SDK drop changes the observation shape, re-capturing
fixtures diffs the change (the drift detector for ADR-0005).

## Consequences

- Fixtures are committed JSON; regeneration procedure documented in
  `docs/developer_guide.md`.
- Fixture set must be re-captured after SDK updates and occasionally enriched
  (rare decision kinds appear only in some games).
- JSON, not pickle: fixtures must not couple to any class definitions.
