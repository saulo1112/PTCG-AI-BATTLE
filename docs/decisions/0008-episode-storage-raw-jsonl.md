# ADR-0008: Episode storage as JSONL of raw observations + actions

- **Status**: Accepted
- **Date**: 2026-07-05

## Context

Replays serve debugging today and (potentially) training data later. Our
parsed models will evolve; the raw observation dict shape is controlled by
the engine and is the most stable contract we have.

## Problem

Store episodes as parsed/typed objects or as raw engine output?

## Alternatives

1. **Parsed objects (pickle or custom serialization)** — convenient to
   consume; breaks whenever models change; not diffable.
2. **Raw obs dicts + chosen actions as JSONL** — replays survive any parser
   evolution; consumers re-parse with the current parser.

## Decision

`ptcg_ai.replay` stores episodes as JSONL: a metadata line, one line per step
(`raw_obs`, `action`, `player`), and an outcome line. Anything derived
(parsed states, features) is recomputed on load.

## Justification

Storage format outlives code. Raw JSONL is greppable, streamable, and
identical in spirit to the fixture format (ADR-0007) — one mental model.

## Consequences

- Slightly larger files and re-parse cost on load; acceptable at our scale
  (compress later if needed).
- Episode files are append-friendly for long self-play runs.
- A future replay buffer / dataset layer consumes JSONL without migration.
