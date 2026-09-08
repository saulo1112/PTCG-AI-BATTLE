# ADR-0005: Forward-compatible observation parsing

- **Status**: Accepted
- **Date**: 2026-07-05

## Context

`cg/api.py` states verbatim: *"new elements may be appended to the Enum
during the competition"* and *"new attributes may be appended to each class
during the competition."* An agent that crashes on an unknown enum value or
dict key loses the game on the spot, and rating is win/loss only.

## Problem

How do we turn raw observation dicts into typed objects without becoming a
crash risk when the schema grows mid-competition?

## Alternatives

1. **Use `cg.to_observation_class` everywhere** — zero code; but couples all
   logic to vendored types (ADR-0002) and its strictness under schema drift
   is not under our control.
2. **Own mirror models with tolerant parsing** — small mechanical parser we
   control end-to-end.

## Decision

`ptcg_ai.observation` defines mirror dataclasses using the SDK's field names
verbatim (so the official docs remain the reference), each carrying an
`extras: dict` for unrecognized keys. Enum fields use tolerant IntEnums whose
`_missing_` fabricates an `UNKNOWN_<n>` pseudo-member instead of raising.
Parsing never raises on additive schema changes.

## Justification

Robustness is a ranked-play requirement, not a nicety. Field names verbatim
(camelCase) keep a 1:1 correspondence with SDK docs and raw JSON, minimizing
translation bugs; `extras` makes schema drift *observable* (tests assert
it's empty today, and a monitor can flag when it stops being empty).

## Consequences

- camelCase fields in our models deviate from PEP 8 — accepted deliberately
  and documented in `observation/models.py`.
- Mirror enums must be updated when we *want* to use new values; unknown
  values are safe but nameless until then.
- Fixture regression tests (ADR-0007) are the drift detector.
