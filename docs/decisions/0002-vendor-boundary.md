# ADR-0002: Vendored SDK is read-only, accessed only through the environment adapter

- **Status**: Accepted
- **Date**: 2026-07-05

## Context

`pokemon-tcg-ai-battle/` (cg SDK + native engine libraries + C++ source +
card CSVs) and `References/` (official PDFs) are third-party competition
material under a competition-use-only license that forbids redistribution.
The organizers may ship updated SDK drops during the competition — the SDK
itself warns that enums and dataclass attributes may grow.

## Problem

How does our code depend on the SDK without being coupled to its layout,
its types, or its license?

## Alternatives

1. **Import `cg` freely across the codebase** — least ceremony; but every SDK
   change ripples everywhere, and vendored types leak into our domain logic.
2. **Copy the SDK into `src/`** — makes imports trivial; but duplicates
   license-restricted code and forks it from upstream drops.
3. **Adapter boundary, SDK referenced in place** — one module owns the import;
   everything else sees our own types.

## Decision

`pokemon-tcg-ai-battle/` and `References/` are read-only vendor directories:
never modified, moved, reformatted, or renamed. Exactly one module —
`ptcg_ai.environment.sdk` — imports `cg`, from its vendored location resolved
via configuration. The submission builder copies `cg/` into the build
artifact at build time only.

## Justification

A new SDK drop is absorbed by replacing the vendored folder; only
`environment/` (adapter) and `observation/` (parser tolerances) may need
changes. License compliance is structural rather than by convention.

## Consequences

- No `import cg` anywhere outside `environment/sdk.py` (the generated
  submission entrypoint is independently self-contained).
- Our own mirror types (`observation.models`, `cards.database`) must be kept
  in sync with SDK drops — the fixture regression tests detect drift.
- The repository must remain private (license).
- Minor cost: one extra hop (`load_sdk`) before any engine call.
