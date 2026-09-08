# ADR-0001: Two-artifact split — research package vs submission bundle

- **Status**: Accepted
- **Date**: 2026-07-05

## Context

The Kaggle submission is a `tar.gz` (≤ 197.7 MiB) with `main.py` at the top
level, a `deck.csv`, and whatever the agent needs at runtime, running on
2 vCPUs / 12.2 GiB RAM. The research work (self-play, evaluation, benchmarks,
tracing, tests) wants a normal installable Python package with development
dependencies that must never leak into the submission.

## Problem

Should the repository BE the submission (develop inside the sample submission
layout), or should the submission be a build artifact produced from a separate
research package?

## Alternatives

1. **Develop inside the submission layout** — zero packaging work; but test
   code, configs, and dev deps pollute the artifact, and the artifact format
   constrains all project structure.
2. **Research package + build step** — clean src-layout, unrestricted tooling;
   costs a small builder module and the discipline that the entrypoint stays
   self-contained.

## Decision

Develop in `src/ptcg_ai/` (any dependencies allowed); a `ptcg_ai.submission`
builder produces the self-contained submission tarball (`main.py` +
`deck.csv` + bundled `cg/`) as a build artifact in `build/`. The tarball is
never edited by hand.

## Justification

Six weeks of iteration happen in the research package; the submission is
regenerated in seconds and validated structurally before every upload. This
also makes "what exactly did we submit" reproducible from git history plus
the builder version.

## Consequences

- The submission entrypoint (`submission/_entrypoint.py`) must remain
  stdlib-only and must not import `ptcg_ai`.
- `ptcg_ai.submission.validate` becomes a mandatory pre-upload gate
  (size, structure, entrypoint smoke test).
- Future learned policies must be exported/frozen into the bundle by the
  builder (weights as files, inference code inlined or vendored).
