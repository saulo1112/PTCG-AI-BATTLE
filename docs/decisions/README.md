# Architecture Decision Records

Every significant architectural decision is recorded here using the template
in [0000-adr-template.md](0000-adr-template.md). ADRs are immutable once
Accepted — a change of mind produces a new ADR that supersedes the old one.

| # | Title | Status |
|---|-------|--------|
| [0001](0001-two-artifact-split.md) | Two-artifact split: research package vs submission bundle | Accepted |
| [0002](0002-vendor-boundary.md) | Vendored SDK is read-only, accessed only through the environment adapter | Accepted |
| [0003](0003-config-dataclasses-yaml-profiles.md) | Configuration via frozen dataclasses + YAML profiles | Accepted |
| [0004](0004-universal-policy-interface.md) | One universal Policy interface for all decision types | Accepted |
| [0005](0005-forward-compatible-parsing.md) | Forward-compatible observation parsing | Accepted |
| [0006](0006-benchmark-gated-algorithm-commitment.md) | Benchmark-gated algorithm commitment | Accepted |
| [0007](0007-testing-strategy.md) | Testing via captured fixtures + SDK-gated integration tests | Accepted |
| [0008](0008-episode-storage-raw-jsonl.md) | Episode storage as JSONL of raw observations | Accepted |
| [0009](0009-observability-first-class.md) | Observability as a first-class concern | Accepted |
| [0010](0010-search-scope.md) | Determinized all-contexts receding-horizon search (rung 5) | Accepted |
| [0011](0011-episode-format-v2.md) | Episode format v2: terminal obs, timing, gzip | Accepted |
| [0012](0012-experiment-artifacts.md) | Experiment artifacts and manifest | Accepted |
| [0013](0013-decision-architecture.md) | Layered decision pipeline with a shared evaluator | Accepted |
| [0014](0014-submission-packaging.md) | Bundle a stdlib agent package with a safe-random fallback | Accepted |

ADR-0010 was written 2026-07-07 (M6-0) once the reserved gates were measured:
throughput does not bind, native memory peaks ~33 MiB (Q8 resolved), and
`search_begin` succeeds at every live select context. Determinization quality
(Q4) is handled by arena-gating rung 5 vs rung 4 before trusting it.
