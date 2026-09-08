# Risk Analysis

Living register. Severity = likelihood × impact on final rating.
Owner-phase = when the mitigation must be in place.

| # | Risk | L | I | Severity | Mitigation | Owner |
|---|---|---|---|---|---|---|
| R1 | Agent crash / illegal action on Kaggle → forfeited games | M | H | **High** | SafePolicy wraps every submission; entrypoint is exception-proof and clamps counts; `validate-submission` + smoke test before every upload | Phase 0 ✅ |
| R2 | Exceeding the 2000 s per-episode budget by summing many slow decisions (no per-move limit exists — Q1 resolved 2026-07-05, see competition_analysis.md) | L | M | Low | Per-decision `elapsed_ms` already instrumented (episodes/traces); SafePolicy cumulative-budget watchdog in Phase 4; reconfirm 2000 s holds on competitive (not just validation) episodes | Phase 4 |
| R3 | SDK/schema update mid-competition | M | M | Med | Vendor boundary (ADR-0002); tolerant parser (ADR-0005); fixture drift tests; documented update procedure in sdk_analysis.md | Phase 0 ✅ |
| R4 | Premature algorithm commitment wastes the runway | M | H | **High** | ADR-0006 benchmark gate; rule-based fallback always submission-ready | Phase 2 |
| R5 | Search unviable on 2 vCPUs | M | M | Med | It's a gate, not a bet: benchmark first; fallback = stronger rules + targeted shallow lookahead | Phase 2 |
| R6 | Determinization bias (strategy fusion) makes search play weak despite throughput | M | M | Med | Arena-validate search vs rule-based before trusting it; belief-quality experiments (Q4) | Phase 3 |
| R7 | Learning fails to beat rule-based within runway | M | M | Med | Learning is optional (own ADR); Phase 2 agent is the floor; freeze date protects the endgame | Phase 3 |
| R8 | 197.7 MiB cap vs model weights | L | M | Low | Builder reports size; validator enforces the cap; quantization if ever needed | Phase 3 |
| R9 | Real-battle RNG not reproducible → flaky debugging | H | L | Low | Accepted engine property; use episodes/traces for postmortems and deterministic search mode for reproduction | Phase 0 ✅ |
| R10 | SDK single-battle-per-process breaks parallel self-play | H | L | Low | Known upfront: multiprocessing workers, one battle each | Phase 3 |
| R11 | License breach (repo public, SDK redistributed) | L | H | Med | Repo private; SDK never copied into src/; ADR-0002 read-only rule; delete vendor material after the competition | ongoing |
| R12 | Self-play overfitting to our own meta / decks | M | M | Med | Evaluate vs all ladder rungs and multiple decks; watch leaderboard rating vs arena predictions diverging | Phase 3 |
| R13 | Windows-dev vs Linux-Kaggle behavior divergence | L | M | Low | Same engine sources per docs; smoke entrypoint runs on upload (validation episode); optional WSL fidelity runs | Phase 1 |
| R14 | Action-space explosion in odd states (huge option lists) slows search | M | L | Low | Decision-shape stats from fixtures/traces; cap expansions per node if needed | Phase 3 |
| R15 | Schedule: 6 weeks, one engineer | H | H | **High** | Ladder discipline (always a submittable agent); freeze date; scope cuts pre-agreed (deck iteration before exotic algorithms) | ongoing |
| R16 | Bundled `agentpkg/` fails to import/run on Kaggle → silent degrade to safe-random (lost strength, not a forfeit) | M | M | Med | try/except fallback to inline safe-random (ADR-0014); `validate-submission` isolated-import check (only `cg/` + stdlib on path); a live fallback is logged | Phase 2 (M1) |

Retired risks: none yet. Review this table at each phase boundary.
