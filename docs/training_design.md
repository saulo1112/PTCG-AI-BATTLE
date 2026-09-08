# Training Design (design only — no implementation before its phase)

**Scope disclaimer**: this document designs the *future* learning pipeline.
Per ADR-0006 and the roadmap STOP line, none of it is implemented until the
Phase 2 benchmarks and ADR-0010 exist, and learning itself gets its own ADR
at Phase 3 entry.

## Data flow (the parts that already exist are marked ✅)

```
self-play battles ✅(BattleRunner) ─▶ Episodes (raw JSONL, ADR-0008) ✅
        ─▶ dataset builder (re-parse + features)      [Phase 3]
        ─▶ trainer (paradigm per ADR)                 [Phase 3]
        ─▶ frozen policy artifact                     [Phase 3]
        ─▶ arena gate ✅(run_matchup) ─▶ submission builder ✅
```

Design choices already locked by existing ADRs: episodes store raw
observations (re-parse at training time, so features can evolve freely);
evaluation is arena-based with Wilson intervals; the submission carries a
*frozen* artifact (weights + stdlib-compatible inference or bundled deps
within 197.7 MiB).

## Candidate paradigms vs the 2-vCPU inference constraint

| Paradigm | Data need | Inference cost | Key risk |
|---|---|---|---|
| Behavior cloning from search play | medium (search-policy games) | one forward pass | caps at teacher strength |
| Policy gradient / PPO self-play | very high | one forward pass | training instability, 6-week runway, engine throughput (see `bench battle`) |
| AlphaZero-style (search + value/policy distillation) | high | search × net cost per node — hardest to afford | double budget pressure: per-move time AND training compute |
| Value net + shallow search | medium | small search + cheap net | needs good determinization first |

The battle-throughput benchmark converts directly into "games per GPU-day ×
decisions per game" — the feasibility arithmetic for each row is mechanical
once Phase 1 numbers exist.

## Infrastructure requirements (when built)

- **Parallel self-play**: multiprocessing workers, one battle per process
  (SDK global-state constraint), episodes streamed to disk as JSONL.
- **Versioning**: every artifact tagged with git SHA + config profile;
  checkpoints retained with their arena results.
- **Experiment tracking**: `experiments/` module (Growth Plan) — one config
  + results record per experiment; plain files first, a tracking service
  only if file discipline breaks down.
- **Evaluation-in-the-loop**: candidate vs frozen ladder rungs
  (rule-based baseline is the fixed yardstick); promotion by Wilson-interval
  criterion (docs/decision_system.md).
- **Compute plan**: local CPU suffices for rule-based and search phases;
  learning needs a decision on cloud GPU spend — deferred to the Phase 3 ADR
  with the benchmark numbers in hand.

## Reward / objective notes

Rating counts win/loss/draw only — the terminal objective is binary win
probability (draws = 0.5). Any shaped signals (prize differential, tempo)
are *auxiliary* for credit assignment and must never override the terminal
result in the objective.
