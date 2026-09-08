# M10 findings — trying to EXCEED imitation-v1: no cheap tactical leak, and cheap search degrades it (2026-07-10)

**Bottom line: imitation-v1 remains champion (ladder ~688). M10 was the first
attempt to *exceed* the clone rather than re-clone a new teacher, and it produced
a clear, evidence-backed negative: v1 has no cheap tactical leak to patch, and
BC-guided determinized search (even with a learned value function) plays *worse*
than plain v1, not better.** Nothing shipped — the arena gate did its job. The
milestone leaves behind a diagnosis of exactly why v1 loses, a learned value
function, and the full search-over-BC machinery for a future, larger attempt.

## Phase 0 — where v1 actually loses (86 real ladder games, `Logs/Submission imitation v1/`)

New `scratchpad/diagnose_v1.py` (reuses `kaggle_replay.iter_player_decisions`,
`ObservationParser`/`resolve_option`/`GameState`, `damage_650` for Rocket Rush's
prose damage). Record reproduced: **44W–41L** (+1 self-match).

- **Lethal discipline is PERFECT — 256/256 (100%) available KOs taken, 0 missed.**
  This is the decisive Phase-0 result: it kills the lethal-override idea (M8.1
  measured v2 at 3% misses; v1's Team Rocket clone has none that a structured-
  damage detector can see). No cheap tactical patch exists here.
- **Loss reasons** (direction-only, inferred from the winner's last board per
  `docs/replay_analysis.md`): prize-race 38, bench-out 2, deck-out 1. v1 does NOT
  lose to mill or bricking — it loses fair combat.
- **Losses are bimodal by prize margin**: 14 close (we needed 1–2 more prizes),
  9 mid, **18 blowouts (needed 5–6; in 11 games we scored ZERO prizes)**. The ~18
  blowouts are matchup/variance (unfixable by decision quality); only the ~14 close
  losses are addressable by better play.
- **Diverse loss field**: Mega Lucario 8, Alakazam 7, Cinderace 4, Archaludon 3,
  other 19 — no single villain, reconfirming the M8.1 field thesis.

## Phase 1 — both cheap patches ruled out by evidence

- **Lethal override: dead** (Phase 0: 0/256 misses).
- **`--alpha` win-weighting ablation** (never run before; the hook existed since
  M8): retrained `bc_650` at α ∈ {0.5, 0.7, 1.0}. Offline is **identical** — MAIN
  0.862/0.863/0.863, weighted 0.874/0.871/0.871 (differences are noise). Biasing
  toward the teacher's *winning* games changes nothing, because the teacher plays
  the same in wins and losses (they lose to matchup/variance, not different
  decisions — consistent with the perfect lethal discipline). No gauntlet needed.

Net: v1 has no concentrated, cheaply-fixable loss mode. The only remaining lever
was the sanctioned one — search.

## Phase 2 — BC-guided search with a learned V (the "exceed the teacher" lever)

**Step 1 — learned value function** (`scratchpad/train_value.py` →
`data/models/v_650.json`): logistic regression P(win|state) on the same 7
antisymmetric features `decision/evaluator.py` uses, fit to the 18,533 teacher
decisions' win/loss label (split by game). This isolates "are the hand-tuned
weights suboptimal?" from "are the features enough?".
- **Held-out AUC (rank won-vs-lost states): learned 0.796 vs hand-tuned 0.770
  (+0.026).** The learned weighting is better (prize weight ~2×, deck_out ~10×,
  survival ~4×; reserve flipped negative), so the M6 hand-tuned V *was*
  suboptimal — but both sit at 0.77–0.80, so the 7 features are the real ceiling.
  Gate passed (+0.026 > +0.02) but modestly — a warning the lift might be too
  small to matter.

**Step 2 — `ImitationSearchPolicy`** (`src/ptcg_ai/decision/search_bc.py`,
subclasses M6's `SearchPolicy`): swaps just the two pieces the evidence flags —
seed+fallback = the BC clone (never greedy), leaf eval = the learned V (fixed-root
perspective; sign verified: won states mean V +0.230, lost −0.277). Determinized
search plumbing, `SearchSession` hygiene, fast paths all inherited unchanged.
Compute is Kaggle-safe (~0.7 s/searched decision, ~15–20 s of the 600 s pool;
self-governs via the 6 s/decision cap and 120 s reserve floor).

**Step 3 — arena vs v1 (same deck both sides, isolates the search layer):**

| rollout policy | score vs v1 | n |
|---|---|---|
| greedy (fast) | **0.208** | 120 |
| BC (reflects real play) | **0.320** | 50 |

Both **clearly lose** (BC-rollout 95% CI [0.208, 0.458], upper bound < 0.5). The
BC-rollout fix — motivated by the diagnosis that greedy pilots this deck terribly
(0.21) so greedy rollouts corrupt the leaf board the V judges — helped (+0.11) but
did not close the gap. Note greedy-rollout's 0.208 ≈ greedy's own 0.210 on this
deck (deck-strength probe): cheap search drags v1's excellent MAIN play (0.86)
down toward greedy's level (0.23).

## The lesson (reconfirms M6 with better tools)

On a deck where the BC clone is already excellent and greedy is terrible, **cheap
determinized search degrades strong play**: its overturns of BC's MAIN picks are
net-negative because the leaf V (0.796 AUC, coarse at *move* granularity — adjacent
candidate moves reach near-identical positions) can't reliably rank moves, only
whole positions. M6 lost 0.44 with a hand-tuned V; M10's learned V (+0.026 AUC)
was not a big enough improvement to change the outcome. **Exceeding v1 needs a
materially better value function (richer features — hand composition, card
identity — or a small MLP) and/or deeper search, not the cheap version.** That is
a new-milestone-sized investment, deferred.

## Reusable assets (kept)

- `data/models/v_650.json` — learned leaf V (AUC 0.796).
- `src/ptcg_ai/decision/search_bc.py` — `ImitationSearchPolicy` + `LearnedEvaluator`
  (stdlib-safe; `bc_rollout` flag). Not in the submission allowlist — inert, v1
  unaffected.
- `scratchpad/{diagnose_v1,train_value,arena_m10,audit_search_overturns}.py`. (Note:
  `audit_search_overturns.py` can't drive search offline — the replay dataset's
  observations are stripped of `logs`, which `search_begin` needs; the arena is the
  only valid search test.)

## Status

imitation-v1 (`build/imitation-v1.tar.gz`, ladder ~688) remains the live best and
only recommended agent. greedy-v5, rule-based, M6 search, imitation-v2/v3, and the
M10 BC-guided search are all superseded/rejected. No M10 artifact shipped.
