# M24 findings — pooling three pilots of the SAME deck — NEGATIVE (2026-07-23)

**Bottom line: pooling the decisions of three independent top-20 pilots who run the EXACT same
THIRD PTCG Club deck (Team Rocket Rush + Mewtwo ex/Articuno) into one behavior-cloning dataset does
NOT improve clone fidelity — it DEGRADES it. On a fixed held-out set of THIRD's own MAIN decisions
(same 1334 rows scored for every variant), the pooled-3 clone scores 0.570 vs solo-THIRD's 0.594:
a paired −0.024, 90% CI [−0.041, −0.006], entirely negative. The pre-registered escalation gate
(pooled must beat the best individual by ≥+0.02 with CI-lo > 0) FAILS. Nothing in production was
touched: no bundle retrained, no `.tar.gz` rebuilt, no upload. The mechanism is clean and is the
real finding: the three "same deck" pilots make MEANINGFULLY DIFFERENT MAIN decisions, so pooling
their data dilutes the THIRD-specific policy rather than reinforcing it. This is the first-ever
pooling attempt in the project, and it closes "more data from other pilots of the same deck" as a
fidelity lever for a SPECIFIC-teacher clone.**

Continues [m23_findings.md](m23_findings.md). The M24 hypothesis (more independent skilled pilots
of the same exact deck → higher clone fidelity without more model capacity) is contradicted by its
own held-out measurement.

## Why this was worth a cheap test

The shipped `imitation-third` clones ONE pilot (THIRD PTCG Club, #12/1117) with 160 games. M23
discovered the identical 23-card deck is also piloted by {{ team_name }} (#14, 1106, sid 54900147)
and Dries @ Tufa Labs (#19, 1094, sid 54896167). Pooling is the safe, capacity-free bet: keep the
linear scorer (the MLP already lost the ladder in M11/M14 by overfitting one teacher's
idiosyncrasies), just feed it more data of the same deck. The whole experiment is offline, additive
and reversible — no `src/` change (the `THIRD_PTCG` profile already exists), no production artifact.

## Setup (all reused, nothing re-downloaded)

- Replays already on disk: `replays/54840044` (THIRD, 161), `replays/54900147` (team_name, 141),
  `replays/54896167` (Dries, 161).
- Full datasets already built (the `_screen` files WITHOUT the `_sub` suffix are full-data):
  `third_full.jsonl.gz` (160 games / 10917 rows), `teamname_screen.jsonl.gz` (140 / 9285),
  `driestufalabs_screen.jsonl.gz` (160 / 10293).
- Pooled dataset = byte-concatenation of the three gzips (valid multi-member gzip):
  `third_pooled3.jsonl.gz` = **454 unique game_ids, 30495 rows, 19549 MAIN rows**.
- Four variants trained/scored, all `profile=THIRD_PTCG`, `deck=thirdptcgclub.csv`, seed 0:
  solo-THIRD (reused `bc_third_ptcg.json`), solo-Dries (`bc_third_dries.json`), solo-team_name
  (`bc_third_teamname.json`), pooled-3 (`bc_third_pooled3.json`).

### The leak concern the plan flagged — verified clean
`game_id = path.stem` = the globally-unique Kaggle episode id. Overlaps between the three sources
are tiny (third∩team=1, third∩dries=3, team∩dries=2 = 6 mutual matches where two of these players
faced each other). Because `split_by_game` hashes `seed:game_id`, an identical id lands on the SAME
split side in every source, so concatenation cannot leak a game across the train/val boundary. The
evaluator asserts it directly: **THIRD-val games that landed in the pooled TRAIN split = 0.** The
~6 mutual matches are double-counted (~1.3%) but they are different seats/pilots of the same deck —
that is exactly the signal we wanted to pool, not a leak.

## The measurement — one fixed held-out set, same rows for every variant

Each `train_bc` run splits its OWN rows, so the `weighted_bc_acc` printed per model is on a
DIFFERENT val set and is not comparable. `scratchpad/eval_fidelity_pooled.py` fixes this: it defines
**E = the MAIN-context decisions in the val partition of THIRD's dataset (seed 0) = 1334 decisions**
and scores every model on E. No variant trained on E (solo-THIRD held it out; solo-Dries/team_name
never saw THIRD games; pooled excluded them via the same hash), so it is a clean held-out measure of
fidelity to the SHIPPED teacher.

### Primary: MAIN accuracy on E (THIRD held-out, 1334 rows, identical for all)

| variant | teacher(s) | train MAIN rows | acc on E |
|---|---|---|---|
| **solo-THIRD** | THIRD only | 4283 | **0.5937** ← best individual |
| solo-Dries | Dries only | 5608 | 0.5075 |
| solo-team_name | team_name only | 4928 | 0.4783 |
| pooled-3 | all three | 16024 | 0.5697 |

- Sanity: solo-THIRD on E = 0.5937 reproduces the ~0.594 MAIN val already reported for
  `bc_third_ptcg.json` — validates that E and the evaluator are correct.
- **pooled-3 − best individual (solo-THIRD) = −0.0240, 90% CI [−0.0412, −0.0060]** (paired bootstrap
  over the 1334 decisions, 5000 resamples). CI entirely below zero.
- **Pre-registered gate: FAIL.** (Required pooled−best ≥ +0.02 AND CI-lo > 0 AND pooled ≥ solo-THIRD;
  none holds — the sign is wrong.)

### Secondary diagnostic: MAIN accuracy on the pooled held-out (union val, 3525 rows)

| variant | acc on E_union |
|---|---|
| solo-THIRD | 0.5455 |
| solo-Dries | 0.5606 |
| solo-team_name | 0.5535 |
| **pooled-3** | **0.6048** |

pooled-3 is best here — but this is expected and irrelevant to the ship objective: pooled trained on
the union distribution, so of course it fits union-val best. We do NOT ship an average-of-three-
pilots clone; we ship a THIRD clone, and for that (E) pooling is strictly worse. Reporting it only to
be honest that pooling produces a better *generic-pilot* model, just not a better *THIRD* model.

## The mechanism (the real finding)

The cross-pilot numbers on E are the story. Each solo clone scores far LOWER on THIRD's held-out
decisions than on its own:

| clone | acc on its OWN val | acc on THIRD's held-out (E) | drop |
|---|---|---|---|
| solo-Dries | 0.684 | 0.5075 | −0.176 |
| solo-team_name | 0.622 | 0.4783 | −0.144 |

A Dries-trained clone is a 0.684-fidelity model of Dries but only a 0.508 predictor of THIRD's MAIN
choices — barely above what THIRD's own greedy baseline (0.387 on this set) plus a little transfer
buys. **The three pilots run identical 60-card lists but make materially different main-phase
decisions** (attack/attach/bench sequencing), so their pooled optimum is a compromise policy that
matches none of them as well as a dedicated clone does. Pooling reinforces only the decisions the
pilots AGREE on and averages away each one's specifics — and the shipped clone's value is precisely
THIRD's specifics.

This is the deck-level analogue of the M8/M23 lesson (clonability of the *exact build* > teacher
elo): here it becomes clonability of the *exact pilot* > volume of same-deck data. More rows help
only if they come from the same decision policy; rows from a different policy on the same deck are
closer to label noise for the target teacher.

## Verdict & recommendation

- **NEGATIVE. Pooling multiple same-deck pilots does not raise fidelity to a specific teacher; it
  lowers it.** Gate failed with a clean negative CI. First pooling attempt in the project; the lever
  is closed for the specific-teacher-clone objective.
- **Nothing shipped or changed in production.** `build/imitation-third.tar.gz` and
  `data/models/bc_third_ptcg.json` are untouched; no `.tar.gz` rebuilt; no upload. No `src/` change.
- **Do not escalate.** The pre-registered escalation (retrain the bundle on pooled weights + re-run
  `head_to_head.py --baseline` vs kanga and vs `bc_third_ptcg.json`) was gated on a clear positive
  and is therefore NOT taken.
- Honest caveat, consistent with M14/M21/M22: offline MAIN accuracy is a proxy the ladder has
  overruled before. But here the proxy moved the WRONG way with a clean CI, and the mechanism
  (pilots disagree on the same deck) is a direct, non-speculative measurement — there is no positive
  signal to spend a scarce slot on.
- If pushing further, the standing M23 recommendation is unchanged: the remaining lever is a
  *different clone-friendly deck in the ~900–1050 band* (where 懒惰的金枪鱼 and ITF came from; ~270
  unscreened M22 candidates remain), not more data on the THIRD deck and not a higher-elo teacher.

## Files

New (all deletable, none in production): `docs/m24_findings.md`,
`data/imitation/third_pooled3.jsonl.gz`,
`data/models/{bc_third_dries,bc_third_teamname,bc_third_pooled3}.json`,
`scratchpad/eval_fidelity_pooled.py`. No `src/` change; no bundle built; nothing uploaded.
