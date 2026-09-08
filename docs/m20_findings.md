# M20 findings — mining discrete decision patterns from strong players' replays (2026-07-21)

**Bottom line: NEGATIVE on the pre-registered criterion (K-D fired). Across 13 players
spanning 586–1223 elo and ~21k player-turns, no discrete, cross-player-replicated,
deck-controlled behavioural divergence clears the bar. The two candidates that survived a
weaker version of the gate both turned out to be INSTRUMENT ARTIFACTS, each caught by a
control that was added because the sanity check refused to pass. What DOES replicate is one
real but *diffuse* difference — strong players convert fewer of their play opportunities into
cards played (resource conservation) — and it is precisely the wrong shape to act on: it is a
scorer-level stylistic bias, not an if-then rule, which is the exact class of change M11/M14
proved cannot be validated offline and made real play worse. Line CLOSED. imitation-v1
(linear, ~686) remains the champion.**

This tested the M20 hypothesis raised after [m17_findings.md](m17_findings.md): rather than
cloning strong players' decisions wholesale (fails — M8/M9/M14) or importing rules from
external strategy guides (fails — M17, most claims were Pokémon TCG *Pocket* or absent
cards), mine *discrete, explainable* patterns from the replays already on disk and audit each
against hard engine facts, the way M17's one surviving rule (T3 prose damage) was derived.

New: `scratchpad/mine_patterns.py` (dev-only), `data/rl/m20_mine.{json,log}`.

---

## What was available (verified, not assumed)

The user's uncommitted 2026-07-21 candidate batch turned out to contain the strongest teacher
data this project has ever had. Scores read live from the Kaggle episode API
(`tools/find_candidates.py --ours 54555926 54556007`):

| pool | player | score | turns mined |
|---|---|---|---|
| strong | 懒惰的金枪鱼 (`replays/54708568`) | **1222.8** | 3494 |
| strong | jiatu.l (`replays/54611538`) | **1084.6** | 2657 (held out) |
| strong | vibechu / Majkel1337 (M6's #1/#2) | top-tier | 349 / 286 |
| strong | kenN2439, sam_the_rice_cake, [RU] Nikita Kuznetsov, shu, Kazama Yusuke | 940 / 897 / 800 / 776 / 729 | 2452 / 511 / 2587 / 218 / 630 |
| teacher | greengreenpurple — **the bot imitation-v1 clones** | 650 | 2107 |
| weak control | PP.TAKEHIRO_KAWADA, Team Rocket, vvhan | 706 / 646 / 586 | 342 / 1129 / 602 |

The weak control pool is not decoration: without it, "strong pool vs the one 650 bot" cannot
separate piloting skill from that one decklist's idiosyncrasies. It is what killed half the
candidates (below).

## Method

Each MAIN decision is reduced to deck-independent coordinates — `OptionKind`
(PLAY/ATTACH/EVOLVE/ABILITY/ATTACK/RETREAT/END, the taxonomy the BC featurizer already uses)
plus a situation cell built only from quantities `GameState` already derives
(`dies_if_pass` via `max_threat`, attack availability, `reserve_attackers`, prize race). The
statistic is a **conditional** rate — a kind is counted only over turns where it was actually
offered — so "strong players retreat more" can never be an artifact of their deck offering
more retreats.

**Pre-registered K-D:** a cell must clear support ≥40 in both pools, ≥3 discovery players
agreeing in direction, |gap| ≥ 0.25 vs the teacher, the held-out player (jiatu.l, excluded
from discovery) within 0.15 of the strong pool, and the weak-pool control moving the same way
by at least half the gap.

## The instrument had to be fixed twice, and the sanity check is why

This is the most transferable part of M20. The pre-registered sanity check was: the harness
must rediscover a fact we already know from M10 — strong pilots essentially never miss a
lethal (v1 was measured at 0/256).

**Failure 1 — wrong unit.** The first version measured per-*decision* rates and read lethal
discipline as **0.10–0.40**, which would have been a spectacular false finding. Cause: MAIN is
a multi-decision phase (measured: 74 decisions across 17 turns in one game), so a per-decision
rate dilutes every strategic choice with the setup clicks preceding it in the same turn.
Fixed by making the unit the **player-turn** (state read at the turn's first decision;
availability and choice unioned over the turn). Lethal discipline then read **0.92–1.00**
across all 13 players — the instrument now reproduces known ground truth.

**Failure 2 — the denominator.** With the turn-level harness, two candidates survived:
"strong players END their turn without attacking far more often" (+0.310) and "strong players
play fewer cards" (−0.268). Adding attack-availability as an axis destroyed the first: the END
rate had been silently mixing *chose* not to attack with *could not* attack at all, which is a
property of a deck's energy curve, not of piloting. The teacher's Team Rocket swarm can attack
from turn 1 (Rocket Rush is cheap); slower decks cannot. The entire +0.310 was that.

Two further families died to the weak-pool control alone, and they are worth naming because
they are the traps a less-controlled version of this study would have shipped:

| apparent finding | gap vs teacher | why it is not skill |
|---|---|---|
| strong players ATTACH energy far more | up to **+0.876** | teacher's Rocket Rush needs 2 energy total; other decks need 3–4 per attacker |
| teacher EVOLVES every single turn it can (1.000) | **−0.460** | TR is a single 1-step line; strong decks hold evolutions |

## Result: K-D fired

Best surviving gaps after both corrections (full table in `data/rl/m20_mine.log`):

| kind | cell (dies / can-attack / bench-ready / prize) | n | strong | 650 | gap | players agree | holdout | weak pool |
|---|---|---|---|---|---|---|---|---|
| PLAY | F / T / T / ahead | 679 | 0.673 | 0.897 | −0.224 | 6 | 0.792 | 0.912 |
| END | F / T / F / behind | 177 | 0.232 | 0.011 | +0.221 | 3 | — | 0.000 |
| RETREAT | T / T / T / ahead | 357 | 0.146 | 0.000 | +0.146 | 6 | 0.015 | 0.000 |

**No cell clears the pre-registered bar.** Per the same discipline M19 applied, the criterion
is not relaxed after seeing the data.

Note the a-priori favourite named in the plan — "strong players retreat instead of attacking
when the Active is about to die" — is **not supported**. Retreat is rare for everyone
(0.03–0.21 in every cell), the largest gap is +0.146, and both the held-out player (0.015) and
the weak control (0.000) contradict the strong pool rather than confirming it.

## The one thing that does replicate — and why it is still not actionable

`PLAY` at the turn level is coarse ("played ≥1 card"), so the harness also measures
*intensity*: cards actually played per play-offering decision. That signal is consistent in
every cell where it can be measured, and the held-out player tracks the strong pool:

| cell | strong | teacher 650 | weak pool | holdout (jiatu.l) |
|---|---|---|---|---|
| can attack, bench ready, ahead | **0.322** | 0.581 | 0.459 | 0.392 |
| can attack, no bench, behind | **0.540** | 0.658 | 0.574 | 0.538 |
| cannot attack, bench ready, even | **0.516** | 0.670 | 0.764 | 0.499 |

Strong players hold resources; the 650 teacher — the bot imitation-v1 clones — dumps its hand.
That is a genuine, replicating, elo-tracking difference, and it is the user's own hypothesis
("¿cuándo guarda una carta en mano en vez de jugarla?") confirmed as directionally real.

**It is nevertheless the wrong shape to ship, for two independent reasons:**

1. **It is diffuse, not discrete.** It does not localize to a board situation; it is a global
   propensity (~0.32–0.54 vs ~0.58–0.67) present in every cell. There is no if-then condition
   to write. Implementing it means biasing the MAIN scorer against PLAY — a subtle,
   everywhere-active change to the champion's policy. That is *precisely* the M11/M14 class:
   an offline-motivated scorer tweak, unverifiable by any gauntlet we have (M14 showed a known
   +0.13 ladder gap reads as −0.003), and the last two times we shipped one on offline evidence
   it made real play worse.
2. **It remains partly confounded.** The intensity denominator is endogenous — playing a card
   generates the next play-offering decision — so the metric is close to a re-encoding of
   "cards played per turn", which is also a function of how many playable cards a decklist
   draws. The weak-pool control narrows this but does not eliminate it.

## Known instrument limitations (documented, not worked around)

- **Prose damage blinds `has_lethal`.** `GameState.attack_damage` is structured-only (the M17
  T3 finding). Measured here: for vibechu and Majkel1337, **96% and 79% of their offered attack
  options read as 0 damage**, so those two players contribute zero lethal-conditioned turns.
  kenN2439 is similarly under-counted (163 lethal turns from 2452). Any future re-run of this
  study should route damage through `scratchpad/value_features_v2.py` instead.
- `dies_if_pass` uses the same structured `max_threat` and is under-counted for prose-attacking
  opponents.
- vibechu/Majkel1337 have no live score in our episode history (we never faced them); they are
  ordered by M6's leaderboard position, and their placeholder 1000.0 is never used as a number.

## Verdict

- **K-D fired as pre-registered → the "mine discrete rules from strong replays" line is
  CLOSED.** Fase B (hard-fact audit) and Fase C (implement + gate) did **not** run; no Kaggle
  slot spent, the champion untouched.
- Cost: one day, all of it CPU-cheap offline analysis. No `src/` change.
- This is the sixth consecutive confirmation (M11, M14, M16, M17, M19, M20) that the remaining
  gap to the >1000-elo band is not reachable through a subtle, offline-validated adjustment to
  our pilot. M20 adds something the previous five did not: it shows the gap is not stored in
  *discrete, situation-triggered decisions* at all. Across 21k turns the strong players and the
  650 teacher make the same categorical choices in the same board situations; where they differ
  it is by degree (resource conservation), not by rule.
- **Standing recommendation unchanged and now better evidenced: let imitation-v1 (~686) ride.**

## Reusable

- `scratchpad/mine_patterns.py` — turn-level behavioural contrast harness. Reads either replay
  directories or prebuilt `data/imitation/*.jsonl.gz`, needs no native engine beyond the card
  DB, runs the full 13-player study in ~6 minutes. The two controls it now carries
  (attack-availability axis, weak-pool deck control) are the load-bearing parts; the sanity
  block is what makes its output trustworthy at all.
- The candidate scores mined here (`replays/*` at 586–1223) are a live map of the ladder field,
  reusable by `quick_screen.py` for the parallel teacher search.
