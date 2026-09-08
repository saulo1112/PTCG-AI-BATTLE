# M9 findings — a new-teacher clone did NOT beat imitation-v1; clone-lift, not deck strength, is v1's edge (2026-07-09)

**Bottom line: all four clonability-screened teacher candidates were exhausted and
none beats imitation-v1. imitation-v1 remains the live champion (ladder ~688).**
M9 built a full replay-download pipeline, screened five ~850–940-elo candidates,
built and gated imitation-v3 (a clone of the ~940-elo Bellibolt pilot `kenN2439`),
and probed two backup decks. The decisive, ladder-validated field gauntlet failed
v3 by a wide margin, and cheap probes ruled out the rest. **Nothing was shipped —
the promotion gate did its job (unlike M8, no submission was spent on a
regression).** The lasting result is a reusable toolchain plus a sharp new lesson.

## New reusable machinery (kept in the codebase)

- **`tools/` replay downloader** — reverse-engineered Kaggle's internal
  `POST /api/i/competitions.EpisodeService/ListEpisodes` (payload
  `{"ids":[], "submissionId":<id>, "successfulOnly":true, "includeInProgress":false}`,
  auth via exported session cookies + `x-xsrf-token`). `kaggle_api.py`
  (`load_session`/`list_episodes`/`fetch_replay`), `download_replays.py` (single
  submission, incremental, rate-limit backoff), `download_competitors.py` (many).
  `make replay SUBMISSIONS="..."`. Cookies live in `tools/cookies.txt.json`
  (gitignored). This makes sourcing any competitor's replays a one-command job.
- **`scratchpad/clonability_screen.py`** — scores every `replays/<sid>/` folder
  against the `docs/handoff.md` §7 checklist (auto-detects player name, fixed-deck
  ratio, Stage-2 lines, MAIN/TO_HAND concentration, macro greedy-as-predictor
  accuracy, gust-trainer usage rate, and elo mean/range/day-span pulled fresh from
  `ListEpisodes`).
- **`scratchpad/deck_strength_probe.py`** — pilots candidate decks with GREEDY vs
  the gauntlet field to isolate raw deck strength from clone quality.
- **`scratchpad/audit_flashing_draw.py`** — the M8.1-style over-click audit,
  generalized (teacher vs clone take-rate on a named ability).
- **`scratchpad/arena_m9.py`** + `field_gauntlet.py` v3 candidate row.
- **Three new `DeckProfile`s** in `deck_profiles.py`: `BELLIBOLT_940` (dim 514,
  shipped-quality), `CINDERACE_METAL` (dim 488, probe-quality). All module-level
  functions, ship-safe (`test_imitation_ship_safety` passes).

## The clonability screen (5 candidates, ~850–940 elo)

| submission | player | fixed deck | Stage-2 | MAIN+TO_HAND | greedy-acc | gust | elo |
|---|---|---|---|---|---|---|---|
| 53815664 | **kenN2439** | 100% | no | 52.9% | 0.429 | 0.0% | 939 |
| 53787873 | Yuzuki | 100% | no | 52.6% | 0.427 | 0.0% | 938 |
| 53909538 | uninc2000 | 100% | **yes (Alakazam)** | 73.8% | 0.471 | 19.2% | 926 |
| 54347432 | Yoshiki Okayama | 100% | **yes (Cinderace)** | 69.0% | 0.474 | 17.5% | 807 |

`kenN2439` won the screen (no Stage-2, zero gust, highest elo, stable 21 days).
Yuzuki runs the **identical** Bellibolt netdeck (same clone target, skipped).

## imitation-v3 (clone of kenN2439, `BELLIBOLT_940`)

Iono's Bellibolt ex Lightning engine (Electric Streamer unlimited {L} attach →
Voltaic Chain board-energy scaling; Kilowattrel's Flashing Draw engine). Dataset
559 games / 77,303 decisions / 0 illegal. The M8.1 lesson was designed in from the
start: the snapshot carries hand **composition** ({L}-in-hand, draw-supporter and
board-energy counts), not just hand size.

**Stage 0 (offline) — PASSED, and cleanly:**

| context | BC | greedy | random |
|---|---|---|---|
| MAIN | **0.770** | 0.234 | 0.151 |
| ATTACH_FROM | 0.646 | 0.108 | 0.191 |
| SWITCH | 0.983 | 0.026 | 0.221 |
| TO_ACTIVE | 0.925 | 0.259 | 0.242 |
| weighted (learned) | **0.736** | 0.197 | — |

MAIN 0.770 sits well above v2's failed 0.694. The Flashing-Draw over-click audit
(the direct check for M8.1's failure mode) came back clean: teacher take-rate
0.401 vs clone 0.453, **gap +0.052** (vs v2.0's Lunar-Cycle +0.195) — the
hand-composition features worked. The degeneracy rule behaved correctly
(`DRAW_COUNT`/`IS_FIRST` → skip→greedy, not wrongly zeroed).

**Clone quality — GOOD.** Pilot-lift vs greedy on the same Bellibolt deck: **0.800**
(n=60), i.e. the clone pilots the deck +0.30 over greedy. The clone is not broken.

**Stage 2 (field gauntlet) — FAILED decisively.** Harness first validated against
the M8.1 record: v1 0.876, v2.0 0.811, v2.1 0.830 (documented: 0.883/0.820/0.832).
Then v3:

| candidate | macro WR over 97 decks |
|---|---|
| v1 (imitation, Team Rocket) | 0.876 |
| **v3 (imitation, Bellibolt)** | **0.662** |

Paired delta **v3 − v1 = −0.214** (90% CI [−0.250, −0.177]), v3 better on only
**7 of 97 decks**, worse on 73. (Head-to-head vs v1 was also lost, 0.160 at n=100,
38/84 losses by deck-out/no-active.) **Not shipped.**

## The central finding — clone-lift dominates deck strength

Why did a well-cloned 940-elo deck lose to a clone of a 650-elo player? The
deck-strength probe (all decks piloted by **greedy** vs the same field) is the key:

| deck (greedy-piloted) | macro WR vs field | clone lift → actual |
|---|---|---|
| Team Rocket (v1's deck) | **0.210** | **+0.666 → 0.876** |
| Bellibolt (v3's deck) | 0.494 | +0.168 → 0.662 |
| Cinderace (Yoshiki) | 0.608 | (not built to gate) |
| Alakazam (uninc2000) | 0.392 | — |

**v1's dominance is not deck strength — Team Rocket is the *weakest* deck under
greedy (0.210).** It is an exceptional clone-lift: greedy pilots the swarm terribly,
the 650 human pilots it superbly, and it clones with very high fidelity (86.2%
MAIN), yielding +0.666. Bellibolt is a stronger raw deck but a simpler, more
greedy-friendly one, so its clone lift is only +0.168 — not enough. **Raw greedy
deck strength does not predict cloned performance; the clone lift does, and it
varies from +0.17 to +0.67 across decks.**

## Backup candidates ruled out cheaply

- **Cinderace (Yoshiki)** — best raw deck (0.608) and has Boss's Orders gust, so
  worth a *clonability* check before investing. Built `CINDERACE_METAL`, trained on
  192 games / 10,788 decisions: offline **MAIN accuracy 0.639** — the worst of any
  clone we have built, below even the failed Lucario's 0.694. The deck is
  mechanically heavy (Memory Dive cross-evolution attacks, Cinderace Stage-2 played
  via its Explosiveness setup ability, Archaludon on-evolve energy accel, Raging
  Hammer self-scaling damage). Discarded: a modest clone lift on 0.608 lands well
  under v1's 0.876.
- **Alakazam (uninc2000)** — worst raw deck (0.392) *and* Stage-2/complex. Ruled
  out without building.
- **Yuzuki** — identical deck to v3; same failure by construction.

## The corrected lesson (extends M8.1)

The clonability screen (handoff §7) optimizes for how *easily* a teacher clones,
but a highly-clonable simple deck often lacks the disruption (gust) that wins
games, while strong disruptive decks are complex and clone poorly. **A candidate
must clear a two-dimensional bar: high clone fidelity AND a deck whose
greedy-floor + achievable clone-lift can exceed the champion's macro WR (~0.88).**
Team Rocket is a rare sweet spot (terrible greedy floor → huge lift, trivially
clonable). Future teacher sourcing should target that profile — fast aggro/swarm
decks greedy misplays but humans pilot well — not just "easy to clone" or "high
elo". A cheap pre-filter now exists: run `deck_strength_probe.py` (greedy floor)
and a train-only MAIN-accuracy read (clone fidelity) *before* committing to a full
gate.

## Status

imitation-v1 remains the live best agent (`build/imitation-v1.tar.gz`, ladder
~688). No M9 artifact shipped. All new tooling, profiles, and probes are kept for
the next attempt.
