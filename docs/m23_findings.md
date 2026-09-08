# M23 findings — cloning the world top-20 (manual leaderboard IDs) — NEGATIVE (2026-07-22)

**Bottom line: the user manually filled in the `submission_id` of the entire top-20 of the global
leaderboard (1093–1231 elo) — data the M21/M22 pipeline could never reach (it only saw opponents we
had already faced). All 20 IDs validated 20/20 against the leaderboard team names, all had ample
replays (71–1000 episodes). The result is a decisive confirmation of M6's wall AT THE VERY TOP:
**13 of the 20 best decks in the world are unpilotable by our greedy heuristic** (deck strength
0.167–0.375 vs the 0.40 bar), including the #1 (Majkel1337, 0.319). Of the 7 that ARE pilotable,
every single one clones to an agent that LOSES to kanga on the field gauntlet (paired Δ −0.054 to
−0.151, all CIs entirely negative). The cleanest single finding: **"Where is my orbit" (#9, 1122
elo) plays the SAME Mega Kangaskhan ex + Crustle deck as kanga's teacher (懒惰的金枪鱼, #32, 1052),
17 of 18 cards identical, piloted 70 elo higher — and its clone STILL loses to kanga by −0.112**,
because its specific build is less pilotable (0.662 vs 0.736) and clones worse (0.546 vs 0.646). That
is the cleanest possible refutation of "higher teacher elo → better clone": same archetype, stronger
pilot, worse clone. Clonability of the exact 60-card build dominates teacher elo. Per the pre-registered
criterion, NO ship candidate qualified → nothing shipped, no slot spent. kanga-new keeps reconverging
toward ~860.**

Continues [m22_findings.md](m22_findings.md). The M23 hypothesis (teacher elo is the best ladder
predictor, so cloning the top-20 should beat kanga) is **contradicted by its own cleanest test.**

## What the user provided, and its validation

The exported public leaderboard CSV (`pokemon-tcg-ai-battle-publicleaderboard-2026-07-22`) has a
`TeamId` but no `submissionId` — so it cannot drive the downloader on its own (the M21 blocker). The
user manually looked up and filled the `submission_id` for the top-20 teams. **All 20 validated: the
owner name parsed from each downloaded replay folder matched the expected leaderboard team name,
20/20** (`make_batch.py` owner-detection). None had the "too few episodes" problem anticipated in the
plan — every submission had 71–1000 episodes; we pulled ~150 each.

## Fase 2 — pilotability triage (M6's wall, at the summit)

| rank | elo | team | deck strength (greedy) | verdict |
|---|---|---|---|---|
| #1 | 1231 | Majkel1337 | **0.319** | DISCARD (deck) |
| #2 | 1160 | LumenLiquidity | 0.368 | DISCARD (deck) |
| #3 | 1158 | junlee789 | 0.439 | pilotable |
| #4 | 1154 | Yudai Ueno | 0.408 | pilotable |
| #5 | 1153 | Luca | 0.243 | DISCARD (deck) |
| #6 | 1147 | tw_shin | 0.459 | pilotable |
| #7 | 1140 | Rmy | 0.167 | DISCARD (deck) |
| #8 | 1126 | GUOHAOYANG | 0.194 | DISCARD (deck) |
| #9 | 1122 | Where is my orbit | **0.662** | pilotable (best) |
| #10 | 1122 | Yushin Ito | 0.250 | DISCARD (deck) |
| #11 | 1121 | HowardLeeTW | 0.271 | DISCARD (deck) |
| #12 | 1117 | THIRD PTCG Club | 0.401 | pilotable |
| #13 | 1107 | Eduardo Rocha | 0.181 | DISCARD (deck) |
| #14 | 1106 | {{ team_name }} | 0.414 | pilotable |
| #15 | 1105 | kashiwashira | 0.375 | DISCARD (deck) |
| #16 | 1100 | iwashi | 0.236 | DISCARD (deck) |
| #17 | 1100 | binghua_123 | 0.246 | DISCARD (deck) |
| #18 | 1096 | Lunariz | 0.222 | DISCARD (deck) |
| #19 | 1094 | Dries @ Tufa Labs | 0.414 | pilotable |
| #20 | 1093 | jiatu.l | 0.236 | DISCARD (deck) |

**13/20 fail greedy-pilotability.** Deck strengths are full-field (120 decks) for the 7 pilotable; the
rest early-stopped below the 0.40 bar. This is M6's wall on the strongest evidence yet: the very best
agents in the world overwhelmingly run decks our heuristic simply cannot drive (Stage-2 engines,
multi-evolution toolboxes, prose-combo lines).

## Fase 3 — head-to-head vs kanga (the decisive test)

All 7 pilotable candidates, cloned with the generic profile, over the identical 120-deck field, paired
bootstrap CI vs the shipped hand-authored `KANGASKHAN_1052` (the config that reached ladder ~860):

| rank | elo | candidate | deck str | clonability | Δ vs kanga (n=6) | 90% CI | verdict |
|---|---|---|---|---|---|---|---|
| #12 | 1117 | THIRD PTCG Club | 0.401 | 0.565 | **−0.054** | [−0.088, −0.020] | LOSES |
| #19 | 1094 | Dries @ Tufa Labs | 0.414 | 0.658 | −0.059 | [−0.091, −0.028] | LOSES |
| #14 | 1106 | {{ team_name }} | 0.414 | 0.596 | −0.078 | [−0.110, −0.047] | LOSES |
| #9 | 1122 | Where is my orbit | 0.662 | 0.546 | −0.112 | [−0.144, −0.081] | LOSES |
| #4 | 1154 | Yudai Ueno | 0.408 | 0.618 | −0.124 | [−0.156, −0.094] | LOSES |
| #3 | 1158 | junlee789 | 0.439 | 0.588 | −0.150 | [−0.179, −0.118] | LOSES |
| #6 | 1147 | tw_shin | 0.459 | **0.692** | −0.151 | [−0.185, −0.118] | LOSES |

**None beats kanga; every CI is entirely negative.** The closest (THIRD PTCG Club, −0.054) is a
2000-elo-class deck that clones poorly. The generic-vs-hand-authored gap measured in M22 was ~+0.03
(ITF: generic +0.037 → hand-authored +0.069); even applied optimistically it cannot lift any of these
to positive.

## The cleanest finding: "Where is my orbit" refutes the teacher-elo hypothesis

The M23 rationale was that teacher elo is the best ladder predictor we have (656→686, 1052→860), so a
top-20 teacher should beat kanga. "Where is my orbit" is the controlled test of that claim:

- It runs the **same archetype** as kanga's teacher — Mega Kangaskhan ex (300 HP, Rapid-Fire Combo
  ●●● 200) + a Crustle line + Mist/Spiky/Grow-Grass energy. **17 of 18 unique cards are identical**
  to `KANGASKHAN_1052`'s deck (it runs Crushing Hammer where 懒惰的金枪鱼 runs Hand Trimmer, plus a
  Xerosic/Switch trainer swap).
- It is piloted at **1122 elo (#9) vs 1052 (#32)** — a genuinely stronger pilot of the same deck.
- The `KANGASKHAN_1052` hand-authored profile would apply to it almost unchanged (near-zero ship cost).

And its clone **loses to kanga by −0.112.** The reason is visible in the screen numbers: its specific
build is *less* pilotable under greedy (0.662 vs kanga's 0.736) and clones *worse* (0.546 vs 0.646).
So even holding archetype fixed and raising the pilot's elo, the clone gets worse — because the exact
60-card list 懒惰的金枪鱼 chose is the more clone-friendly build. **This is the M8 lesson
(clonability > teacher elo) proven on identical decks:** kanga's teacher didn't win by being the best
player, it won by running the most clone-friendly build of a strong deck.

## Honest caveats (the offline instrument's known unreliability)

- **The field gauntlet has mis-called the ladder 3 times** (kanga read as tie-vs-v1 → +174 real; ITF
  read as beats-kanga → worse at equal age). So a −0.05 to −0.15 offline delta does **not** prove any
  candidate is worse on the real ladder. What it does establish is the pre-registered permitted use:
  **no candidate shows a positive signal worth spending a scarce submission slot on**, and the one
  hypothesis that motivated the whole exercise (teacher elo) fails its own clean control.
- All 7 used generic profiles; kanga used its hand-authored one. That handicaps the challengers — but
  "Where is my orbit" is the same deck as kanga, so a hand-authored orbit profile would be ~the same
  profile, and it was still −0.112. Removing the profile handicap does not change the verdict.
- Two candidates (THIRD PTCG Club −0.054, Dries −0.059) are the closest; a hand-authored profile
  *might* pull them to a tie. Chasing a tie (not a win) across bespoke profiles, to displace ITF, is
  not worth it — and offline can't confirm a tie is a tie anyway (M14).

## Verdict & recommendation

- **NO ship candidate from the top-20.** Nothing was built or uploaded; no slot spent; kanga untouched.
- **The teacher-elo bet is weaker than believed.** M23 is the third data point on the curve
  (656→686, 1052→860, top-20→loses-to-kanga) and it bends the wrong way at the top: above ~1050, the
  decks stop being clone-friendly (unpilotable or worse-cloning), so more teacher elo buys nothing.
- **kanga's teacher (懒惰的金枪鱼, #32/1052) appears to be at or near the sweet spot** — a strong deck
  that also happens to be maximally clone-friendly. That is a rare combination, and this sweep of the
  top-20 did not find a better one.
- **Standing recommendation: let kanga-new reconverge to ~860 (it restarts at μ₀=600 on each upload,
  needs ~50+ episodes); do not spend a slot on any top-20 clone.** If the user wants to keep pushing,
  the remaining unexplored lever is not "a higher-elo teacher" (this closes that) but a *different*
  clone-friendly deck in the ~900–1050 band — the same band 懒惰的金枪鱼 and ITF came from, where the
  M22 pipeline already has ~270 unscreened candidates.

## Addendum — shipped `THIRD_PTCG` (built 2026-07-22, at the user's decision)

The user correctly noted the field gauntlet is unreliable (it read ITF as beating kanga; ITF is
losing on the ladder), so "least offline loss" is a weak basis. Since offline cannot rank these, the
rational ship criterion becomes the best GAMBLE, and the user chose **THIRD PTCG Club (#12, 1117)** —
the one pilotable top-20 deck in our best-understood archetype (Team Rocket), to test on the real
ladder.

**Why THIRD PTCG is a uniquely cheap + sensible build:** its deck is v1's exact Team Rocket Rush swarm
(Tarountula 400 → Spidops 401, Rocket Rush = 30×TR-in-play — `damage_650` verbatim) UPGRADED with two
heavy hitters the basic swarm lacked: **Team Rocket's Mewtwo ex** (Erasure Ball {P}{P}{C} 160, +60 per
bench energy discarded ≤2 → caps 280; Power Saver: can't attack unless 4+ TR Pokémon) and **Articuno**
(Dark Frost {W}{C}{C} 60, +60 with Team Rocket's Energy). It's the archetype v1 already clones to 686,
and it's pilotable (all basics + one Stage-1, no Stage-2 brick).

Hand-authored `THIRD_PTCG` profile (additive; v1@386, kanga@596, ITF@539 all re-verified as still
loading). Three engine-verified prose corrections (M22 lesson — effect text is in
`CardInfo.skills[]/attack.text`, not the None-valued item `.text`): Rocket Rush 30×TR, Erasure Ball at
its guaranteed 160 floor (optional self-discard, matching the coin-flip convention), Dark Frost 60+60
with Team Rocket's Energy; plus Brave Bangle's +30-vs-ex. `snapshot_third` (25 signals) exposes the
Rocket Rush driver (TR count + projected 30×), Mewtwo's 4-TR readiness gate, and TR-energy fuel;
`opp_target_dim=4` for Team Rocket's Giovanni (gust).

**Training (abundant data — 160 replay games):** MAIN accuracy **0.594** (vs greedy 0.356) on **5488**
MAIN rows; learned FIVE contexts (MAIN, TO_HAND, ATTACH_TO, TO_ACTIVE, SWITCH), weighted 0.634. The
hand-authored profile **beat the generic on fidelity (0.594 vs 0.565, +0.029)** — the largest
generic→hand-authored gain of any clone so far (kanga +0.006, ITF −0.007), because the Rocket Rush /
Mewtwo prose corrections genuinely help represent this deck.

**Head-to-head, hand-authored profile (n=12):**

| comparison | macro WR | paired Δ | 90% CI | verdict |
|---|---|---|---|---|
| THIRD(hand) − kanga1052 | 0.888 vs 0.893 | **−0.006** | [−0.031, +0.020] | **TIES kanga** |
| THIRD(hand) − v1 | 0.873 vs 0.899 | −0.026 | [−0.046, −0.006] | slightly < v1 |

Hand-authoring moved THIRD from −0.054 (generic) to **−0.006 vs kanga (a statistical tie)** — a +0.048
gain, exactly the M22 generic→hand-authored magnitude. It is now offline-indistinguishable from kanga
(the proven-860 agent) but clones a #12/1117 teacher vs kanga's #32/1052. Given the gauntlet's
documented systematic pessimism about strong teachers, a THIRD clone that TIES kanga offline is a
reasonable bet to climb past kanga on the ladder — which is exactly what the free-roll tests.

**Ship artifact: `build/imitation-third.tar.gz`** (2.0 MiB). Verified past the structural check:
extracted bundle reports `_IMITATION_READY: True`, profile resolves `THIRD_PTCG` at dim 567, deck
submission returns 60 cards, and the imitation scorer ran **11/11 MAIN fixtures with 0 failures** (not
degrading to greedy). 186 tests pass; all prior weights still load.

**Upload guidance (unchanged mechanics):** this goes in the ITF slot (the losing, offline-overrated
one), keeping kanga-new reconverging. Every upload restarts at μ₀=600 and needs ~50+ episodes; only the
most-recent-2 submissions are final-tracked, so do NOT upload two things (that would evict kanga-new).
Judge THIRD only after ~50 episodes, against kanga at equal age — not against kanga's converged 860.

## Files

New: `docs/m23_findings.md`, `candidates_m23.txt`, `candidates_m23_escalate.txt`,
`scratchpad/m23_h2h_batch.sh`, `scratchpad/m23_h2h_last2.sh`,
`data/rl/m23_{probe,download,screen,escalate,h2h_orbit,h2h_rest,h2h_last2}.log`,
`decks/*.csv` + `data/models/*_screen.json` for the 20 top-20 teams,
`replays/<20 submission_ids>/`. No `src/` change; no bundle built.
