# M21 findings — candidate discovery automated, and the first clone of a >1000-elo teacher (2026-07-21)

**Bottom line: the teacher-search bottleneck was never the screen's speed — it was that nobody
could enumerate candidates. Fixed: our own episode history already records every opponent's
`submissionId` AND their ladder rating, and joining that against the public leaderboard by
`teamId` matched 289/289 opponents, including teams ranked #11, #24, #32. Screened 8 spanning
ranks #11–#285. Four of eight top decks FAIL our greedy-pilotability gate (0.24–0.39) —
M6's wall, reconfirmed on fresh evidence. Of the four that pass, three fail clonability
outright (0.60–0.62). One survived: 懒惰的金枪鱼 (rank #32, ~1052 elo), deck strength 0.736,
clonability 0.646 — below the 0.75 bar, so the screen says DISCARD. We measured it anyway:
the clone TIES imitation-v1 on the field gauntlet (+0.010, CI spans 0). Then, per the user's
call, we took the designed path (option B) and hand-authored a real `KANGASKHAN_1052`
DeckProfile: fidelity moved only 0.646→0.652 (M15's predicted "small win"), the head-to-head
stayed a tie (+0.011) — but it made the clone SHIPPABLE, which the generic profile never was.
`build/imitation-kanga.tar.gz` is built and verified to actually run the imitation tier.
imitation-v1 (~686) remains champion; this is a free-roll candidate, not an offline win.**

Numbered M21 because M20 was taken by the parallel pattern-mining line (`m20_findings.md`).

## The unblock: candidate discovery from our own match history

`docs/m15_findings.md` closed the teacher search "pending a genuinely NEW candidate not yet
sourced". That framing assumed candidates were hard to find. They were not — they were already
in our own logs:

- Every episode returned by `kaggle_api.list_episodes` carries, for BOTH agents, `submissionId`,
  `teamId` and `updatedScore` (their live Kaggle rating). So our own match history is a ranked
  directory of opponents, with exactly the IDs `tools/download_competitors.py` consumes.
- New `tools/find_candidates.py` mines that (289 distinct opponents from 2 of our submissions;
  57 in the 700–900 band alone) and emits the ready-to-run download command.
- The public leaderboard CSV has `TeamId` but **no** `submissionId`, so it cannot drive the
  downloader on its own — but joining it to the mined opponents by `teamId` matched **289/289**,
  giving each reachable candidate its *current* rank/score. That is what "go from the top down"
  actually looks like here: not the whole leaderboard, but the reachable slice of it, which
  happens to include ranks #11, #24, #32, #47, #71.
- `tools/make_batch.py` closes the last manual step: the episode API has no player NAME, but the
  downloaded replays do (`info.Agents[].Name`), and the submission's owner is the agent appearing
  in ~every file of its own folder.

Pipeline, end to end: `find_candidates.py` → `download_competitors.py --limit` → `make_batch.py`
→ `quick_screen.py --batch`.

## Screen speedups — and one methodological correction

`quick_screen.py` used to take ~1 h/candidate. Three changes, no threshold moved:
1. **Field cache** (`field_gauntlet.extract_field`, keyed by a fingerprint of the replay folders):
   15.6 s → 0.01 s. It had been re-parsing ~355 replay JSONs on every single run.
2. **Progressive deck-strength gate**: play the field in shuffled chunks, stop when the 95% CI over
   decks clears or misses 0.40. Clear cases resolve on 12/120 decks (~3 s); only ambiguous ones pay
   the full field. Vibechu at 0.388 correctly refused to early-stop.
3. **`--batch`**: many candidates in one process, amortizing SDK/card-DB/field load.
4. **`--limit` on the downloader** (incremental, so a survivor is topped up later, never re-fetched):
   top submissions have 600–900 episodes; a screen needs ~100–150.

**Correction, important:** a 4th change — `--screen-games` subsampling for the clonability half —
was justified on the claim that "less training data can only LOWER accuracy, so a subsampled pass
is safe". **That reasoning is wrong and the data proved it**: 懒惰的金枪鱼 scored **0.686 on 60
games and 0.646 on all 461**. Subsampling shrinks the *validation* set too, making it narrower and
easier, so the estimate is **optimistic, not conservative**. The BORDERLINE escalation band is what
saved us; a candidate landing at 0.76 on a subsample would have been called PROMISING in error. The
subsample remains useful for triage but must never be treated as one-sided.

Separately, the notorious "1.7 h featurization" (6067 s) was **not** inherent: the identical dataset
featurized in **17 s** on a quiet machine. It was contention, not cost.

## The 8-candidate screen (ranks #11–#285)

| candidate | rank | now | deck strength | clonability | verdict |
|---|---|---|---|---|---|
| jiatu.l | 11 | 1103 | 0.278 | — | DISCARD (deck) |
| **懒惰的金枪鱼** | **32** | **1052** | **0.736** | **0.646** | below bar; measured anyway |
| Kazama Yusuke | 47 | 1031 | 0.278 | — | DISCARD (deck) |
| vvhan | 77 | 1003 | 0.708 | 0.623 | DISCARD (clone) |
| PP.TAKEHIRO_KAWADA | 83 | 1000 | 0.371 | — | DISCARD (deck) |
| shu | 109 | 983 | 0.806 | 0.611 | DISCARD (clone) |
| Team Rocket | 166 | 956 | 0.394 | — | DISCARD (deck) |
| sam_the_rice_cake | 285 | 912 | 0.583 | 0.605 | DISCARD (clone) |

**Half the world's best decks are unpilotable by our heuristics** (0.24–0.39 vs the 0.40 bar) —
M6's wall on four new data points. And of those we *can* pilot, clonability lands at 0.60–0.65:
the clones do learn a lot (shu: greedy 0.358 → BC 0.611; 懒惰的金枪鱼 MAIN: greedy 0.294 → BC
0.646) but never reach the 0.75 bar. **The better the player, the harder to clone** — their edge
lives precisely in the decisions our ~600-dim featurizer cannot represent.

## Testing the teacher-elo hypothesis directly

Open question from the user: even at low fidelity, shouldn't cloning a 1052 teacher beat v1
(which clones a 661 teacher at higher fidelity)? Rather than argue from priors (M8's 800-elo clone
and M15's kenN2439 both said no), we measured it — new `scratchpad/head_to_head.py` runs both over
the identical 120-deck field with a paired bootstrap CI:

| clone | field macro WR | paired Δ vs v1 |
|---|---|---|
| generic profile | 0.882 (v1 0.872) | +0.010, CI [−0.021, +0.040] |
| hand-authored profile | 0.894 (v1 0.883) | +0.011, CI [−0.018, +0.042] |

**Both TIE v1.** The honest reading is two-sided: teacher elo does **not** transfer proportionally
(no 800–900 agent appeared), but a **0.65-fidelity clone of a 1052 player is worth about as much as
a high-fidelity clone of a 661 player** — the teacher's quality really does offset the fidelity
loss. Caveat that limits both readings: the field gauntlet is greedy-piloted and **saturates near
0.90** (M11); at 0.88–0.89 it cannot resolve a subtle gap, and M14 already falsified our finer
instrument for exactly this. Offline cannot settle it.

## Option B — the hand-authored `KANGASKHAN_1052` profile

The generic profile is explicitly *"never for a shipped submission"* (`deck_profiles.py:915`) and
`get_profile` raises on unknown names, so a generic-profile bundle would have silently fallen back
to greedy. The user chose the designed path over an entrypoint hack.

Deck (18 unique cards): **Mega Kangaskhan ex** (300 HP Basic, `megaEx` → **3 prizes**, Rapid-Fire
Combo ●●● 200 + "flip until tails, +50/heads") and a **Crustle** line (150 HP, Ability prevents ALL
damage from the opponent's *ex* attackers, Superb Scissors {G}●● 120). Energy is 12 special + 1
basic {G}: Mist/Spiky give {C} (so Kangaskhan's ●●● is colour-blind) while only 5 cards give {G}.

What the profile encodes that the generic one could not:
- `damage_kangaskhan`: Rapid-Fire Combo at its **guaranteed 200**, not its expected 250 — matching
  `damage_940`'s treatment of coin flips, so lethal/overkill slots never claim a KO the attack
  cannot guarantee.
- `wants_kangaskhan`: the real {G} scarcity — Crustle still "wants" energy at 3 attached if none of
  them is {G}, because a Crustle with three colourless simply cannot attack.
- `snapshot_kangaskhan` (28 signals): **whether the opposing Active is an `ex`** — the deck's entire
  plan, since Crustle blanks ex attackers — plus Hero's Cape (+100 HP → a 400 HP Kangaskhan),
  Battle Cage, and {G} sources in hand.

Result: MAIN 0.646 → **0.652**, weighted 0.664 → 0.667. Exactly M15's predicted "small win". The
profile's value was **shippability**, not fidelity.

## Ship artifact

`build/imitation-kanga.tar.gz` (2.0 MiB / 197.7 limit), passes structural + smoke validation, and —
verified explicitly, because `_build_imitation` swallows every exception and silently degrades to
greedy — the extracted bundle reports `_IMITATION_READY: True` with `KANGASKHAN_1052` resolving at
`feature_dim` 596. It genuinely runs the clone.

**Disposition:** a ladder free-roll candidate, not an offline win. It qualifies under M12's rule
(free-rolls are for bets that are *genuinely different* from v1, not another fidelity tweak): a
different deck, a different teacher, +391 elo. Kaggle keeps the best historical score, so the cost
is one slot's ladder time. Not uploaded — the user's call.

## Files

New: `tools/find_candidates.py`, `tools/make_batch.py`, `scratchpad/head_to_head.py`,
`docs/m21_findings.md`, `data/candidates_joined.json`, `decks/懒惰的金枪鱼.csv`,
`data/models/bc_kangaskhan_1052.json`, `build/imitation-kanga.tar.gz`.
Changed: `src/ptcg_ai/imitation/deck_profiles.py` (+`KANGASKHAN_1052`, additive — existing profiles
and weights untouched), `tools/download_replays.py` + `download_competitors.py` (`--limit`),
`scratchpad/quick_screen.py` (cache/progressive/batch/subsample + BORDERLINE escalation),
`scratchpad/field_gauntlet.py` (field cache).
