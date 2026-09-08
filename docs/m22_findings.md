# M22 findings — second screening round, and the first candidate to BEAT v1 offline (2026-07-22)

**Bottom line: ran the M21 pipeline on 12 fresh candidates spanning ranks #71–#866. Four fail
greedy-pilotability (M6's wall again). Eight pass, and every one of them fails the 0.75
clonability bar (0.530–0.667) — so the screen's own verdict is DISCARD across the board. Per
the user's standing instruction not to treat 0.75 as eliminatory, the three most interesting
were taken to a direct head-to-head against imitation-v1 anyway. Result: `ITF_Esys_Kasu`
(rank #826, seen at 863.8, deck strength 0.731, clonability 0.530 — the LOWEST fidelity of the
round) **BEATS imitation-v1 on the 120-deck field gauntlet: +0.043 90%CI [+0.017,+0.068] at
n=6, replicated at n=12 as +0.036 [+0.015,+0.059], 38 decks better vs 24 worse.** This is the
first candidate in the project's history whose paired CI against v1 is entirely positive —
M21's Kangaskhan tied (+0.010, CI spanning 0); M8's v2 and M9's v3 lost. It is NOT shippable
as-is (generic profile), and the user's decision on a hand-authored profile is pending.**

Continues [m21_findings.md](m21_findings.md); numbered M22 (M20 = pattern mining, M21 = the
discovery pipeline).

## Round composition

`data/candidates_joined.json` had 289 mined opponents, only ~13 previously screened. Selected 12
**spread across the leaderboard rather than top-down**, because M21 showed top decks are
disproportionately unpilotable — and prioritising those whose `seen` score (their live rating when
we actually faced that submission) was high, so we screen the agent we met, not a stale one.

`replays/` had been emptied after M21, so all 12 were re-downloaded (`--limit 150`; several
submissions have fewer episodes than that in total — noted per candidate below).

## Results — all numbers on the FULL 120-deck field and FULL replay data

The batch triage ran first (progressive early-stop + `--screen-games 60`); every candidate that
passed deck strength was then re-run with `--full-field --screen-games 0`. Both corrections
mattered: the 12-deck early stop is a *threshold test vs 0.40*, not a measurement (🐱KittenLeague🐱
read 0.542 on 12 decks and **0.451** on 120; `f` read 0.694 → **0.764**), and the subsample
overstates clonability (M21's correction — see the note at the end).

| rank | score | seen | player | deck strength | clonability | games | screen verdict |
|---|---|---|---|---|---|---|---|
| 171 | 952.6 | 722.4 | f | **0.764** | 0.557 | 151 | DISCARD (clone) |
| 826 | 823.0 | 863.8 | **ITF_Esys_Kasu** | **0.731** | 0.530 | 64 | DISCARD (clone) |
| 433 | 872.3 | 759.0 | Fakble 6 | 0.676 | **0.667** | 79 | DISCARD (clone) |
| 836 | 821.9 | 768.9 | DiuDiu | 0.662 | 0.643 | 151 | DISCARD (clone) |
| 71 | 1005.3 | 612.4 | Banjo | 0.640 | 0.530 | 52 | DISCARD (clone) |
| 765 | 829.0 | 844.1 | xsong2020 | 0.569 | 0.644 | 151 | DISCARD (clone) |
| 866 | 819.4 | 827.7 | SYC | 0.558 | 0.663 | 151 | DISCARD (clone) |
| 102 | 987.8 | 752.8 | 🐱KittenLeague🐱 | 0.451 | 0.580 | 102 | DISCARD (clone) |
| 559 | 855.0 | 751.2 | mikelou1 | 0.310 | — | 52 | DISCARD (deck) |
| 664 | 841.7 | 726.0 | tenajima | 0.276 | — | 151 | DISCARD (deck) |
| 201 | 941.1 | 722.1 | Orion | 0.264 | — | 133 | DISCARD (deck) |
| 328 | 895.4 | 746.8 | KakuTakagawa | 0.236 | — | 111 | DISCARD (deck) |

Reference from M21: 懒惰的金枪鱼 (rank #32, 1052) = strength 0.736, clonability 0.646.

**M6's wall, four more data points:** 4/12 decks score 0.236–0.310 under our greedy pilot. Two of
them (Orion #201, KakuTakagawa #328) are strong ladder agents whose decks we simply cannot drive.

**Clonability never reached the bar** — 0.530–0.667 across eight decks, consistent with M21's
0.605–0.646. Eight more independent confirmations that our ~600-dim featurizer cannot represent a
strong pilot's decisions.

## Head-to-head vs imitation-v1 (the measurement that mattered)

`scratchpad/head_to_head.py`, generic profile (as M21 used it), same 120-deck greedy-piloted
field for both sides, paired per-deck bootstrap CI:

| candidate | strength | clonability | macro WR | v1 macro | paired Δ vs v1 | 90% CI | verdict |
|---|---|---|---|---|---|---|---|
| **ITF_Esys_Kasu** n=6 | 0.731 | 0.530 | 0.933 | 0.890 | **+0.043** | [+0.017, +0.068] | **BEATS v1** |
| **ITF_Esys_Kasu** n=12 | " | " | 0.928 | 0.892 | **+0.036** | [+0.015, +0.059] | **BEATS v1** (replicated) |
| Fakble 6 | 0.676 | 0.667 | 0.837 | 0.865 | −0.028 | [−0.063, +0.006] | ties v1 |
| f | 0.764 | 0.557 | 0.829 | 0.886 | −0.057 | [−0.099, −0.017] | LOSES to v1 |

The n=12 run is 1440 games per side and reproduces n=6 within noise.

### ITF_Esys_Kasu vs the current best clone (M21's Kangaskhan)

The decisive comparison for the user's actual decision — which clone deserves the free-roll
slot. `head_to_head.py` was generalised (`--baseline`) so any two clones can be compared on the
identical field in one process. Kangaskhan ran under its **hand-authored** `KANGASKHAN_1052`
profile (its shippable form), ITF under the generic screen profile; 0 SafePolicy interventions
on both sides:

| | macro WR | paired Δ | 90% CI | decks better/worse |
|---|---|---|---|---|
| **ITF_Esys_Kasu − kanga1052** (n=12) | 0.918 vs 0.881 | **+0.037** | [+0.019, +0.055] | 48 / 21 of 120 |

**ITF beats the current best clone with a CI entirely above zero.** The three measurements are
mutually consistent: ITF−v1 ≈ +0.036, kanga−v1 ≈ +0.011 (M21), so ITF−kanga ≈ +0.025 expected
vs +0.037 measured — coherent within the instrument's noise, which is a useful internal check
that none of the three runs is an artifact.

### Why this deck clones unusually well at low fidelity

ITF's decklist (18 unique cards — same compactness as Kangaskhan's): **Mega Starmie ex**
(330 HP, `megaEx` → 3 prizes; Jetting Blow 120 for 1 energy, Nebula Beam 210 for 3) over a
Staryu line, 4× Cinderace, 4× Crushing Hammer (energy denial), 1× Boss's Orders, 4× Ignition
Energy, 9× basic {W}.

Crucially, **every attack has honest structured damage — no prose scaling**. That matters:
the generic profile's `generic_damage` is *correct* for this deck, whereas for Team Rocket it
would badly misread Rocket Rush. So the 0.530 clonability number understates how well the
generic clone actually plays here, which is a plausible mechanism for the head-to-head result
and further evidence that MAIN-accuracy is the wrong proxy. It also means a hand-authored
`damage_fn` would be near-trivial (structured + weakness/resistance, like `damage_800`); the
real work would be `snapshot_fn` signals and `opp_target_dim=4` for Boss's Orders.

## What this does and does not say about the "deck strength beats fidelity" hypothesis

The user's working hypothesis after M21 was that deck/teacher strength may matter more than clone
fidelity, and that the 0.75 bar discards valuable candidates. This round gives it **strong support
and one clean counterexample, which together are more informative than either alone**:

- **Support:** the winner has the round's *lowest* clonability (0.530). Under the 0.75 bar — and
  even under a relaxed 0.60 bar — it would have been thrown away. The bar is demonstrably not
  measuring what we care about.
- **Counterexample:** `f` has the round's *highest* deck strength (0.764, above the M21
  candidate's 0.736) and still **loses** to v1 (CI entirely negative). So raw greedy-pilotability
  of the deck is not sufficient either.
- **Neither screen number predicts the head-to-head.** Ranking the three duelled candidates by
  strength gives f > ITF > Fakble; by clonability gives Fakble > f > ITF. The actual ordering is
  ITF ≫ Fakble > f. **The only reliable instrument is the head-to-head itself**, which is cheap
  (~10–20 min) now that the field is cached. Recommendation: demote both screen gates to triage
  (deck strength ≥0.40 to prove pilotability at all, clonability reported but not eliminatory) and
  spend the time on `head_to_head.py` for anything that clears triage.

## Honest caveats on the ITF_Esys_Kasu result

1. **Not shippable as measured.** It used the auto-built generic profile, which
   `deck_profiles.py` explicitly marks as never-for-submission, and `get_profile` raises on
   unknown names — a generic-profile bundle would silently fall back to greedy. A hand-authored
   profile is required. **Not started, per the user's instruction.**
2. **Smallest dataset of the round (64 replay games).** Its 0.530 clonability rests on the least
   data; more episodes exist to top up incrementally (`download_competitors.py` is incremental).
3. **The field gauntlet saturates near 0.90** (M11/M21) and the candidate sits at 0.93. The paired
   per-deck delta is the correct read and it is clean and replicated — but this is the *same class*
   of offline instrument M14 showed cannot resolve subtle gaps. What can be said: this gate
   correctly predicted the real ladder outcome for both v2 (worse) and v1 (better), and no previous
   candidate ever cleared it.
4. Its team currently sits at rank #826 / 823 elo, but `seen` was **863.8** — the highest observed
   live rating in the round. Current team rank may reflect a different, newer submission.

## Pipeline fix applied

M21's most important methodological correction — that `--screen-games` subsampling **overstates**
clonability (0.686 @60 vs 0.646 @461) because it shrinks the validation set too — had been written
into `m21_findings.md` but **never into the code**. `quick_screen.py` still printed
`[subsampled -> lower bound]` and still carried the comment *"Safe even on a subsample: more data
can only raise this"* — the exact reasoning M21 falsified. Fixed (labels/docstring only, no
threshold moved): the subsample is now labelled OPTIMISTIC, and a subsampled pass returns
`PROMISING_UNCONFIRMED` demanding a `--screen-games 0` re-run instead of `PROMISING`.

## The hand-authored `ITF_ESYS_KASU` profile + ship artifact (built 2026-07-22)

Built at the user's request after they accepted the risk. Additive change to
`deck_profiles.py` (dim 539); every existing profile and weight file still loads unchanged
(`bc_650_v1` 386, `bc_kangaskhan_1052` 596 both re-verified).

**A card-data reading error was found and corrected in the process.** Earlier passes over this
deck read `CardInfo.text` for items/tools/special energy and got `None`, concluding there was no
effect text. That is the wrong attribute — the real text lives in `CardInfo.skills[].text`, and it
was populated the whole time. Reading it correctly changed two modelling decisions:

- **Nebula Beam ({C}{C}{C}, 210)** — its text says *"This attack's damage isn't affected by
  Weakness or Resistance, or by any effects on your opponent's Active Pokémon."* So `damage_itf`
  returns a flat 210 and **skips** the W/R adjustment every other attack in the deck receives.
  Modelling it as ordinary structured damage (the generic profile's behaviour) is simply wrong
  against weak/resistant defenders. This is the same class of engine-exact correction as M17's T3.
- **Ignition Energy** — provides {C}, or **{C}{C}{C} when attached to an Evolution Pokémon**
  (both Mega Starmie ex and Cinderace qualify), and discards itself at end of turn. A single
  attach can therefore outright pay Nebula Beam. `wants_itf` deliberately delegates to
  `gs.wants_energy`, because the observation's `energies` tuple already reflects the engine's
  resolved value — hand-rolling a cost rule here would double-count it.

`snapshot_itf` (25 signals) adds what the generic snapshot could not see: which attacker holds the
Active spot, Mega Starmie on board, Hero's Cape (+100 HP), and counts of {W}/Ignition/Boss's
Orders/Crushing Hammer in hand. `opp_target_dim=4` for Boss's Orders.

**Fidelity: 0.530 (generic) → 0.523 (hand-authored)** on the full 63-game dataset — no gain, the
M21 pattern repeating. The profile's value is shippability, not fidelity.

**Critically, the head-to-head result transfers to the shippable artifact** — this had to be
re-measured, because every number above was produced by the *generic* profile while the bundle
ships the hand-authored one:

| comparison (n=12, hand-authored profile) | macro WR | paired Δ | 90% CI | decks better/worse |
|---|---|---|---|---|
| **ITF(hand) − v1** | 0.933 vs 0.899 | **+0.034** | [+0.017, +0.049] | 42 / 20 |
| **ITF(hand) − kanga1052** | 0.948 vs 0.879 | **+0.069** | [+0.045, +0.092] | 48 / 16 |

Both CIs entirely positive; the margin over Kangaskhan is *larger* with the hand-authored profile
(+0.069 vs the generic's +0.037).

**Ship artifact: `build/imitation-itf.tar.gz`** (2.0 MiB / 197.7 limit). Verified beyond the
structural check — because `_build_imitation` swallows every exception and silently degrades to
greedy, the extracted bundle was exercised directly: `_IMITATION_READY: True`, profile resolves to
`ITF_ESYS_KASU` at dim 539, and on 11 real MAIN fixtures the imitation scorer ran **11/11 with 0
failures** (`_bc_used` 0→11, `_bc_failures` 0). It genuinely runs the clone. 186 tests pass.

## Recommendation

**Clone ITF_Esys_Kasu, and give it the free-roll slot instead of the Kangaskhan bundle.** It
beats v1 (+0.036, replicated) and beats the Kangaskhan clone (+0.037) on the gate that
historically called both v2's regression and v1's strength correctly, and it is the first
candidate ever to clear it. `build/imitation-kanga.tar.gz` is now dominated by a better
candidate, so spending a slot on it would be spending it on the weaker of the two.

Status: steps 1–3 are **done** (see the section above). Replay top-up returned 0 new episodes —
63 is all that submission has, so the small-sample caveat stands and cannot currently be retired.
`build/imitation-itf.tar.gz` is built and verified; the upload is the user's call.

### Rating mechanics the user should know before uploading (from `competition_analysis.md`)

The user observed that re-uploading an agent "tends to play significantly worse". That is a
rating artifact, not a policy regression, and it has two documented causes:

- **Every upload is a new `submission_id` starting at μ₀ = 600** with wide σ, regardless of the
  weights being identical. It does not inherit the previous rating and must re-converge; the
  handoff already notes ~50+ episodes before a score is meaningful. Early readings also come from
  matchmaking against the *low-rated* pool, not the pool the old agent had climbed into.
- **Only the most recent 2 submissions are actively tracked for final evaluation.** Uploading ITF
  can push an older agent out of that window even while its score still shows on the leaderboard.
  Worth checking the active-submission count before uploading.

Practical consequence: do **not** judge ITF against kanga's stable ~850 until ITF has accumulated
~50+ episodes.

## Open / next

- ~276 of 289 mined candidates remain unscreened. The three duelled here cost ~20 min each; the
  field-cached pipeline makes a much wider sweep affordable, and this round showed the screen
  gates would have discarded the one candidate that actually won.

## Files

New: `docs/m22_findings.md`, `candidates_m22.txt`, `candidates_m22_escalate.txt`,
`data/rl/m22_{download,screen,escalate,h2h_f,h2h_fakble6,h2h_itf,h2h_itf_n12}.log`,
`decks/{f,itfesyskasu,fakble6,diudiu,banjo,xsong2020,syc,kittenleague,orion,kakutakagawa,mikelou1,tenajima}.csv`,
`data/models/*_screen.json` for the same.
Changed: `scratchpad/quick_screen.py` (subsample-optimism labelling only).
