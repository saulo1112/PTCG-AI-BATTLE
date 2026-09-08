# M32 — Loss diagnosis of the champion `imitation-mlp` on the real ladder (2026-07-28)

First look at the champion's **own** losses. `imitation-mlp` (MLP clone of Yushin Ito #1,
Stage-2 Alakazam combo, profile `ALAKAZAM` dim 658, weights `bc_alakazam_mlp.json`) has led the
ladder for two days (889 vs a concurrently-uploaded kanga at 845.5, M28.1) but we had never
downloaded its replays. Same instrument as M26 (Kanga) and the nsr postmortem.

**Data:** `replays/55011997`, 100 episodes (1 self-match excluded) → **53W-46L, WR 53.5%**.
**Tools (read-only, new):** `scratchpad/diagnose_mlp.py`, `scratchpad/diagnose_mlp_other.py`.

---

> **⚠️ SECTION 1 IS PARTLY SUPERSEDED BY SECTION 1-bis.** The per-archetype split below is
> correct as measured, but the *interpretation* ("three hard counters") does not survive the
> teacher comparison: Yushin **beats** two of the three. Read 1-bis before acting on 1.

## 1. Headline — the gap is a **matchup wall**, not a pilot defect

Three specific opponent decks account for **27 of the 46 losses (59%)** while the champion is
comfortably ahead of everything else:

| archetype | n | W | L | WR | loss reasons |
|---|---|---|---|---|---|
| **Marnie's Grimmsnarl ex** | 27 | 10 | 17 | **37%** | prize-race 15, bench-out 2 |
| **Dragapult ex** | 6 | 0 | 6 | **0%** | prize-race 4, bench-out 2 |
| **Mega Kangaskhan** | 4 | 0 | 4 | **0%** | **deck-out 4** |
| — *subtotal* | **37** | **10** | **27** | **27%** | |
| Alakazam (mirror) | 27 | 19 | 8 | 70% | prize-race 7, deck-out 1 |
| Mega Lucario | 9 | 8 | 1 | 89% | |
| Archaludon | 9 | 8 | 1 | 89% | |
| Cinderace | 3 | 1 | 2 | 33% | |
| other (long tail) | 11 | 6 | 5 | 55% | |
| — *everything else* | **62** | **43** | **19** | **69%** | |

**69% against the field vs 27% against three decks.** This split was invisible until now because
the archetype marker table inherited from `diagnose_v1`/`diagnose_kanga` had no marker for any of
them — all three were lumped into `other` (which read as a flat 35% and looked like diffuse
weakness). Markers `648 / 756 / 121 / 1191 / 104` are now added to `diagnose_mlp.py`.

### The nemesis deck (17 games, 5W-12L on its main list; 27 games, 10W-17L across variants)

```
10x Basic {D} Energy      4x Munkidori (112)          4x Marnie's Impidimp (646)
 4x Buddy-Buddy Poffin    4x Poké Pad                 4x Team Rocket's Petrel (1219)
 4x Lillie's Determination 4x Spikemuth Gym (1259)    3x Marnie's Morgrem (647)
 3x Marnie's Grimmsnarl ex (648)  3x Rare Candy       3x Night Stretcher
 2x Froslass (104)        2x Snorunt                  2x Boss's Orders
 1x Unfair Stamp (1080)   1x Pokégear  1x Tool Scrapper  1x Dawn
```

It is a **damage-counter-manipulation + hand-disruption** deck, and it is close to a designed
counter to our win condition: Powerful Hand *places damage counters* (20 × hand_size) and
**Munkidori moves damage counters away**; **Unfair Stamp / Petrel / Spikemuth Gym attack hand
size**, which is literally our damage dial; **Froslass** snipes the fragile Abra/Kadabra line off
the bench. Losses are overwhelmingly **prize-races (15/17)** — we are not being blown out, we are
being out-traded.

**Mega Kangaskhan (0-4) is our own former champion deck** (`decks/懒惰的金枪鱼.csv`, run by other
competitors) and it beats us **exclusively by deck-out (4/4)** — the Alakazam triple draw engine
mills itself while Kangaskhan grinds. Note 9 of 46 losses overall are deck-outs, a failure mode
Kanga never had.

---

## 1-bis. The discriminator — **does Yushin also lose these matchups?**

Section 1 leaves two incompatible readings: (a) *deck/archetype weakness* — the teacher loses them
too, so piloting cannot fix it; (b) *clone gap* — the teacher wins them and the clone doesn't. We
have his 1284 episodes on disk, so this is directly measurable. `scratchpad/diagnose_mlp_teacher_matchup.py`
finds his seat by **deck identity** (his exact 60-card list), not by player name.

**Teacher: 612W-451L = 57.6%** over 1063 games (221 true mirrors excluded). **Clone: 53.5%/99.**

| archetype | teacher W-L | WR | clone W-L | WR | delta |
|---|---|---|---|---|---|
| **Marnie's Grimmsnarl** | **319-284** | **53%** | 10-17 | 37% | **−16%** |
| other (long tail) | 89-26 | 77% | 6-5 | 55% | −23% |
| **Mega Kangaskhan** | **75-31** | **71%** | 0-4 | 0% | −71% |
| Team Rocket | 27-75 | 26% | 0-1 | 0% | −26% |
| Alakazam (mirror) | 52-8 | 87% | 19-8 | 70% | −16% |
| **Dragapult ex** | **16-12** | **57%** | 0-6 | 0% | −57% |
| Mega Lucario | 18-5 | 78% | 8-1 | 89% | +11% |
| Cinderace | 8-0 | 100% | 1-2 | 33% | −67% |
| Archaludon | 3-1 | 75% | 8-1 | 89% | +14% |

**Three corrections to section 1:**

1. **"Three hard counters" is WRONG.** The teacher goes **71% vs Mega Kangaskhan** and **57% vs
   Dragapult** — he beats both comfortably. Our 0-4 and 0-6 are small-sample noise and/or clone
   gap, **not** deck-level counters. Retract that framing. (The teacher's own Kangaskhan losses
   are only 23% deck-outs, vs 4/4 for us — the deck-out failure mode may be clone-specific, but
   n=4 cannot establish it.)
2. **Grimmsnarl IS a genuine deck-level wall — and it is the meta.** The teacher, the world #1,
   manages only **53% over 603 games** — barely even, and below his 57.6% overall. Piloting cannot
   turn that into 70%. Note the sample: **Grimmsnarl is 57% of the teacher's entire field** (603
   of 1063) versus 27% of ours — because he plays at ~1230 and we play at ~889. **The deck that
   holds the #1 to a coin flip dominates the top of the ladder, and we will face more of it the
   higher we climb.**
3. **The clone's deficit is broad, not Grimmsnarl-specific.** It trails the teacher in essentially
   every matchup (only Lucario/Archaludon are positive, at n=9 each). The honest summary is
   **53.5% vs 57.6% overall — a ~4-point uniform clone gap** — and the −16% on Grimmsnarl sits at
   ~1.7 binomial SE with n=27, i.e. suggestive, not established.

**The ceiling this implies:** a *perfect* clone of Yushin on this deck is worth **57.6%**. That is
the entire remaining headroom of the imitation line with this deck — about 4 points — and M31
already showed the last four levers to close it are exhausted.

**Arithmetic check on the Grimmsnarl number specifically:** the teacher's own matchup-specific dip
(57.6% overall → 52.9% vs Grimmsnarl, i.e. −4.7 points that exist for even the best pilot) plus the
clone's general −4-point gap predicts clone-vs-Grimmsnarl ≈ 49%. The **measured** value is **37%**
— a **~12-point residual** unexplained by either the deck's own difficulty or the clone's general
deficit. Section 1-ter identifies a concrete, verified behavioural mechanism for that residual.

---

## 1-ter. The residual is real and matchup-specific — verified two ways

Bootstrapped with `scratchpad/bootstrap_ph_turn.py` (kept in the temp scratchpad, not committed;
reproducible from the code in this section). Question: is the clone's slow Powerful-Hand trigger
against Grimmsnarl (section 3's aggregate "1 turn slower" finding, isolated to just these 27 games)
a real signal at n=27, or noise?

**Raw per-game first-use turn, clone vs Grimmsnarl (24 of 27 games used PH at all):**
`[3,3,4,4,4,4,4,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,13]` — median **6**, and notably **half the
sample (12/24) lands on exactly turn 6**, unlike the teacher's smooth decay (see below). One
outlier at turn 13.

**Teacher vs Grimmsnarl (591 of 603 games used PH):** median **4**, histogram
`{3:156, 4:157, 5:141, 6:100, 7:15, 8:11, 9:5, 10:3, 11:1, 12:2}` — a smooth tail, no turn-6 spike.

**Test 1 — bootstrap against noise.** Resampled 20,000 24-game samples (with replacement) from the
teacher's own 591-game Grimmsnarl distribution: **P(resampled median ≥ 6.0) = 0.1%**. If the clone
timed Powerful Hand exactly like the teacher, a turn-6 median at n=24 would almost never occur by
chance. This is not noise.

**Test 2 — control in a matchup the clone WINS (Mega Lucario, 89% WR).** Same instrument, same
teacher/clone comparison, different archetype: clone median **4** (n=9), teacher median **4.5**
(n=22), **P(resampled median ≥ clone's 4) = 93.3%** — completely unremarkable, indistinguishable
from teacher-level play. **The delay does NOT appear in a matchup the clone is winning.**

**Conclusion: this is a real, matchup-conditioned behavioural gap, not a restatement of the general
~4-point clone deficit.** It is specifically triggered by facing Grimmsnarl (plausibly the
Munkidori/hand-attrition pressure named in section 1), not a generic "the clone is a bit slower"
trait — the Lucario control rules that out. Remaining honest caveat: this does not control for
which *specific* 27 Grimmsnarl games/variants we happened to face (a composition effect distinct
from pure sampling noise, which the bootstrap does not detect); it is well short of a randomized
comparison. But it is now a targeted, falsifiable hypothesis rather than a hunch — the natural next
experiment is a Grimmsnarl/Munkidori-conditioned additive feature (contrast with M31's V2, which
enriched features *untargeted* at any specific matchup and lost the veto), retrained and re-measured
on this exact slice before spending a `head_to_head` or ladder slot.

---

## 1-quater. First fix attempt (`ALAKAZAM_MATCHUP`) — CLOSED NEGATIVE at the direct behavioural check

Built the narrowly-targeted feature named above: `ALAKAZAM_MATCHUP` (dim 658→684,
`deck_profiles.py`), two additive binary flags — `facing_disruption` (opponent board/discard
contains Munkidori 112 or the Impidimp/Morgrem/Grimmsnarl-ex line 646/647/648) and
`hand_attack_seen` (Unfair Stamp 1080 / Team Rocket's Petrel 1219 / Spikemuth Gym 1259 stadium).
Both are public information (own board + own discard + shared stadium). 203 tests pass.

**Offline proxy (M31's Colab/GPU pipeline, extended to slice TEST by archetype) — weakly
positive, two independent 10-seed runs:**

| run | per-seed mean TEST | ensemble TEST | ensemble vs Grimmsnarl | ensemble vs rest |
|---|---|---|---|---|
| baseline ALAKAZAM | 0.7689 ± 0.0040 | 0.7826 | 0.7498 | 0.7302 |
| MATCHUP run 1 | 0.7643 ± 0.0069 | 0.7890 | 0.7593 (+0.0095) | 0.7368 (+0.0066) |
| MATCHUP run 2 | 0.7693 ± 0.0053 | 0.7897 | 0.7601 (+0.0103) | 0.7385 (+0.0083) |

Per-seed means are statistically indistinguishable from baseline in both runs (combined SE ≈
0.007-0.009, |t| < 1) — a repeat of M31's own lesson to distrust an ensemble-only number. Still,
both independent runs lean the same direction, and the Grimmsnarl slice improves *more* than the
rest both times — a small but real-looking directional signal on the proxy.

**Decisive check — counterfactual re-score of the actual 27 real clone-vs-Grimmsnarl ladder games
(`scratchpad/counterfactual_ph_timing.py`), not a proxy.** Loaded the trained MLP ensemble directly
(no bundle needed — new `--emit-raw` on `colab_train_mlp.py`) and replayed every real MAIN decision
in those 27 games through it, asking whether it would fire Powerful Hand earlier than the shipped
V1. **Sanity check passed:** V1 re-scored through this harness reproduces the bootstrap script's
raw turns exactly (`[3,3,4,4,4,4,4,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,13]`, median 6.0) — the
harness is faithful.

**The candidate does NOT fire earlier — it fires slightly LATER (median 6.0 unchanged, mean 5.46 →
6.21).** Paired on the 24 games both models fired in: **20 unchanged, 4 WORSE (2–4 turns later),
0 improved.** New right-tail mass at turns 7/8/10 that V1 never reaches.

**Verdict: CLOSED NEGATIVE.** This is the offline proxy lying *again* (M11/M14/M31, now a 7th
instance) — but a sharper one: the proxy's small positive lean was in the *correct targeted slice*
and still didn't predict the real, targeted behaviour. Working theory for the mechanism: both flags
are "sticky" (stay 1.0 for the rest of the game once the archetype is identified, usually by turn
1-2), so they carry no per-turn *timing* signal — the model most likely picked up a spurious
correlation ("this flag on → generally wait a bit longer") from the training distribution instead
of the intended "this flag on → commit earlier before Munkidori strips the counters." **Do not
build a bundle or spend a `head_to_head`/ladder slot on this weights file.** No `src/` change is
load-bearing here except the (safe, additive, tested) `ALAKAZAM_MATCHUP` profile definition itself,
which stays in the codebase as a reusable base for a future attempt but is not used by anything
shipped. Reusable, verified-correct infra from this attempt: the sliced-Colab-eval extension, the
`--emit-raw` raw-ensemble export, and `scratchpad/counterfactual_ph_timing.py` (a real-outcome
behavioural check reusable for ANY future MAIN-scoring change, not just this one — arguably the more
valuable output of this attempt than the (failed) feature itself).

---

## 2. Everything else is clean — same clean-negative shape as M26

**No decision bug, no drift, no infrastructure fault:**

- **Bundle parity 100.0% (5999/5999)** — 100% in wins *and* 100% in losses. The ladder agent is
  byte-for-byte the trained clone.
- **`bc_failures` = 0** (no SafePolicy interventions); routing 6334 learned / 872 greedy, normal.
- **Lethal discipline 96.2% (377/392)**, and **the teacher's own is 96.3% (648/673)** — the clone
  matches its teacher *exactly* on KO conversion. Only 8 losing-game misses across 7 games.

**Per-turn behaviour vs the teacher** (clone live, 100 games / teacher dataset, 150 games) — the
clone is faithful, consistently ~5% *below* the teacher on every axis:

| axis | mlp-live | teacher |
|---|---|---|
| attacks per game | 5.16 | 5.64 |
| first attack turn (median) | 3 | 3 |
| attack-turn fraction | 0.744 | 0.769 |
| games that ever attacked | 95/100 | 149/150 |
| max bench (avg) | 4.63 | 4.85 |
| hand size (avg) | 11.20 | 11.37 |

---

## 3. The one real behavioural gap — combo assembly is **one turn slower**

New phase (no analogue in M26 — Kanga had no combo to assemble):

| axis | mlp-live | teacher |
|---|---|---|
| Alakazam reached (frac games) | 0.930 | 0.987 |
| …in **won** games | **1.000** | **1.000** |
| …in **lost** games | **0.851** | **0.962** |
| Alakazam median turn (loss) | **5** | **4** |
| Powerful Hand used (frac games) | 0.920 | 0.987 |
| …in lost games | 0.830 | 0.962 |
| Powerful Hand first use, median turn | **5** | **4** |
| PH hand_size avg (= damage/20) | 12.58 | 13.00 |

**Assembling Alakazam is perfectly correlated with winning: 100% of wins have it, and ~15% of
losses never get there** (teacher: 4%). The clone brings the combo online a median of **one turn
later** than its teacher.

**Honest caveat — reverse causality.** A game may lack Alakazam *because* the opponent disrupted
us, not the other way round; and the teacher's 150 games come from a different (higher-rated)
opponent pool, so this is not a controlled comparison. It is a lead, not a proven defect.

---

## 4. What this does and does not license

- The champion has **no fixable pilot leak** — this is the fourth consecutive clean parity/lethal
  audit (v1 M10, Kanga M26, nsr, now mlp). Stop looking for one.
- **The imitation line on this deck has ~4 points of headroom left** (53.5% → the teacher's 57.6%),
  and M31 measured all four ways to claim it as negative. Further tuning of this clone is not
  where the remaining value is.
- **Marnie's Grimmsnarl is the deck-level wall and it owns the top of the ladder** — 57% of the
  #1's field, holding him to 53%. Any plan that involves climbing must answer it.
- Sample sizes per archetype on *our* side are small (Dragapult n=6, Kangaskhan n=4, Grimmsnarl
  n=27); the teacher-side numbers (n=603 for Grimmsnarl) are the solid ones. Per M29, matchup mix
  also *is* the ladder-score noise, so a Grimmsnarl-heavy field depresses the score on its own.
- **Option (a), a Grimmsnarl-conditioned feature on the existing clone, is CLOSED NEGATIVE** (1-quater)
  — the first attempt (`ALAKAZAM_MATCHUP`) improved the offline proxy but made the real, targeted
  behaviour (Powerful-Hand timing) *worse* on a decisive counterfactual check against the real 27
  games. A second attempt with a less "sticky"/more per-turn-informative encoding is possible but
  unproven and each iteration costs a real Colab round-trip; not clearly worth further spend yet.
- **Option (b), cloning a Grimmsnarl pilot, is the remaining live path** — a regime change, higher
  cost (source + screen + clonability + veto) but a higher ceiling (a deck that isn't capped at
  ~53% against itself, and is the most-played deck in the band we're climbing into). Candidate
  pilots already identified from our own match history (M32.1/memory): submissionIds 55011432
  (1072 elo, 183 eps), 54785331 (977 elo, 669 eps), 54900560 (1013 elo, 443 eps), 54981508
  (1036 elo, 231 eps) — no blind leaderboard search needed.

**Nothing was built, changed, or uploaded.** `src/` untouched; both new files are `scratchpad/`
read-only diagnostics.
