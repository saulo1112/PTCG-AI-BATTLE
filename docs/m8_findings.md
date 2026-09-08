# M8 findings — imitation-v2: climbing the ladder by re-targeting BC (2026-07-08)

**Bottom line: imitation-v2 (behavior clone of the ~800-elo Mega Lucario pilot
`[RU] Nikita Kuznetsov`) beats imitation-v1 head-to-head 0.660 (n=300, Wilson
[0.605, 0.711]).** M8 turned M7's one-off clone into a *repeatable ladder-climb
recipe*: a new stronger teacher is now **data + one `DeckProfile` instance**, not
new code. imitation-v1 (ladder ~657) stays live as the re-shippable fallback.

## What changed since M7

- **`DeckProfile` refactor** (`src/ptcg_ai/imitation/deck_profiles.py`, SHIPPED,
  stdlib-only): the featurizer is now parametric over a profile carrying the
  deck's card/attack vocabularies, target-Pokémon one-hot, per-attack damage
  corrections, and state-snapshot extractors — all as module-level functions
  (never lambdas) for ship safety. Two profiles: `TR_650` and `LUCARIO_800`.
  **Gate passed: `TR_650` reproduces M7's 386-dim vectors bit-for-bit** (sha256
  over 2,000 fixed decisions identical pre/post refactor), so `bc_650_v1.json`
  and imitation-v1 are byte-for-byte unchanged.
- **Data-driven context selection** (`train.py`): no hard-coded learned-context
  list. Per context, verdict = learn (≥250 rows AND greedy-as-predictor <95% AND
  not degenerate), zeros (all options resolve to one card id — degenerate; a zeros
  vector makes the live take-k rule fire), or skip. This auto-discovered a **live
  greedy bug**: on Aura Jab's from-discard attach (ATTACH_TO, minCount=0) greedy
  returned `[]` (never attached); v2's zeros vector fixes it for free.
- **Win-weighting hook** (`--alpha`) implemented for the "exceed the teacher"
  ablation; shipped v2 uses α=1.0 (pure BC) — the safe default that reproduced
  the teacher's level.

## The 800 teacher & deck

`[RU] Nikita Kuznetsov`, 434 replays, 219W-214L (50.6%), ONE deck 435/435:
Mega Lucario ex Fighting (`decks/lucario800.csv`, **50/60 identical to
`decks/meta_lucario.csv`** — the top meta archetype greedy piloted to only 0.53).
30,802 logged decisions. Structured-damage deck (unlike the 650 swarm's prose
Rocket Rush) except Cosmic Beam (0 without a benched Lunatone; ignores W/R),
hand-computed in `damage_800`. New featurizer pieces: F-energy-in-discard (Aura
Jab fuel), lunatone/solrock-in-play flags, Gravity Mountain / Hero's Cape flags,
and a 4-dim **opponent-target block** for Boss's Orders / on-evolve gusts (SWITCH
is a mixed own/opp context). FEATURE_DIM 467.

## Results

Offline held-out top-1 (split by game):

| context | verdict | BC | greedy | random |
|---|---|---|---|---|
| MAIN | learn | **0.694** | 0.374 | 0.145 |
| ATTACH_FROM | learn | 0.753 | 0.116 | 0.253 |
| SWITCH | learn | 0.825 | 0.701 | 0.315 |
| TO_ACTIVE | learn | 0.873 | 0.633 | 0.275 |
| TO_HAND | learn | 0.666 | 0.468 | 0.390 |
| SETUP_ACTIVE | learn | 0.979 | 0.833 | 0.797 |
| ATTACH_TO, DISCARD_ENERGY, ACTIVATE, IS_FIRST, DRAW_COUNT | zeros | take-k | — | — |

Arenas (swapped, Wilson 95%):

| arena | score | note |
|---|---|---|
| (a') vs imitation-v1, n=300 | **0.660** [0.605, 0.711] | **PROMOTION — SHIP** (beats our champion) |
| (b') vs greedy(lucario800), n=200 | 0.650 | pilot lift on the same deck |
| (c') vs greedy-v5 (sample), n=150 | 0.733 | beats v5 |
| (c') vs greedy(meta_lucario), n=150 | 0.727 | wins the near-mirror |
| (c') vs greedy(meta_dragapult), n=150 | 0.780 | no collapse |
| (d') v2 mirror, n=50 | 0.420 [0.294, 0.558] | sanity (variance; CI spans 0.5) |

Across ~850 arena games: **bc_failures=0, SafePolicy interventions=0**,
0.11–0.27 s/game — robust and submission-safe.

## Why the margin over v1 (0.66) is smaller than M7's over greedy (0.72)

v1 is a far stronger opponent than greedy-v5 was. The elo formula predicts an
800-player scores ~0.70 vs a 657-player; 0.66 is in range and clears the ship
bar. The pilot-lift over greedy is also smaller (0.65 vs M7's 0.985) because the
Lucario deck is more greedy-friendly (structured damage, one 340-HP attacker) —
greedy pilots it to 0.28 vs v1, vs 0.02 for the swarm.

## Shipped

`build/imitation-v2.tar.gz` (2.0 MiB): `lucario800.csv` + `bc_800_v1.json`
(profile LUCARIO_800). Entrypoint tiers imitation → greedy → safe-random; default
`ptcg build-submission` still greedy-v5 (`_IMITATION_READY` False without weights).
Build via `python scratchpad/build_imitation.py decks/lucario800.csv
data/models/bc_800_v1.json imitation-v2`.

## Next lever

The recipe is now mechanical — repeat on the next stable one-deck player up the
ladder (~900+) with fresh logs: extract deck → add a `DeckProfile` instance →
dataset → train → arena. The other sanctioned stretch (BC logits as priors / a
learned V inside the rung-5 search stack) remains the path to *exceed* a teacher
rather than match one, if climbing by re-targeting plateaus.

---

# M8.1 addendum — imitation-v2 UNDERPERFORMED on the ladder; diagnosis + a
# corrected promotion gate (2026-07-09)

**M8's "next lever" above was wrong to trust untested — read this before
repeating the recipe.** Once imitation-v2 accumulated ~48 real ladder games,
its score was **≈610**, well below imitation-v1's **≈688** — despite v2 having
beaten v1 0.660 in the M8 head-to-head arena. v1 was reinstated as the live
best agent; v2 was never a true regression risk (Kaggle keeps your best score)
but the gap demanded explanation before trying again.

## Diagnosis (all measured on v2's 49 real ladder replays)

Ruled out first, with hard evidence: BC did engage correctly on Kaggle (100%
action match between the shipped bundle's logged choices and a local replay of
`ImitationPolicy` across 3,077 real decisions, 0 exceptions) — this was not an
infrastructure bug. Lethal-attack discipline was fine (3.0% turn-level misses,
better than the 800 teacher's own 7.1%). No timeouts, no illegal deck, no
stuck/looping games.

**Root cause 1 — a real, small bug.** The context-classification rule in
`train.py` marked any context "degenerate" (→ a zeros weight vector → the live
policy just takes the first `k` options) whenever ≥99% of decisions had all
options resolving to the same card id. That's correct for CARD/ENERGY selects
(e.g. discarding identical energies), but NUMBER (`DRAW_COUNT`) and YES/NO
(`IS_FIRST`, `ACTIVATE`) selects *also* resolve to `card_id=None` for every
option, so they were wrongly swept into "degenerate" too. On the ladder this
meant v2 drew the **minimum** number of bonus cards (often literally 0) on
every mulligan-bonus `DRAW_COUNT` decision (11/11 observed), instead of the
teacher's/greedy's correct "always take the max". IS_FIRST/ACTIVATE were
accidentally harmless (YES sat at option index 0 in 100% of observed cases),
so only `DRAW_COUNT` actually cost anything — small, ~0.22 bad decisions/game.

**Root cause 2 — the real cost: insufficient clone fidelity.** v2's offline
MAIN-context accuracy was 69.4%, well below v1's 86.2%. On the ladder this
showed up as a causal chain: v2 over-clicked Lunatone's "Lunar Cycle" ability
(88.7% of offers taken vs the teacher's own 69.2%) — that ability's cost is
discarding a Fighting Energy *from hand* — which burned the hand energy needed
to pay for the deck's big attacks (Mega Brave 270, Wild Press 210), so v2's
attack mix skewed toward the cheap/conditional Cosmic Beam (70, or 0 without a
benched Lunatone) instead. The featurizer's state snapshot tracked hand *size*
but not hand *composition* (how much energy, how many draw-supporters, etc.),
so the clone had no signal to know it couldn't spare the energy. Net effect:
lost prize races (9 of 24 ladder losses took ≤1 prize) plus two losses from
running its own deck out **while ahead** on prizes, against mill-style
opponents (Dragapult, TR Articuno) it never learned to play around.

**Root cause 3 — the real process failure.** M8's promotion gate was "beat the
current champion head-to-head". That is provably insufficient: v2 beat v1
0.660 head-to-head yet lost to it on the real ladder. The ladder is a diverse
*field* of opponents, not one matchup — a classic non-transitivity trap
(A beats B, B beats the field better than A does).

## What was fixed, and what still failed

Built `imitation-v2.1`: (a) the `train.py` degeneracy rule now only allows a
zeros-vector verdict for CARD/ENERGY-like selects (`SelectKind` ∈
{CARD, ATTACHED_CARD, CARD_OR_ATTACHED_CARD, ENERGY}); COUNT/YES_NO selects
route to "skip" → the correct greedy handler (`_max_number`, `_is_first`,
`_safe_default`). (b) A new `DeckProfile` (`LUCARIO_800_V2`, feature dim 633,
vs 467 for the original `LUCARIO_800`) adds hand-composition features (F-energy
count in hand, draw-supporter/Boss/PowerPro/search-item/basic/evolution counts
in hand) and the active Pokémon's energy count, so the clone can see what v2.0
couldn't. Retrained: offline MAIN accuracy rose to 71.1% (weighted learned-
context accuracy 72.4% vs v2.0's 71.0%) — a real but modest gain.

**Also built the corrected promotion gate**: `scratchpad/field_gauntlet.py`
extracts every distinct legal opponent deck from our own ladder replay history
(97 decks at the time of this run — real mills, stalls, mirrors, meta decks,
things the single head-to-head arena never tested against) and evaluates a
candidate against the whole field under greedy piloting, reporting a **paired
per-deck delta vs the current champion** with a bootstrap 90% CI.

**Result: v2.1 still lost the field gauntlet.** Macro win rate over the 97-deck
field: v1 0.883 > v2.1 0.832 > v2.0 0.820. Paired delta v2.1 − v1 = **−0.050**
(90% CI **[−0.085, −0.014]**, entirely negative), worse on 50 of 97 decks,
better on only 22. **The field gauntlet's ranking (v1 > v2.1 > v2.0) matches
the real ladder's ranking (v1 ≈688 > v2.0 ≈610) — it is a validated
predictor.** Meanwhile v2.1 still "won" the head-to-head arena against v1
(0.625) — confirming that metric alone is not trustworthy for a ship decision.
**imitation-v2.1 was NOT shipped.** imitation-v1 remains the live best agent.

## The corrected lesson for the next attempt

1. **Clonability screen a candidate teacher BEFORE building anything**: prefer
   a simple, single-evolution-line, structured-damage, low-gust-trainer deck
   (like the 650 swarm) over a mechanically complex one (like the 800
   Lucario), even at some cost in the teacher's own elo. See
   `docs/handoff.md` §7 for the full checklist.
2. **Never ship on a head-to-head arena result alone.** Run
   `scratchpad/field_gauntlet.py` against the current champion before any
   ship decision; require the paired-delta CI to clear break-even.
3. The `DeckProfile`/`train.py`/`policy.py` machinery, the degeneracy-rule
   fix, and `field_gauntlet.py` are all permanent, reusable improvements —
   kept in the codebase even though v2.1 itself was not shipped.
