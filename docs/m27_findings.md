# M27 findings — a resilience tech card for Kanga + the forced-recovery rule (2026-07-25)

**Bottom line: M26 proved Kanga's losses are deck-structural (55% bench-outs: the all-in Mega
Kangaskhan build has no plan B once its 300-HP attacker is answered) and explicitly not a
decision-policy defect. M27 took the one lever aligned with that root cause — add a resilience
card *compatible with the existing deck* — and carried it end to end. (1) Chosen swap: **Night
Stretcher (1097) ×2 for Hand Trimmer (1087) ×2** (recover the KO'd attacker; the softest flex slot,
corroborated by the same-deck #9 pilot `whereismyorbit`). (2) A pure deck swap on the frozen clone
is a **weak lever**: head_to_head is a dead-center tie, and a new de-risk probe shows why — the
recovery OPPORTUNITY is abundant (95% of losses have a Kangaskhan/Crustle in discard) but the frozen
clone PLAYS the card only ~7–9%, because a bolt-on card has no learned weight (it rides an untunable
generic PLAY-prior; the PASO 2 ceiling, now quantified). (3) The fix — Option 2 — is a **default-off,
weights-payload-carried forced-recovery pre-emption** in `ImitationPolicy._decide` that guarantees the
recovery play in the exact "clock lost, recoverable" moment. It fires correctly (10×/150 games),
roughly doubles NS play (→15% games), passes the head_to_head veto with a **slight positive lean
(+0.012, 40/26 decks)**, keeps 0 interventions, and is pinned by 5 new unit tests (full suite 190
passed / 1 skipped). Nothing uploaded — the genuine benefit vs Alakazam/Lucario/Archaludon is only
judgeable on a ladder free-roll. This is now a legitimate, arguably improved ladder candidate.**

Continues [m26_findings.md](m26_findings.md) (the diagnosis that scoped M27) and the M21–M26 Kanga arc.
New tools/artifacts: `scratchpad/probe_nightstretcher.py`, `tests/unit/test_imitation_recover_rule.py`,
`decks/kanga_nightstretcher.csv`; a default-off rule in `src/ptcg_ai/imitation/policy.py`.

## PASO 0 — the deck (`decks/懒惰的金枪鱼.csv`, all text verified vs `EN_Card_Data.csv`)

60 cards = **12 Pokémon / 13 Energy / 35 Trainers**, an extreme all-in shell. Pokémon: Mega
Kangaskhan ex `756` ×4 (300 HP, Rapid-Fire Combo, 3 prizes — the only real clock) + Dwebble `344`
×4 → Crustle `345` ×4 (150 HP wall, blanks *ex* attackers, Superb Scissors only 120). Energy: basic
{G} `1` + Grow Grass `18` ×4 + Mist `11` ×4 + Spiky `14` ×4. Existing resilience already spent:
Jumbo Ice Cream ×4 (heal 80), Battle Cage ×2, Crustle. **Softest flex slot = Hand Trimmer `1087`
×2** — symmetric hand disruption, and the same-deck #9 pilot `whereismyorbit` (17/18 identical) cut
both to fit its own tech.

## PASO 1 — candidate & PASO 2 — clonability

Chosen: **Night Stretcher `1097`** (Item, "put a Pokémon or a Basic Energy card from your discard
pile into your hand") — recovers the KO'd Mega Kangaskhan/Crustle = the exact bench-out failure;
50/76 decks in `decks/` run it, Kanga does not. Alternatives kept as fallbacks: Poké Pad `1152`
(consistency, already greedy-whitelisted), Wally's Compassion `1229` (full-heal the mega, Supporter),
Sacred Ash `1129` (bulk recovery). **Clonability finding:** a deck swap is a `decks/*.csv`-only change
— `feature_dim` (596) lives on the profile, not `deck.csv`, so `bc_kangaskhan_1052.json` loads
unchanged; re-cloning is useless (the teacher never played 1097 → zero imitation signal). **But** in
the learned MAIN context the clone scores every option, including the new card, with its weights and
does NOT fall back to greedy's whitelist (`policy.py`); a bolt-on card lands in the shared overflow
slot (`features.py:145`) with no dedicated weight, so it is played on an untunable generic PLAY-prior.

## The pure swap — measured NEGATIVE

- Legal (60/≤4/1-ACE-SPEC; Hero's Cape stays the sole ACE SPEC). Bundle on unchanged weights passes
  smoke test, dim 596, fidelity 0.652.
- head_to_head (same weights, only the deck differs), field=120, 0 interventions: n=6 −0.007
  [−0.031,+0.017]; **n=12 −0.000 [−0.016,+0.016], 33/34 decks** — dead-center tie. The clone barely
  fires the swap, so the deck plays like the original.

## De-risk probe (`scratchpad/probe_nightstretcher.py`, read-only)

Live Kanga replays are gitignored/absent this session; HALF 1 used the **teacher dataset** (same deck,
461 games — a valid proxy: M26 showed the clone matches the teacher on every behavioral axis).

- **HALF 1 — opportunity: HIGH.** In LOST games, **52.9%** of MAIN turns have a Kangaskhan/Crustle in
  our discard; **95% (190/201) of lost games** had a key attacker recoverable at some MAIN turn. The
  target almost always exists — the tech is well-targeted.
- **HALF 2 — behavior: LOW.** Instrumented tech clone, 150 sim games: NS played in **9% of games**,
  **7% of the 229 legal offers**. The generic prior under-plays it — the mechanism: NS has no
  dedicated weight, and the only state-conditioning (PLAY⊗snapshot) is shared across ALL items, so the
  clone cannot learn "board thinning → recover" for NS specifically.

## Option 2 — the forced-recovery rule (the fix)

A **default-off** pre-emption in `ImitationPolicy._decide`, carried in the weights payload as
`recover_rule = {recover_id, clock_id}` (an absent field = every existing agent byte-identical). No
entrypoint/builder change — the shipped entrypoint already builds `ImitationPolicy(weights)`. When
present, before the learned scorer, if context is MAIN and the clock (756) is absent from play AND
hand yet sits in discard and the recover Item (1097) is a legal play, it forces that play. It is a
**routing** change, not a feature change, so weights stay valid (dim unchanged) — unlike editing
`DeckProfile` featurizer fns (the M17 corruption lesson).

- **Fires + play-rate up:** probe pre-empt fired **10×/150 games**; NS played **15% of games** (from
  9%), **13% of offers** (from 7%); WR 0.907 unchanged; 0 interventions.
- **Veto pass, positive lean:** head_to_head n=12 vs kanga-orig = **+0.012, 90%CI [−0.003,+0.026],
  40/26 decks better/worse** — still a formal tie in the ±0.05 band, but no harm and a mild positive
  nudge (vs the pure swap's 0.000 / 33-34). Recovery is now *guaranteed* at the critical moment.
- **Regression clean:** 5 unit tests (`tests/unit/test_imitation_recover_rule.py`) pin
  fire/no-op(clock-in-play)/no-op(clock-in-hand)/no-op(no-target)/default-off; full suite **190 passed,
  1 skipped**.

## Verdict

- **Standing M26 conclusion holds for the deck itself**, but Option 2 gives Kanga a real resilience
  mechanism at near-zero risk: no offline harm, a slight positive lean, guaranteed recovery in the
  losing scenario. The genuine benefit vs the high-HP hitters (Alakazam/Lucario/Archaludon) is only
  testable on a **ladder free-roll** — the offline gauntlet remains a veto, not a ranker (M22/M25).
- **Reusable lesson:** to make a bolt-on card actually fire despite the frozen generic prior, a small
  deterministic **pre-empt in the rung-6 policy** (weights-payload-gated, ablatable, dim-safe) is the
  clean lever — not a deck swap alone, not a re-clone.

## Files

New (tracked): `docs/m27_findings.md`, `decks/kanga_nightstretcher.csv`,
`scratchpad/probe_nightstretcher.py`, `tests/unit/test_imitation_recover_rule.py`; modified
`src/ptcg_ai/imitation/policy.py` (default-off rule). Gitignored local artifacts (regenerate on
demand): `data/models/bc_kangaskhan_1052_ns.json` (= `bc_kangaskhan_1052.json` + a `recover_rule`
key) and `build/imitation-kanga-nsr.tar.gz` (`build_imitation.py decks/kanga_nightstretcher.csv
data/models/bc_kangaskhan_1052_ns.json imitation-kanga-nsr`). Not uploaded; the ladder call is the
user's.

## Next (M28, open)

Either (a) upload `imitation-kanga-nsr` as a ladder free-roll and judge on ~50 eps — the only real
test of the resilience benefit — or (b) if pursuing further, generalize the pre-empt (e.g. Crustle
recovery, or a heal-based survival card) or return to the M23/M25 clone hunt for a ~900–1050 deck
that natively runs recovery so the behavior is learned, not bolted on.
