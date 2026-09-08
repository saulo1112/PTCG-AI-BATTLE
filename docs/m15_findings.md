# M15 findings — m093jp with the authored profile: still ties kenN2439, screening pool exhausted (2026-07-17)

**Bottom line: m093jp (the closest of 4 screened candidates, generic-profile MAIN 0.732) was retrained
with the already-authored `BELLIBOLT_940` DeckProfile (m093jp plays the identical decklist to
kenN2439/v3). Result: MAIN accuracy 0.771 — a marginal +0.039 lift over the generic screen, but a
statistical TIE with v3's own 0.770 on the same deck+profile (M9). Since v3 lost the field gauntlet
−0.214 vs v1, and m093jp clones no better than v3, there is no basis to expect a different outcome.
Gate G1 (hard bar MAIN ≥ 0.78) failed; per the pre-registered plan, the candidate is discarded WITHOUT
running the arena/gauntlet phases. All 4 screened teacher candidates are now exhausted.**

## Why m093jp needed no new deck authoring

`decks/m093jp.csv` is byte-identical (15 distinct card ids, same attack vocabulary) to kenN2439's
Iono's Bellibolt ex deck, already covered by the hand-authored `BELLIBOLT_940` profile
(`src/ptcg_ai/imitation/deck_profiles.py:785-798`) — prose-damage corrections for Voltaic Chain
(20+20×board-{L}) and Quick Attack (guaranteed 10, ignoring the coin flip), per-card energy
priorities (`wants_940`), correct `opp_target_dim=0` (no gust trainers). Retraining just meant
pointing `ptcg_ai.imitation.train` at the existing profile name — no new code.

## Retrain result vs the three anchors

```
uv run --group dev python -m ptcg_ai.imitation.train \
    data/imitation/m093jp_screen.jsonl.gz decks/m093jp.csv \
    data/models/bc_m093jp_v1.json BELLIBOLT_940
```

| anchor | MAIN acc | note |
|---|---|---|
| generic-profile screen | 0.732 | cheap triage number (M14 recommendation) |
| **m093jp, authored profile (this run)** | **0.771** | 73,175 rows / 556 games, greedy baseline 0.242 |
| v3 — kenN2439, same deck+profile (M9) | 0.770 | lost field gauntlet macro 0.662 vs v1's 0.876 (delta −0.214) |
| v1 (champion) | 0.862 | still an 0.091 gap, unclosed |

The authored profile confirmed the M14 hypothesis that the generic screen underestimates true
clonability (+0.039 lift) — but the ceiling it revealed is the same one kenN2439 already hit, not a
higher one. Full per-context table in the training log: `ATTACH_FROM` 0.637, `DISCARD` 0.656,
`TO_ACTIVE` 0.944, `SWITCH` 0.961, `TO_BENCH` 0.382 (near-random, unlearnable), `TO_HAND` dropped
(lift < 0.05 over greedy's already-strong 0.716).

## Why this ends the search here (not just this candidate)

Since m093jp and kenN2439 pilot the *identical* deck through the *identical* profile, clone fidelity
is a direct, controlled proxy for "how skillfully does this teacher play this deck" — no deck-quality
confound. Two teachers converging to the same fidelity (0.770 vs 0.771) is strong evidence they are
comparable pilots of this deck, and one of them (v3) already has a decisive, measured field-gauntlet
loss to v1. Phases 1 (pilot-duel mirror) and 2 (field/strong gauntlets) were **not run** — the
pre-registered Gate G1 existed precisely to avoid spending that compute on a candidate this unlikely
to differ from an already-rejected result.

**All 4 screened candidates are now exhausted:** budew (0.480), eduardorochadeandrade (0.610),
legendbrothers (0.623), m093jp (0.732 generic / 0.771 authored — the only one close enough to justify
hand-authoring). None clears a bar that would justify the arena/gauntlet phases. Combined with M9's
finding (clonability > teacher elo, exhaustively confirmed across 3 human teachers + this one), the
teacher-search lever is now closed pending a **genuinely new** candidate — not a re-check of anything
already screened.

## What remains open (updates the M12 findings list)

Per [m12_findings.md](m12_findings.md)'s open-options list, option (b) "a different, clonable strong
teacher" is now more precisely: **exhausted for all 4 already-screened candidates**; still open only
if a new, not-yet-seen simple-deck ~700–900 teacher is sourced. Options (a) ladder-as-instrument
free-rolls, (c) self-play RL, and (d) deck-level change remain untouched, exactly as M12 left them.

## Files

- Created: `data/models/bc_m093jp_v1.json` (44 KB, authored-profile clone, not shipped — a
  research artifact, not a candidate weight).
- No changes to `src/`, `bc_650_v1/v2.json`, `bc_940_v1.json`, or any shipped submission.
