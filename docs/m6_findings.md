# M6 findings — search + deck/pilot investigation (2026-07-07)

**Bottom line: no change beat greedy-v5. It remains the shipped agent. The
binding constraint is confirmed to be *pilot skill on engine decks*, which
heuristics and 1-ply search cannot cheaply replicate.** 11 controlled
experiments; every alternative lost to greedy+sample and (where tested) to the
meta decks worse than greedy+sample does.

## What was built (preserved as research, NOT shipped)

- **Rung-5 determinized search** (ADR-0010): `planning/determinize.py` (exact
  my-zone bookkeeping incl. serial-aware limbo handling; 3.2% fallback on real
  games, 0% on MAIN), `planning/session.py` (context-managed search lifecycle +
  cg→dict observation converter), `decision/evaluator.py` (linear,
  perspective-safe, antisymmetric `V`), `decision/search.py` (all-contexts
  receding-horizon `SearchPolicy` with greedy rollout/fallback + `greedy_bias`).
  GameState gained `build_for`, `reserve_attackers`, `max_threat`. 40+ new tests,
  all green.
- **M6-0 probes** (all passed): `search_begin` works at every live context;
  native RSS peak ~33 MiB (Q8 resolved — memory is not the constraint); ~0 coin
  nodes in my-turn rollouts (so `manual_coin=False`).
- **Rule-based pilot extension**: whitelisted draw abilities + a fastest-attacker
  active-selection override (`_pick_best_pokemon`) in `decision/rule_based.py`.
- **Top-player deck extraction**: `scratchpad/extract_top_decks.py` recovers a
  player's exact 60-card list from the first action of each replay. Wrote
  `decks/vibechu.csv` (#1, Slowking engine) and `decks/majkel.csv` (#2).

## The experiments (all vs greedy-v5 = greedy+sample, swapped sides)

| Lever | Result | Why it failed |
|---|---|---|
| Search on sample deck | **0.44** (G1, n=300) | greedy's fixed priorities are near-optimal on this simple deck; a linear `V` deviates and loses. Weight tuning made it worse (0.35–0.39). |
| Search on Ogerpon | slow, abandoned | complex contexts → 10 s/game; superseded |
| Search on the #1 deck (Vibechu) | **0.19** | can't pilot its Slowking engine (they score 0.62 on the *same deck*) |
| Rule-based ability pilot, all-ex Psychic deck | **0.21** | attackers cost 3 energy; pilot attaches 1/turn, no accel → 5 attacks in 6 games |
| Cheap non-ex attacker deck vs Lucario | **0.09** (bar 0.60) | 110–140 HP non-ex hitters too weak to trade |
| Cheap non-ex attacker deck vs Dragapult | **0.45** (bar 0.94) | same |
| Better-wall Stage-2 (Slaking ex) | **0.10** | two-step evolution bricks → bench-out 17/30 |
| M5 decks (Lucario, water_v2, Ogerpon, +2) | all < 0.53 | (prior milestone) energy-race / pilot gap |

## The decisive diagnoses

1. **Pilot skill is the wall, proven on identical decks.** Vibechu's exact
   Slowking deck: them 62%, us 19%. The 43-point gap is pure piloting.
2. **The ability pilot mechanically works** — 81% ability-fire rate on the
   Psychic deck (matches the #1 player's 0.81/turn). It just can't rescue a deck
   that's too slow or too weak.
3. **Our pilot only excels at one archetype: a strong one-step-evolution wall
   with a cheap attack and lots of energy** — i.e. the sample deck. Mega
   Abomasnow ex is Stage-1 from Snover (one evolution → reliable setup, 78%
   board), a damage-reducing 350-HP wall with a 3-energy 200 attack. Every
   deviation breaks one of the pilot's competencies: cheaper attackers are too
   weak; harder hitters cost 4 energy or have "can't attack next turn"; better
   walls need two-step evolution that bricks.
4. **The arena gate "beat greedy+sample" is biased** toward that wall archetype:
   greedy+sample wins the mirror-pilot arena easily yet only scores ~497 on the
   ladder, because real ladder opponents out-pilot greedy-piloted meta decks.
   Judging decks against the meta decks (Lucario/Dragapult) instead did not
   rescue any candidate — they lost there too.

## The one lever not spent

**Imitation learning** from the top players' 105 logged games (~5,000 decisions)
— featurize each decision + fit a scorer that reproduces their choices — is the
only remaining lever that attacks piloting skill directly rather than around it.
It is CPU-cheap and principled, but a multi-day build with real uncertainty. Not
pursued this round (user elected to stop and keep v5).

## Reusable assets for any future attempt

- `scratchpad/extract_top_decks.py` — exact top-player deck reconstruction.
- `decks/vibechu.csv`, `decks/majkel.csv` — proven top-tier lists.
- The rung-5 search stack + evaluator + determinizer (tested, ADR-0010) — the
  leaf `V` and the `search_begin`/`step` plumbing an imitation-tuned scorer or a
  deeper search would reuse.
- `scratchpad/arena_m6.py` — arena with search/rule/greedy pilots, meta gauntlet,
  bias/tuning modes, and search diagnostics (fallback rate, plays/turn).
- M6-0 probes (`scratchpad/probe_m6_*.py`) — the SDK search-API characterization.
