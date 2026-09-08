# M28 findings — cloning the #1 (Yushin Ito) + BREAKING the fidelity ceiling with an MLP (2026-07-26)

**Bottom line: the user asked the load-bearing question — every promising clone caps at 60–65% MAIN
fidelity; can we clone with HIGHER fidelity? — pointing at Yushin Ito (rank #1, 9 days of games).
Reframed honestly: "higher offline fidelity" is a lever this project FALSIFIED four times (M11/M14 an
MLP raised fidelity +9pts then played WORSE; the gauntlet is a veto not a ranker, M22/M25). But the
never-pushed axis is TEACHER STRENGTH — we had maxed fidelity/clonability on ≤1052-elo teachers and
never cloned the #1. We built the first real clone of the #1's Stage-2 Alakazam combo and found: (1)
the linear model DOES scale with data — MAIN 0.470 (generic,150g) → 0.545 (ALAKAZAM profile,150g) →
0.603 (full 1000g); (2) most importantly, **an MLP on the big data BREAKS the ceiling: held-out
(TEST) MAIN 0.556 → 0.780, a +0.224 lift, with a SMALLER overfit gap than linear (0.018 vs 0.038)** —
the exact OPPOSITE of M11/M14, because big data (1000 games) + a strong/consistent teacher lets the
hidden layer capture real combo skill instead of noise. The MLP clone PILOTS the combo (0
interventions, 0.955–0.963 WR vs the field) and BEATS the champion kanga on the gauntlet by +0.074
[+0.052,+0.095]. Nothing shipped yet — the ladder is the only judge and this is exactly the offline
signal that has misled before; the free-roll (paused for M29's variance analysis) is the verdict.**

Continues [m23_findings.md](m23_findings.md) (which triaged Yushin out at the greedy gate without ever
building a clone) and the fidelity-ceiling thread (M11/M14). See also [m29_findings.md](m29_findings.md).

## The target — Yushin Ito's deck (`decks/yushinito.csv`, verified vs `EN_Card_Data.csv`)

Stage-2 **Alakazam combo/control**, the highest-skill archetype in the pool: Abra 741 → Kadabra 742 →
Alakazam 743 (Rare Candy 1079 skips the Stage 1). **Alakazam "Powerful Hand" (1072, {P}) = "place 2
damage counters × cards in hand"** = **20 × hand_size, PROSE damage** the generic model reads as 0
(and, because it PLACES counters, Weakness/Resistance do NOT apply). Fezandipiti "Cruel Arrow" (183) =
flat 100. Triple draw engine (Psychic Draw on evolve, Dudunsparce 66/305, Fezandipiti 140), Shaymin
343 bench-protection, Enhanced Hammer 1081 energy denial, Dawn 1231 line-tutor. **M23 triaged this
DISCARD (greedy strength 0.250)** — but that gate is *greedy* piloting a combo it cannot sequence; a
high-fidelity clone is exactly what looks past it.

## Phase 0 — data + generic clonability (bypassing quick_screen's strength short-circuit)

`quick_screen.py` short-circuits at the 0.40 greedy-strength gate (Yushin ≈0.25) and never trains the
clone, so we ran dataset + generic-profile linear BC directly. 150/1000 episodes → 11,661 decisions
(6,773 MAIN); generic **MAIN 0.470** (greedy 0.196). Marginal pass (data abundant; the generic profile
is blind to hand_size — the combo's core variable — so 0.470 is a floor).

## Phase 1 — the ALAKAZAM DeckProfile (`src/ptcg_ai/imitation/deck_profiles.py`, dim 658)

Additive, mirrors `KANGASKHAN_1052`: `damage_fn` models Powerful Hand (20×hand_size, no W/R) + Cruel
Arrow (100); the snapshot foregrounds **hand_size** plus {P}-on-active, the Alakazam-line board,
Shaymin/Rare-Candy, and a "Powerful Hand already lethal" flag. MAIN 0.470 → **0.545** (150g). Then the
"lots of data" premise: on the full **1000 episodes** (~38.8k MAIN train), linear MAIN → **0.603**
(profile +0.075, data +0.058, both real; big data also unlocked more learnable contexts).

## Phase 2 — the capacity experiment (`scratchpad/train_mlp_alakazam.py`)

3-way game split (train/val/**test**), reusing `scratchpad/train_mlp_main.py`'s MLP machinery, h=48,
3-seed ensemble, reporting the train−test overfit gap:

| MAIN scorer | TEST (held-out) | overfit gap (train−test) |
|---|---|---|
| linear | 0.556 | +0.038 (it even UNDER-fits: train only 0.594) |
| **MLP ensemble** | **0.780** | **+0.018** |

**+0.224 held-out lift, and the MLP overfits LESS than linear** (3 seeds agree 0.771–0.779; no leak —
`split_by_game`, M24). This inverts M11/M14: there the MLP overfit a mediocre small-data teacher; here
big data + the #1 teacher make the extra capacity capture the genuinely nonlinear combo policy
("draw up, then Powerful-Hand when lethal") the linear model literally cannot represent.

## Phase 3 — shippable weights, bundle, gauntlet veto

`scratchpad/build_mlp_alakazam_weights.py` trains the final ensemble on the full 38,783 train (ensemble
val 0.792) and writes `data/models/bc_alakazam_mlp.json` (MAIN = mlp_ensemble; other contexts linear
from `bc_alakazam_full.json` — the M11 `bc_650_v2` flow; `policy._score` already ships mlp_ensemble).
Bundle `build/imitation-yushin-mlp.tar.gz` (2.9 MiB, dim 658, smoke test OK). **head_to_head veto**
(field 120, n=12): vs nsr **+0.058 [+0.039,+0.078]**, vs the real champion **plain kanga +0.074
[+0.052,+0.095], 0 interventions → BEATS kanga** (best gauntlet number the project has produced). The 0
interventions + 0.96 WR prove the MLP clone actually PILOTS the combo — it does not whiff to greedy
(0.25), resolving the main "78% clone mispilots" worry.

## Verdict & caveats

- **The fidelity ceiling is real but BREAKABLE** — not with the linear model, only with an MLP on big
  data from a strong teacher. This reopens model capacity as a lever *specifically in the abundant-data
  + strong-teacher regime* (the M11/M14 "no MLP" verdict was regime-specific, not universal).
- **The teacher-strength axis is now testable at the top** for the first time (a 0.78 clone of the #1).
- **Binding caveat:** offline fidelity has NEVER translated to ladder here; the gauntlet is a veto and
  a combo crushing a greedy field is a known false-positive pattern (M22 ITF beat kanga on the gauntlet
  then LOST the ladder). GREEN LIGHT for a ladder free-roll with cautious optimism — NOT a prediction.

## Files

Tracked: this doc, `src/ptcg_ai/imitation/deck_profiles.py` (additive `ALAKAZAM`),
`scratchpad/train_mlp_alakazam.py`, `scratchpad/build_mlp_alakazam_weights.py`. Gitignored local
artifacts (regenerate on demand): `data/imitation/yushinito_full.jsonl.gz`
(`dataset.build_decision_dataset(replays/54773249, "Yushin Ito", …)`), `data/models/bc_alakazam_full.json`
(`ptcg_ai.imitation.train … ALAKAZAM`), `data/models/bc_alakazam_mlp.json`
(`build_mlp_alakazam_weights.py`), `build/imitation-yushin-mlp.tar.gz`
(`build_imitation.py decks/yushinito.csv data/models/bc_alakazam_mlp.json imitation-yushin-mlp`),
`replays/54773249/` (`download_competitors.py --submissions 54773249`). Tests: 191 passed. Not shipped/
uploaded; the ladder free-roll is paused for [m29_findings.md](m29_findings.md).

## Next

Upload `imitation-yushin-mlp` as a ladder free-roll — but per M29, run it CONCURRENTLY with a fresh
plain-kanga baseline (shared field/epoch) and judge the yushin−kanga GAP after ~50 eps, not the
absolute score. If it wins by a margin beyond the ~±100-elo field noise, the teacher-strength × MLP
recipe is a breakthrough; if not, top-clone fragility vs real disruption is confirmed.
