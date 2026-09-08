# M7 findings — imitation learning breaks the pilot wall (2026-07-08)

**Bottom line: behavior cloning of the ~650-elo player `greengreenpurple` is the
FIRST agent to beat greedy-v5.** Every M5/M6 attempt lost to greedy-v5 (best
0.53); this scores **0.717** against it (n=300, 95% Wilson [0.663, 0.765], lower
bound clears the 0.65 ship bar). It attacks piloting skill directly — fitting a
scorer to a strong player's actual decisions — instead of hand-deriving rules,
which is exactly the lever M6 flagged as unspent.

## What was built (`src/ptcg_ai/imitation/`)

- **`kaggle_replay.py` + `dataset.py`** (dev): load one player's decisions from
  Kaggle replay JSONs, honoring the verified **+1 action lag**
  (`steps[i].observation` ↔ `steps[i+1].action`) and keeping only `ACTIVE` cells
  with a live `select`. Built `data/imitation/greengreenpurple.jsonl.gz`:
  **18,533 decisions / 325 games / 0 illegal actions.**
- **`features.py`** (SHIPPED, pure stdlib): one 386-dim per-option featurizer used
  offline AND live (no train/serve skew). Exploits the fixed deck (15-id one-hots),
  option-type × state interactions, and hand-computes **Rocket Rush = 30 × Team
  Rocket Pokémon in play** (+50 Maximum Belt vs ex, ×2 weakness) since GameState
  reads its structured damage as 0.
- **`train.py`** (dev, numpy): per-context linear conditional-logit (softmax over
  options, cross-entropy on the choice); single-pick contexts train vectorized,
  multi-pick (TO_HAND search, DISCARD) use Plackett-Luce. Learns 5 contexts;
  everything else defers to the greedy handler (already 88–100% teacher-match).
- **`policy.py`** (SHIPPED, pure stdlib): `ImitationPolicy(GreedyPolicy)` — scores
  options with the learned weights, greedy fallback on unseen context / prize pick
  / any error. Weights: `data/models/bc_650_v1.json` (~21 KB).

## Results

Offline held-out top-1 (split **by game**, no leakage):

| context | BC | greedy | random |
|---|---|---|---|
| MAIN | **0.862** | 0.356 | 0.162 |
| TO_HAND | 0.925 | 0.521 | 0.412 |
| DISCARD | 0.815 | 0.408 | 0.207 |
| TO_ACTIVE | 0.882 | 0.781 | 0.284 |
| SETUP_ACTIVE | 1.000 | 0.955 | 0.795 |
| weighted | **0.874** | 0.419 | — |

Arenas (swapped sides, Wilson 95%):

| arena | score | note |
|---|---|---|
| (a) vs greedy-v5, n=300 | **0.717** [0.663, 0.765] | promotion gate — **SHIP** |
| (b) vs greedy(650-deck), n=200 | **0.985** | pure pilot delta (greedy pilots this deck to **0.02**) |
| (c) vs Lucario, n=150 | 0.647 | > v5's ≈0.60 bar |
| (c) vs Dragapult, n=150 | 0.967 | > v5's ≈0.94 bar |
| (d) BC mirror, n=50 | 0.640 | sanity (first-player edge) |

Across ~850 arena games: **bc_failures=0, SafePolicy interventions=0, ~0.06 s/game**
— robust and submission-safe.

## The decisive proof

On the **identical** 650 deck, greedy pilots it to 0.02 (M7-0 smoke) and BC to
0.985 vs that same greedy (arena b). The deck did not change — the pilot did. This
is the piloting-skill gap M6 diagnosed (them 0.62 / us 0.19 on Vibechu's deck),
now closed by imitation rather than worked around.

## Shipped

`build/imitation-v1.tar.gz` (2.0 MiB): 650 deck + BC weights, entrypoint tiers
**imitation → greedy → safe-random**. The default `ptcg build-submission` is
unchanged (no weights bundled → `_IMITATION_READY` False → greedy-v5 behavior), so
greedy-v5 stays a re-shippable fallback. Build via `scratchpad/build_imitation.py`.

## Caveats & next lever

- **Arena ≠ ladder** (M6 bias lesson): the expected ladder outcome is the teacher's
  level (~650), because BC cannot exceed whom it imitates (a 47%-winrate, stable
  650 player). The arena wins corroborate; the ladder is the real test.
- **To pass 650** (sanctioned stretch, benchmark-gated): reuse the BC logits as move
  priors and/or fit a learned V on the dataset's win labels inside the existing
  rung-5 determinized search stack — the principled fix to M6's hand-tuned V. The
  whole pipeline is player-agnostic and re-runnable on a higher target with fresh
  logs the user can source.
