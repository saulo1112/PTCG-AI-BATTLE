# M11 findings — a neural-net MAIN scorer (imitation-v1.2): big offline gain, flat on the saturated gauntlet, shipped as a free-roll (2026-07-10)

**Bottom line: a 1-hidden-layer MLP replacing v1's linear MAIN scorer lifts held-out
MAIN accuracy +9.3 points (0.858 → 0.952 on a clean test split), but that fidelity
gain does NOT convert to a field-gauntlet edge (paired delta v1.2−v1 = −0.001, a dead
tie) — because the greedy-piloted gauntlet is saturated (both clones win ~90%) and
can't resolve a subtle improvement that would only show against strong ladder
opponents.** The strict pre-registered ship gate (mean ≥ 0 AND 90%CI-lo > −0.02) was
narrowly missed, but v1.2 is a TIE, never a regression (better on 39/120 decks, worse
on 29). By explicit user decision it was shipped as a low-risk free-roll: Kaggle keeps
your best score, so the real ladder is the only test with the resolution to measure
the fidelity gain. **Build `build/imitation-v1.2.tar.gz`, verified, pending upload.**

## Why a policy network, not a value network

M10 showed BC-guided *search* with a learned value net degrades v1 (the coarse V
can't rank moves). M11 attacked v1's actual strength instead — clone fidelity — by
upgrading the MAIN *scorer* from linear to a small MLP, trained on the same data
through the same featurizer, with an offline gate that is nearly free. One variable
changed (the MAIN score function); everything else — take-k rule, fallbacks, other
contexts' linear weights, entrypoint, builder — untouched.

## The MLP scorer (Step 1, gate G1)

`scratchpad/train_mlp_main.py` (dev-only, numpy) reuses `train.py`'s pipeline
(`_featurize`/`_pack`/`_seg_softmax`) with the same conditional-logit objective
(segment-softmax CE over a decision's options); only the per-option score changes from
`w·x` to `w2·relu(W1 x + b1) + b2`. Architecture 386→32→1, 3 seeds ensembled, L2 grid
{1e-4,1e-3}, early stopping.

**Honest evaluation was essential.** The first run reported 0.963 MAIN accuracy — a
red flag. Root cause: selection bias — early-stopping + seed/L2 selection over ~300
checkpoints, all evaluated on the *same* val set that was then reported. Fixed with a
**three-way split by game** (train fit / val early-stop+select / test report-only):

| model | val acc | TEST acc (unbiased) |
|---|---|---|
| linear (v1's scorer, retrained on identical split) | 0.858 | **0.858** |
| MLP ensemble (3×h32) | 0.965 | **0.952** |

**TEST lift +0.093, robust** (val→test gap only 1.4 pts; 3 seeds consistent
0.947–0.955). The MLP genuinely learns the teacher's conjunctive MAIN logic (e.g.
"fire Rocket Rush only if enough TR in play AND opponent low AND energy ready") that a
linear model smears out. Gate G1 (TEST ≥ 0.875): **PASS.**

## Shipped-safe inference (Step 2)

`src/ptcg_ai/imitation/policy.py`: a context scorer is now a flat linear list (v1) OR
an MLP / MLP-ensemble spec dict (v2). `_score` dispatches; `_mlp_forward` is pure
stdlib (sparse accumulate over nonzero features — the featurizer's one-hot blocks are
mostly zero), honoring the numpy-free Kaggle contract. `load_weights` validates both
shapes. `bc_650_v1.json` still loads byte-identically. New `tests/unit/test_mlp_scoring.py`
proves stdlib forward == numpy forward (no train/serve skew); ship-safety test stays
green. End-to-end check: `ImitationPolicy('bc_650_v2.json')` reproduces the trained
TEST accuracy (0.9532) through the live featurize→score→take-k path.

## The gates that matter (Step 3)

- **G2 — mirror arena v1.2 vs v1** (same deck, both plain BC, n=300): **0.493**
  (148W-152L, CI [0.437, 0.550]). A coin-flip — expected, since two clones of the same
  ~47%-win teacher don't beat each other in a mirror; the extra fidelity has no edge to
  express there. Not clearly below 0.5 → passed to G3.
- **G3 — field gauntlet** (120 distinct legal decks, greedy-piloted): v1 macro WR
  **0.901**, v1.2 **0.900**. **Paired delta v1.2−v1 = −0.001, 90% CI [−0.020, +0.017]**,
  better on 39 decks, worse on 29. A statistical dead tie — tighter around zero than any
  prior candidate, and never a regression. **Strict ship bar (mean ≥ 0 AND lo > −0.02)
  narrowly failed** (mean −0.001; lo at exactly −0.020).

**Why the +9.3 offline gain vanished on the gauntlet: ceiling saturation.** Both clones
already crush the *greedy*-piloted field at ~90%, leaving almost no headroom for extra
fidelity to add wins. The M8.1 doc flags this — "a greedy-piloted field is weaker than
the human ladder." The gauntlet was validated to separate a LARGE true gap (v1 0.88 >
v2 0.83, matching the ladder); it lacks the resolution for a SUBTLE one at a 90%
ceiling. The fidelity gain, if real, only manifests against strong opponents the
gauntlet cannot simulate — i.e., the real ladder.

## Decision: shipped as a free-roll

Given (a) v1.2 is a tie, never a regression, on every offline measure, (b) the gauntlet
is provably saturated and can't measure the intended benefit, and (c) Kaggle preserves
your best-ever score so a swap carries minimal downside, the user chose to ship v1.2
and let the real ladder be the decisive test. Built via
`build_imitation.py decks/greengreenpurple.csv data/models/bc_650_v2.json imitation-v1.2`
→ `build/imitation-v1.2.tar.gz` (2.3 MiB). Verified: structural validation + entrypoint
smoke test pass; the bundled weights carry the MAIN `mlp_ensemble`; a Kaggle-style load
(exec `main.py` with no `__file__`) sets `_IMITATION_READY = True` (MLP engages, not a
silent greedy fallback). Compute is safe (mirror arena ran 0.33 s/game with the MLP;
Kaggle budget is 600 s/episode). **Pending user upload; watch the ladder ~50 games
before judging (ratings need that to stabilize).**

## Reusable assets / lesson

- `data/models/bc_650_v2.json` (v1.2 weights, MAIN=MLP ensemble); `train_mlp_main.py`;
  the payload-v2 MLP inference path in `policy.py` (any context can now be an MLP).
- **Lesson: offline clone-fidelity gains are real but the greedy gauntlet can't verify
  a subtle one — it saturates at ~90%.** To measure future subtle improvements we need
  either the real ladder or a *stronger* offline field (opponents piloted by clones, not
  greedy — currently impractical, we only have clones for a few decks). If v1.2 helps on
  the ladder, the same MLP-scorer recipe extends to other contexts / a stronger teacher.

## Status

imitation-v1 remains the proven champion; **imitation-v1.2 is built, verified, and
shipped as a free-roll pending ladder data.** If the ladder shows no gain after ~50
games, revert to v1 (kept). Nothing about v1's artifacts changed (`bc_650_v1.json`
frozen; builder/entrypoint untouched — v1.2 is just a different `bc_weights.json`).
