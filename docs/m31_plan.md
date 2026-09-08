# M31 plan — applying top-solution techniques from other Kaggle sim competitions (2026-07-27)

Synthesis of two external solution corpora the user supplied — the **FIDE & Google Efficient Chess
AI Challenge** (top-5 writeups) and **NeurIPS 2024 Lux AI Season 3** (top-10 writeups) — turned into
a concrete, gated experiment plan for the Yushin/ALAKAZAM clone.

**Why Lux AI is the load-bearing source:** it is a Kaggle *simulation* competition with the same
shape as ours, and **5 of its top 10 solutions were imitation learning from the top players' replays**
(3rd, 4th, 7th, 8th, 9th) — exactly what we do. The chess corpus is an *engine* competition (search +
NNUE eval under a 64 KB limit); only its meta-methodology transfers.

---

## 0-ter. TRACK E — **CLOSED NEGATIVE at the head_to_head veto (2026-07-28)**

The V2 clone was built end-to-end and vetoed **before** spending a ladder slot:

- Non-MAIN contexts retrained locally at dim 1269 (`bc_alakazam_v2_full.json`). Independent
  confirmation the features carry signal: **linear MAIN 0.5975 → 0.629 (+0.031)** on the identical
  split — features help a weak model much more than the MLP (+0.009), exactly as expected.
- GPU `--emit` produced a 10-member ensemble; measured locally that **accuracy saturates at 3
  members** (k=3 0.7822 vs k=10 0.7819), so it was trimmed → 3.3× faster inference, 13.2 MB → 4.0 MB.
- Bundle `imitation-yushin-v2.tar.gz` (3.7 MiB) builds, smoke-tests, resolves the imitation tier at
  dim 1269, 0 SafePolicy failures, ~34 ms/decision (~2 s/game vs a 600 s budget). 202 tests pass.
- **`head_to_head` (same deck on both sides = pure policy comparison), field 120, n=6:**
  **yushin-v2 0.926 vs yushin-v1 0.960 → paired delta −0.033, 90%CI [−0.056, −0.011], 19/39 decks.
  CI entirely negative — V2 PLAYS WORSE.**

**Verdict: higher offline fidelity again produced worse play** (V2 wins MAIN accuracy by +0.009 yet
loses the gauntlet by −0.033), consistent with the growing train−test gap (0.034 → 0.040–0.045). This
is the **sixth** instance of the project's iron law (M11/M14/M22/M25/M27/M31). **Do not ship V2; the
V1 `imitation-mlp` champion stays.** Caveat for the record: the comparison is not a clean feature
isolation — the V1 champion was trained on 1000 episodes on CPU (M28) while V2 used 1284 on GPU — but
the practical question ("does the candidate beat the champion?") is answered unambiguously.

**M31 as a whole is now closed: all four levers (data volume, capacity, data weighting, feature
enrichment) measured negative or unshippable.** The remaining honest options are a change of regime,
not further tuning of this clone.

## 0-bis. TRACK E RESULTS (2026-07-27, GPU, `GPU_Trials.ipynb`)

`ALAKAZAM_V2` (dim 1269, snapshot 29→76, four contiguous ablatable groups). All runs h=48, 10 seeds,
alpha=1.0, pinned split. Baseline = V1 `ALAKAZAM` (dim 658) 10-seed run.

| run | groups | per-seed mean ± SE | std | ensemble TEST | gap | TEST-won |
|---|---|---|---|---|---|---|
| V1 baseline | — | 0.7721 ± 0.0039 | 0.0124 | 0.7831 | 0.034 | 0.7370 |
| E1 | G1 hand-composition | 0.7675 ± 0.0032 | 0.0102 | 0.7779 | 0.028 | 0.7323 |
| E2 | G2 binned/dual | 0.7755 ± 0.0007 | 0.0021 | 0.7860 | 0.043 | 0.7422 |
| E3 | G3 differentials | 0.7722 ± 0.0019 | 0.0059 | 0.7826 | 0.034 | 0.7365 |
| E4 | G4 composites | 0.7706 ± 0.0020 | 0.0064 | 0.7794 | 0.030 | 0.7345 |
| **E5** | **all** | **0.7797 ± 0.0014** | 0.0044 | **0.7900** | 0.045 | **0.7485** |

**Findings.**
1. **E5 is the best configuration measured**, winning on all three metrics at once: per-seed mean
   **+0.0076**, ensemble **+0.0069**, **TEST-won +0.0115** (the metric aligned with imitating the
   teacher's *winning* play).
2. **Richer features sharply stabilise training** — per-seed std collapses 0.0124 → 0.0044 (E5) /
   0.0021 (E2). Unexpected and valuable: far less risk of drawing a bad seed for the shipped model.
3. **The headline hypothesis FAILED.** G1 (per-card hand composition — the "two 7-card hands" fix)
   is the **worst** group in isolation (−0.0046, below baseline). The winner was G2, the
   dual/binned encoding, i.e. **the external corpus's advice beat our own reasoning.**
4. **Superadditivity:** G1/G3/G4 are neutral-to-negative alone, yet E5 (all) > E2 (best single).
   The hand features apparently only pay off in combination.
5. **HONEST CAVEAT — E5 does NOT clear the pre-registered gate** (≥0.010 on per-seed mean; it gives
   +0.0076). Welch's t vs the V1 baseline: **t≈1.84, p≈0.09** — suggestive, not conclusive. The V1
   baseline's variance is inflated by one outlier seed (0.7393), which is part of why.
6. **The E0 control was never run.** Comparing V2 runs to the *V1* run conflates "more features"
   with "different dim / different init draw". E0 (V2 profile, all groups masked) is the clean
   same-shape control and costs ~4 min — run it before treating E5's margin as established.

## 0. GPU migration — validated (2026-07-27)

`scratchpad/colab_train_mlp.py` (PyTorch port) confirmed functionally correct on Colab (T4): split
sizes match the local run bit-for-bit, and 10-seed distributions are consistent in shape/magnitude
with the local runs. Exact CPU/GPU numeric parity was NOT achieved (float64-CPU vs float32-GPU
precision differences cause chaotic-optimization divergence at the per-seed level; one seed landed as
a clear low outlier — 0.739 vs a ~0.77-0.78 cluster — confirming the process has real high-variance
seeds, not just fine-grained noise). **Working decision: stop chasing exact CPU/GPU parity; use GPU
runs for internal (GPU-vs-GPU) comparisons only, always at a fixed seed count (10).** Full 10-seed
ensemble run: **~170-190s** (vs ~4 h locally) — a ~80-90x speedup, which is what makes Tracks E/B/C
practical to run this session.

**New GPU baseline (h=48, 10 seeds, alpha=1.0, all data): TEST 0.7831, gap +0.034, TEST-won 0.7370.**
All subsequent GPU experiments compare against THIS number, not the old 0.7685 CPU figure.

## 1. Where we are (measured, this session)

- **Champion candidate:** `imitation-mlp` (ALAKAZAM MLP clone of Yushin #1) — ladder **889 vs 845.5**
  for a concurrently-uploaded kanga baseline (>50 eps each) = **+43.5, the first top-teacher clone to
  lead kanga on the real judge** ([m28_findings.md](m28_findings.md)).
- **Data bank:** `replays/54773249` = **1284 episodes** (API exposes a rolling most-recent-1000
  window; re-pulling *accumulates* on disk). Dataset = **108,989 decisions / 66,162 MAIN**.
- **Capacity ablation (3-way split, identical data, this session):**

  | model | TEST | train−test gap |
  |---|---|---|
  | linear | 0.5377 | +0.054 |
  | MLP h=48 | 0.7685 | +0.018 |
  | MLP h=96 | 0.7741 | +0.026 |

- **THE DIAGNOSIS IS NOW COMPLETE — by elimination, features are the bottleneck:**
  1. **More data did NOT help.** 1000 → 1284 episodes left TEST fidelity flat (0.780 → 0.7685,
     different splits). Data volume is saturated; "keep downloading" has hit diminishing returns.
  2. **More capacity did NOT help.** h=48 → h=96 moved TEST by **+0.0056**, which is *within the
     seed spread* of that very run (individual seeds 0.7698 / 0.7691 / 0.7754), while the overfit
     gap grew ~50% (0.018 → 0.026). Classic diminishing returns — **h=128 is not worth 4 h.**
  3. Therefore the remaining lever is **feature expressiveness** (Track E) — which is precisely what
     BOTH external corpora name as their #1 lesson — or the teacher's irreducible decision entropy.
- **Data profile (new):** Yushin wins 59.9% of games; **47.4% of our MAIN training decisions come
  from games he LOST**; only 1.7% of MAIN decisions have a single option (37.5% have 12+).

## 2. What the external corpora actually recommend

| Technique | Source | Applies to us? |
|---|---|---|
| **Two-stage training: pretrain on large/older data → fine-tune on the best current policy** | Lux **4th, 7th, 9th** (independently); Chess 1st (weak→strong data staging) | **YES — highest-confidence import** |
| **Policy/Agent-ID conditioning** (train on many policies, condition inference on the best one) | Lux **8th** | **YES — third way past our pooling dilemma** |
| **Data selection: keep winning games, drop decided/garbage states, drop trivial no-ops** | Lux **3rd, 4th, 8th, 9th** (unanimous); Chess 1st ("data selection is the most important aspect") | **YES — nearly free, unused by us** |
| **Feature enrichment: relative features + continuous *and* binned encodings; light history** | Lux 1st (`simple115` absolute "lacked generalization"), Lux 2nd, Lux 4th (prev-frame stacking helped; long LSTM hurt) | **YES — targets our real bottleneck** |
| **Statistical rigor in local eval (p-values, 100–200 games, paired)** | Lux **9th**; Chess SPRT (20–38M games to resolve ~4 elo) | **YES — methodology** |
| Model capacity | Chess: tiny nets win with good data; Lux RL: 1.8M→3.2M helped | Partially — ablation in flight |
| **Pooling different policies naively** | Lux **9th**: *"mixing data from different policies did not significantly improve performance"*; Lux 4th same | **Confirms our M24/M30 negative — do NOT do this** |
| **RL from an IL-pretrained policy** | Lux **4th**: *"complete collapse of the IL policy or no improvement"* after 2–3 weeks | **Confirms our M16–M19 closure — stays closed** |
| Spatial nets (U-Net/ConvLSTM/Transformer), TTA by rotation/mirror, map-symmetry augmentation | Lux (all) | **NO** — our decisions are option lists, no spatial structure |
| Binary-size/RAM optimization, alpha-beta search internals, quantization, pondering | Chess (all) | **NO** — we use 2.9 of 197 MB; search closed in M10/M17/M18 |

---

## 3. The experiment tracks

**Hard cost note:** one full MLP ablation run ≈ **4 hours** on this machine. That is the binding
constraint, so tracks are designed to triage with the **linear** model (minutes) wherever the
comparison is valid, and spend MLP runs only on finalists.

### Track A — Data selection & weighting — **RESULT: A2 (`alpha=0.5`) WINS**

Linear triage (`scratchpad/track_a_data_selection.py`, fixed 2-way split, fixed L2=1e-4, one shared
featurisation; ~4 h wall because each fit ran ~50 min under contention):

| variant | n_train | TEST-all | TEST-won | Δ vs A0 |
|---|---|---|---|---|
| A0 baseline | 52324 | 0.5975 | 0.5675 | — |
| A1 won-games only (`alpha=0`) | 27947 | 0.5835 | 0.5786 | −0.014 / +0.011 |
| **A2 `alpha=0.5`** | 52324 | **0.6253** | **0.5771** | **+0.028 / +0.010** |
| A3 A2 + drop `n_options<=2` | 49605 | 0.6232 | 0.5755 | +0.026 / +0.008 |
| A4 A2 + flatten turn≥20 | 47401 | 0.5653 | 0.5780 | **−0.032** / +0.011 |

- **A2 improves BOTH held-out metrics** — not a trade-off, strictly better. Likely mechanism:
  losing-game decisions are noisier (desperate/decided positions), so halving their weight cuts label
  noise and improves generalisation *even on the unfiltered set*.
- **Soft weighting beats hard filtering** (A2 > A1). The Lux teams filtered hard, but they had
  8k–20k episodes; with our 1284 it pays to keep the information at half weight instead of discarding.
- **Dropping trivial decisions doesn't help** (A3 < A2) — only 4.8% of our MAIN rows have ≤2 options,
  versus Lux where 95% were no-ops. Their problem is not our problem.
- **Flattening the turn≥20 spike is clearly harmful** (−0.032). Those long games are *signal*, not an
  artefact — the chess-1st "flatten the distribution" trick does **not** transfer.
- **CONFIRMED ON GPU/MLP (2026-07-27) — DOES NOT TRANSFER, TRACK A CLOSED.** GPU baseline (h=48,
  10 seeds, alpha=1.0): TEST 0.7831, gap 0.034, TEST-won 0.7370. With `alpha=0.5`: TEST **0.7796**
  (−0.0035), TEST-won **0.7359** (−0.0011), gap shrinks to 0.022. Flat-to-slightly-negative on both
  accuracy metrics — the linear-model win does NOT reproduce on the model that ships. Mechanism: a
  low-capacity linear model benefits from cleaner (less noisy) signal even at the cost of volume; an
  MLP with enough capacity already separates signal from noise in losing-game decisions, so
  down-weighting them only removes information. **Do not adopt `alpha=0.5`; do not spend a
  head_to_head/ladder slot on it** — another instance of "never trust the proxy without confirming on
  the real model" (same lesson as M11/M14, this time for data weighting instead of capacity).

### Track A (original design notes)

Reuses `train.py`'s existing but **never-used** `--alpha` (down-weights decisions from lost games:
`weight = 1.0 if won else alpha`) plus new row filters.

| Variant | Definition | Rationale |
|---|---|---|
| A0 | all data (baseline) | current |
| A1 | `--alpha 0` = **won-games only** | Lux 3rd/8th/9th trained only on winning games |
| A2 | `--alpha 0.5` (soft) | keeps losing-game signal at half weight |
| A3 | A2 + drop `n_options <= 2` | Lux 3rd dropped 95% of trivial no-ops (ours: only ~4.8% of rows) |
| A4 | A2 + cap the turn≥20 bucket (19.2% of rows) | Chess 1st "flatten the distribution"; long games over-represented |

**Gate (important subtlety):** filtering the *training* target changes what "fidelity" means, so
**held-out fidelity is NOT a valid gate for Track A** — a variant that ignores losing play will score
lower on an unfiltered held-out set by construction. Therefore: report fidelity on **both** a fixed
unfiltered held-out **and** a won-games-only held-out (matched target), and gate the finalist on
**`head_to_head.py`**, not on fidelity. Train/val/test split stays fixed and identical across variants.

### Track B — Two-stage fine-tuning (the highest-confidence import)

Rescues the old submission **54486275** that M30 correctly rejected *for pooling* (same deck, 58/60
cards, but a different policy: our clone scores 0.598 on it vs 0.78 within-policy). Lux 4th/7th/9th
all pretrain on the larger/older corpus and then fine-tune on the current best submission — which
gets the volume without permanent dilution, because the final gradient steps pull the model onto the
current policy.

1. Download the rest of `54486275` (we have 60 of ~1000).
2. Add **warm-start** to the trainers (`_train_mlp` / `_train_single` accept initial parameters) —
   `scratchpad/` only, no `src/` change.
3. Stage 1: train on 54486275. Stage 2: continue from those weights on 54773249 (1284 eps), lower LR.
4. **Gate: TEST MAIN fidelity on the *current-policy* held-out vs the 0.7685 baseline.** The target is
   unchanged here, so fidelity **is** a valid gate for this track.

### Track C — Policy-ID conditioning (alternative use of the same old data)

Lux 8th trained on 10 different agents and conditioned the model on an agent ID, then **conditioned
inference on the best agent**. Our shippable version avoids all runtime plumbing:

- Add an additive profile `ALAKAZAM_V2` = ALAKAZAM snapshot **+ one extra slot** ("is current
  policy"), constant **1.0** in the shipped `snapshot_fn`.
- Offline, when building the pooled training matrix, overwrite that column to **0.0** for
  old-submission rows.
- At inference the flag is always 1.0 ⇒ the model is permanently conditioned to imitate the *current*
  Yushin, while still having learned shared structure from both corpora.
- **Gate: fidelity on the current-policy held-out** (valid — target unchanged). Compare against both
  A0 baseline and Track B.

### Track D — Capacity — **CLOSED (negative)**

h=48 → h=96 gained **+0.0056 TEST, inside the run's own seed spread**, while the overfit gap grew
0.018 → 0.026. Capacity is not the bottleneck; **h=128 cancelled** (saves 4 h). This matches the
chess counterweight: a 2,489-parameter net beat decades of hand-crafted eval — features/data
dominate, not parameter count. Keep **h=48** as the shipping architecture (equal accuracy, smaller
payload, smaller gap).

### Track E — Feature enrichment (targets the measured bottleneck)

Both corpora put feature engineering above architecture. Our snapshot is 29 mostly-absolute scalars.
Concrete, cheap additions (new additive profile, retrain, `pytest`):

- **Relative/differential features** — Lux 1st found absolute-only features "lacked generalization":
  e.g. `hand_size − opp_hand_size`, energy/board differentials, prize differential is already there.
- **Dual encoding (continuous + binned one-hot)** — Lux 1st encoded *each* feature both ways. This is
  the single highest-leverage trick for a shallow model: it lets the scorer learn **non-monotonic**
  responses to e.g. `hand_size` (Powerful Hand damage = 20×hand, but holding cards has a cost).
- **Light history** — Lux 4th: stacking the *previous frame* helped, while LSTM/Conv3D "deteriorated
  performance significantly". So add at most previous-turn deltas, not sequences.
- **Gate: fidelity on the fixed held-out** (valid — target unchanged), then `head_to_head`.

---

## 4. Sequencing (by expected value per compute-hour)

Our own diagnostics say **data volume is saturated** but the model is **not overfitting** — so the
bottleneck is *feature expressiveness / capacity / data quality*, not quantity. That ranks the tracks:

1. **Track A** (hours: ~0.5 linear triage) — free, unanimous in Lux, and 47% of our rows are from lost
   games. Run the linear triage of A1–A4 first.
2. **Track E** (hours: ~1 build + 4 MLP) — attacks the measured bottleneck; strongest cross-corpus
   endorsement.
3. **Track D** (in flight) — h=96 result decides whether h=128 is worth 4 more hours.
4. **Track B**, then **C** (hours: download + 2×4 MLP) — highest-confidence *import*, but tempered by
   our own finding that extra data didn't move TEST; the old policy is also weaker. Run after A/E.

**Combination rule:** the winners of A, D, E are largely orthogonal (data quality × capacity ×
features) and should be **stacked** into one final candidate, which then goes to `head_to_head` and
the ladder.

## 5. Evaluation discipline (non-negotiable, from our scars + both corpora)

- **Offline fidelity is a proxy, never the verdict.** Lux 7th: *"validation error served only as an
  indicator — not an accurate metric for our objective."* Our M11/M14 says the same.
- **`head_to_head.py` is a veto, not a ranker** (M22/M25) — a combo deck crushing a greedy field is a
  known false positive (ITF beat the gauntlet, lost the ladder).
- **Ladder judging follows M29:** a single converged score carries **~±100 elo** of opponent-field
  noise. Upload the challenger **concurrently** with a baseline and judge the **gap** after ~50 eps.
  Adopt Lux 9th's practice of reporting a **p-value** alongside the win rate.
- **Never re-roll an agent hoping for a lucky field** — that optimizes noise (M29).

## 6. Explicitly out of scope

Naive pooling of different policies (M24/M30 + Lux 9th/4th), RL from the IL policy (M16–M19 + Lux
4th), search (M10/M17/M18), spatial architectures/TTA (no spatial structure), and size/RAM/quantization
work (irrelevant at 2.9 MB of a 197 MB budget).

## 7. Files

New tooling will live in `scratchpad/` (dev-only). The only anticipated `src/` changes are **additive
DeckProfiles** (`ALAKAZAM_V2` for Tracks C/E), which require `pytest` green and a retrain. Nothing
ships or is uploaded until a gate passes and the user approves.
