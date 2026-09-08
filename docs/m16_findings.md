# M16 findings — self-play RL from the BC init: the first lever to move the objective the RIGHT way, but sub-threshold to ship offline (2026-07-17)

**Bottom line: policy-gradient RL fine-tuning of the MLP MAIN scorer (from the v1.2 init),
optimizing the real win/loss objective instead of teacher-imitation, produced a small,
STABLE, directionally-correct improvement. At the n=300 confirm, iter8 beats BOTH champions
on the strong-clone gauntlet — pooled +0.027 vs v1.2 and +0.024 vs v1 (positive on 7/8
opponents) — and generalizes strongly to a held-out opponent never trained against (cinderace
+0.110). This is the FIRST lever in M7–M16 to move the objective in the right direction — the
exact opposite of v1.2, where higher BC fidelity made play worse. BUT the edge is small
(+0.027, below the pre-registered +0.05 ship bar) and M14 established that the gauntlet cannot
resolve a pilot difference this small into a confident ladder prediction. Offline verdict: a
real but sub-ship-threshold edge over both champions. The RL checkpoint (iter8) is a genuinely
NEW policy (optimized for winning, not fidelity) with the best held-out generalization of the
run; the only instrument that can settle whether it truly beats v1 is a ladder free-roll,
which is the recommended next step.**

## Why RL, and how it differs from everything before

BC (M7–M15) has a hard ceiling: the teacher's own skill. M14 showed that raising BC fidelity
(the v1.2 MLP, +9.3 pts offline) made real play *worse* — imitating a 650-elo player more
precisely is not the same as playing better. M16 attacks the objective directly: the agent
samples from its own MAIN policy, plays thousands of games vs strong clone opponents, and
REINFORCE nudges up the decisions from won games / down from lost games — "learning from its
own mistakes" — anchored to the BC weights so cheap optimization can't destroy strong play
(the M10 lesson).

## The machinery (all verified — `scratchpad/rl_selfplay.py`)

- **Sampling policy** (`SamplingImitationPolicy`): samples MAIN from softmax(scores/τ), τ=1.0
  (~15% non-argmax exploration, measured); every other context stays deterministic BC.
- **Collection**: self-play vs a 7-opponent training mix (lucario/m093jp/budew/eduardo/
  tr-mirror/legend/kenn), cinderace HELD OUT for eval only; draws discarded; the `on_decision`
  hook captures MAIN rows in the exact dataset schema (no train/serve skew — verified
  `featurize_decision` == per-option `featurize_option`, parity 300/300).
- **Update**: REINFORCE `g = A·(probs−onehot)/(N·τ)` with advantage `A = win/loss − V(state)`,
  trust-region anchor `λ·(θ−θ_BC)` per ensemble member (λ=0.1), lr 1e-3, 25 epochs/batch.
  Gradient check vs finite-diff: 8.4e-08.
- **Critic**: `v_650.json` (logistic P(win|state), 7 features). **Refreshed at iter5→6** on
  222k self-play states: held-out AUC 0.793→**0.825** (+0.032) → adopted as `v_rl.json`. The
  teacher-trained critic was measurably miscalibrated for our own agent's states.
- **Coded kill criteria** (K1 futility / K2 drift / K3 greedy-forgetting / K4 held-out overfit /
  K5 plumbing / K6 budget). K3's floor was recalibrated from an assumed 0.85 to 0.74 after
  measuring the real baseline on the 3-deck greedy slice (v1.2=0.787, v1=0.81 — NOT the ~0.90
  full field; meta_lucario/meta_dragapult are near-mirrors). None fired across 8 iterations.

## The 8-iteration trend

| iter | collect WR avg (n=2000) | eval pooled vs v1.2 (n=160) | held-out cinderace (base 0.290) | greedy anchor |
|---|---|---|---|---|
| 1 | — | +0.011 | — | — |
| 2 | 0.641 | +0.011 | 0.350 | 0.795 |
| 4 | 0.639 | +0.002 | 0.325 | 0.794 |
| 5 | 0.653 | **+0.027** | 0.331 | 0.756 |
| 6* | 0.649 | +0.022 | 0.362 | 0.761 |
| 8 | 0.634 | +0.020 | **0.431** | 0.822 |

\* critic refresh adopted before iter6.

**Reading:** the eval delta is stable and positive across iters 5/6/8 (+0.027/+0.022/+0.020 —
three independent reads, not one lucky point), but it PLATEAUED — it never reached the +0.04
bar, and the collect-WR telemetry (n=2000, the most reliable signal) is essentially flat
(~0.642 throughout). The policy improves vs easy opponents (budew/eduardo/kenn) at the expense
of the hardest (lucario steadily declines) — net a small positive that does not compound. The
critic refresh did not lift pooled but visibly improved generalization: held-out cinderace
0.362→0.431 and greedy anchor 0.761→0.822 (now above v1.2's own 0.787) over iters 6–8.

## C3 confirmation (iter8, n=300) — the decisive de-noised read

The n=300 confirm HELD and sharpened the signal (SE pooled ~0.014):

| opponent | iter8 | v1.2 base | d(v1.2) | v1 base | d(v1) |
|---|---|---|---|---|---|
| lucario | 0.353 | 0.315 | +0.038 | 0.320 | +0.033 |
| m093jp | 0.677 | 0.590 | +0.087 | 0.640 | +0.037 |
| budew | 0.690 | 0.680 | +0.010 | 0.660 | +0.030 |
| eduardo | 0.692 | 0.700 | −0.008 | 0.683 | +0.009 |
| tr_mirror (vs v1) | 0.530 | 0.547 | −0.017 | 0.500 | +0.030 |
| legend | 0.823 | 0.800 | +0.023 | 0.843 | −0.020 |
| kenn | 0.827 | 0.853 | −0.026 | 0.783 | +0.044 |
| **cinderace (held-out)** | **0.400** | 0.290 | **+0.110** | 0.367 | +0.033 |
| **POOLED** | | | **+0.027** | | **+0.024** |

**iter8 beats BOTH baselines on the pooled gauntlet** (+0.027 vs v1.2 ≈1.9σ, +0.024 vs v1
≈1.7σ), positive on 7/8 opponents vs v1, with a clearly-real held-out result (+0.110,
≈3.9σ). Ship-gate check: pooled +0.027 < the pre-registered **+0.05** bar → **strict offline
ship gate NOT met**; cinderace ≥ −0.02 ✓, probe 0.986 ≥ 0.85 ✓. So: a real, confirmed,
positive edge over both champions — but below the magnitude at which we pre-committed to call
it a confident win, and (per M14) below the gauntlet's resolution to predict the ladder.

## Verdict and the ladder decision

Pre-registered gate: best pooled +0.0266 (iter5) < +0.04 → **no advance to a confident ship**.
This is a no-ship by our own offline criteria. Two things make it a MORE interesting result
than the M8/M9/M15 teacher-search kills:

1. **Direction.** RL moved the objective the right way (+0.02 stable, generalizing), the exact
   opposite of BC-fidelity (v1.2, ladder-confirmed worse). Optimizing "win" works; imitating
   harder does not. This validates the premise even though the magnitude is small.
2. **Resolution, not just magnitude.** M14 proved the gauntlet reads the true v1-vs-v1.2 ladder
   gap (+0.13) as ~0 — so a +0.02 offline signal is *below the instrument's ability to predict
   the ladder at all*. We cannot know offline whether iter8 beats v1. Only the ladder can judge.

The RL checkpoint (`data/models/rl_ckpt_iter8.json`) is a genuinely new policy — optimized for
winning, best held-out generalization of the run. Shipping it as a ladder free-roll is the only
way to resolve whether the +0.02 offline edge is real ladder strength. Cost (per the "2 most
recent submissions play" rule): it takes a live slot, displacing v1-resub (~608, declining).

## Why the plateau: the trust-region anchor creates an equilibrium, not a time limit

Post-hoc analysis of why more epochs/iterations didn't break past ~+0.02-0.027: every update
has two opposing forces — the REINFORCE gradient pushes weights toward "win more," and the
anchor term `λ·(θ−θ_BC)` pulls them back toward the BC init. With λ fixed, there is an
equilibrium point where the two cancel; once the policy reaches it, **more epochs just
converge faster to the SAME point**, they don't move the point itself. This matches the
observed pattern exactly: fast initial movement (iter1: +0.011), then oscillation around
+0.02-0.027 for the remaining 7 iterations with no compounding trend. Epochs control
convergence *speed* to the equilibrium, not its *location* — only λ (or the anchor's design)
moves the equilibrium. A second, structural ceiling compounds this: only the MAIN context is
fine-tuned; TO_HAND and every other context stay frozen at BC quality, capping how much total
win-rate improvement is reachable regardless of how well MAIN alone is optimized.

**Implication for a future attempt:** annealing λ down across iterations (once probe/kill
checks confirm no drift) would shift the equilibrium further from the BC init — the most
direct fix, at the cost of re-inviting the M10 failure mode (need the same kill vigilance).
Extending RL to TO_HAND (the stage-expansion trick from M13/M15 already handles its
multi-pick) would remove the second, structural ceiling. Neither was attempted in M16; both
are candidate designs for a follow-up, not yet built.

## Why not aim for a teacher/agent that doubles v1's elo (a broader honest note)

Some other competitors' submitted agents on the ladder are rated well above v1/v1.2 (some
apparently >1000) — proof it's achievable *in this game*, just not via the specific lineage
tried here (BC of accessible human replays + a small anchored RL fine-tune). The likely
gap: those results probably come from real domain expertise about the game's decks/cards
directly encoded into heuristics/search (the exact wall M6 identified — piloting
sophistication, not deck or algorithm), and/or search implemented with a much better
evaluator than the ones we hand-tuned ([decision/evaluator.py](../src/ptcg_ai/decision/evaluator.py)),
and/or more elapsed calendar time. Neither this session's assistant nor the user has deep
TCG domain expertise (flagged as a known constraint since project start, §8 of
[handoff.md](handoff.md)) — this, not a ceiling in the game itself, is the actual bottleneck
for the heuristics/search axis. **New direction (2026-07-17, ongoing): sourcing real
strategy guides/archetype-specific content for the decks already in play (Bellibolt ex,
Lucario ex, Team Rocket swarm, etc.) to translate into concrete `wants_fn`/`damage_fn`/
sequencing-rule knowledge** — the same slots already hand-authored per `DeckProfile` in
[deck_profiles.py](../src/ptcg_ai/imitation/deck_profiles.py), and into `decision/evaluator.py`'s
hand-tuned V for a future, better-informed search attempt. Not yet started; user is
gathering source material.

## What remains open (updates m12_findings.md)

RL is now BUILT and reusable (`scratchpad/rl_selfplay.py`, full loop + kills + critic refresh).
The lever is not exhausted the way BC/teacher-search are — a stronger version (larger batches,
more opponents, longer horizon, or a value-baseline with richer features) could plausibly push
past the +0.02 plateau. But that is a multi-day investment against a marginal-so-far return, and
still unverifiable offline. The other open options (ladder free-rolls, deck-level change) are
unchanged.

## Files

- Built: `scratchpad/rl_selfplay.py` (sampler, collector, REINFORCE update, mini-eval, iterate
  loop with 6 coded kills, critic refresh). `data/models/rl_ckpt_iter{2..8}.json`,
  `data/models/v_rl.json`, `data/rl/loop_manifest.json` + per-iter trajectories/manifests.
- Unchanged: `src/` (sampler is a scratchpad subclass), `bc_650_v1/v2.json`, `v_650.json`
  (refresh wrote a NEW file), vendored SDK.
