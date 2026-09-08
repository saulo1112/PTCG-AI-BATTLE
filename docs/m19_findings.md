# M19 findings — the gradual λ-anneal does NOT break the M16 plateau: fifth confirmation, compute excuse removed (2026-07-20)

**Bottom line: M16 diagnosed the self-play RL plateau (+0.02-0.027) as a trust-region equilibrium
— only lowering λ moves it. M17 tried a fast, calendar-fixed anneal (0.1→0.03) and drifted into a
K2 kill. M19 isolated ONE variable — the λ schedule — and tested the hypothesis that a MUCH more
gradual, CONDITIONAL anneal (descend a level only when the K2/K3 kill floors are cleared with
margin; HOLD and re-stabilize otherwise) could loosen the anchor without drift. It does NOT. Run
locally with no compute limit (GPU confirmed irrelevant for this native-engine-bound loop), from
the exact M16 init (v1.2) and critic (v_rl.json base-7 — NOT v_rl_v2), the run reached +0.027 at
λ=0.10 immediately (= M16 iter8's ceiling), never exceeded it as λ descended (+0.011 at 0.085,
+0.018 at 0.07), and when the novel HOLD mechanism engaged it did NOT stabilize — one more
iteration at λ=0.07 dropped bc_acc 0.937→0.904 into a K2 drift kill. No checkpoint cleared the
pre-registered +0.04 filter, so the n=300 confirm was correctly skipped. This is the FIFTH
independent confirmation (M11, M14, M16, M17, M19) that this class of refinement does not move the
~+0.027 offline ceiling / ~686 ladder — and the first with the compute variable explicitly removed
as an excuse. imitation-v1 (linear, ~686) remains the champion.**

Isolated-variable experiment per the user's non-negotiable rule: only the λ schedule changed vs
M16; init, critic, batch, epochs, tau, networks, and all six kills identical.

## The run (iters 30-33, local, eval every iter)

Init `bc_650_v2` (v1.2), critic `data/models/v_rl.json` (base-7, the M16-refreshed critic),
batch=2000, epochs=25, tau0=1.0, eval_n=160. Schedule `[0.10, 0.085, 0.07, 0.055, 0.04, 0.03,
0.02]`, descend-on-margin (bc_acc≥0.94, agree≥0.87, anchor≥0.76), `lam_max_holds=2`.

| iter | λ | collect WR avg | pooled d(v1.2) | d(v1) | cinderace | greedy_anchor | bc_acc | decision |
|---|---|---|---|---|---|---|---|---|
| 30 | 0.100 | ~0.63 | **+0.027** | +0.025 | 0.281 | 0.800 | 0.952 | margin clear → 0.085 |
| 31 | 0.085 | ~0.65 | +0.011 | +0.008 | 0.312 | 0.800 | 0.948 | margin clear → 0.07 |
| 32 | 0.070 | ~0.64 | +0.018 | +0.016 | 0.362 | 0.767 | 0.937 | HOLD 1/2 (margin not met) |
| 33 | 0.070 | ~0.64 | — | — | — | — | **0.904** | **KILL K2 (drift, bc_acc<0.93)** |

## Diagnosis — the anchor is not a speed limit; the intermediate region has no better stable basin

M16's mechanism is confirmed and sharpened. Lowering λ *does* move the equilibrium off the BC init,
but the policy loses BC fidelity (drifts) **faster than it gains win-rate**: pooled never rose as λ
fell (it oscillated +0.027 / +0.011 / +0.018, all within the M16 plateau band), while bc_acc fell
monotonically (0.952 → 0.948 → 0.937 → 0.904). The novel M19 contribution — HOLD at the current λ to
let the policy re-stabilize before loosening further — was directly tested at iter33 and **failed to
stabilize**: one extra iteration at the same λ=0.07 made bc_acc *worse* (0.937→0.904), not better,
tripping K2. So the region between "anchored at BC" (high fidelity, +0.02 ceiling) and "drifted away"
(low fidelity, degrading play) contains no better *stable* equilibrium for this MAIN-only REINFORCE
setup; gradualness and holds only change the path, not the destination. This is consistent with M16's
second, structural ceiling (MAIN-only fine-tuning) capping reachable win-rate regardless of the anchor.

Note iter30 hit +0.027 (= M16 iter8) on the very first update, because M19 used the good v_rl critic
from iteration one (M16 only refreshed to it at iter6). So M19 started at the M16 ceiling and the
anneal could not climb above it — the cleanest possible demonstration that the ceiling is the anchor
equilibrium, not the number of iterations or the critic.

## Verdict (against the pre-registered criterion)

- **Filter:** no checkpoint reached pooled d(v1.2) ≥ +0.04 (best +0.027). → **n=300 confirm SKIPPED**
  (nothing to confirm), exactly as pre-registered.
- **Outcome:** FIFTH confirmation of the plateau pattern. Per the pre-registered plan, this line is
  **closed with confidence, NOT reopened with "another variation."** Critically, this run **removed the
  compute variable as an excuse**: it ran locally with no artificial limit, and the Explore profiling
  confirmed the loop is native-engine (CPU) bound — a GPU/Colab would not have changed anything. "We
  just needed more compute" is no longer an available explanation.
- imitation-v1 (linear, ~686) remains the proven champion and the maximum verifiable result.

## What this leaves (for the user to weigh — do not auto-pursue)

The one RL lever never attempted is **extending REINFORCE beyond the MAIN context to TO_HAND** (M16's
named second, structural ceiling). It is a different variable from anything M16-M19 touched (which all
manipulated the MAIN-only optimization), so it is not "another λ variation." But its EV is low given
five straight confirmations, and it is a multi-day build. Flagged, not recommended without a decision.

## Reusable / files

- `scratchpad/rl_selfplay.py`: `iterate()` now supports a conditional-gradual λ schedule
  (`--lam-levels`, `--lam-max-holds`) alongside M17's linear anneal — backward compatible, off unless
  `--lam-levels` is passed. Kills K1 (hold futility) / K6 (iter cap) now explicit in this mode.
- Artifacts: `data/models/rl_ckpt_iter30-33.json`, `data/rl/m19_anneal.log`, manifest entries 30-33.
- Plumbing note: `λ` (U+03BB) in `print()` crashes on the Windows console (cp1252) when stdout is
  redirected — kept all telemetry ASCII (`lam=`), caught by a 7-min tiny-batch plumbing test before the
  real run (would have killed the run at the end of iter30).
