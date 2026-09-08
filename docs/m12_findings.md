# M12–M14 findings — the "improve the pilot / verify offline" tactic, CLOSED (2026-07-13)

**Bottom line: after diagnosing imitation-v1.2 over its full 76-game ladder run and
building a rigorous strong-opponent evaluation harness, the tactic of "beat v1 with a
better-verified MAIN scorer" is closed. Two decisive, evidence-backed results: (1)
imitation-v1.2 (the MLP MAIN scorer) is genuinely WORSE than v1 on the ladder — its
rating converged to ~597 vs v1's ~686, and in the same opponent band (500–700) it wins
0.492 vs v1's 0.632; the MLP's +9.3-pt offline fidelity gain made play worse, not
better. (2) No offline clone-vs-clone gauntlet we can build resolves the true ladder
gap — a calibration run with known ground truth (v1 − v1.2 ≈ +0.13) reads +0.003, a
coin flip. Offline verification of a pilot improvement is therefore impossible with our
instruments; the ladder is the only judge. imitation-v1 (linear, ~686) remains the
proven champion and the maximum verifiable result.**

This supersedes the M11 "shipped as a free-roll, pending ladder verdict" state — the
ladder has now returned its verdict.

---

## M12 — diagnose v1.2 and stress-test the OOD-failure hypothesis

M11 shipped v1.2 on faith (the greedy gauntlet saturates, [m11_findings.md](m11_findings.md)).
M12 asked: is v1.2's low early rating an infra bug, an out-of-distribution (OOD) blunder
pattern, or small-sample noise? Four cuts (`scratchpad/diagnose_v1.py` generalized to a
folder arg; `scratchpad/diagnose_v12_parity.py`):

- **Live-parity: 100.0%** (1667/1667 MAIN decisions at 46 games; 2700/2700 at 76). Replaying
  every logged ladder MAIN decision through the local `ImitationPolicy('bc_650_v2.json')`
  reproduces the logged action exactly. The MLP engaged and was deterministic — **no infra
  bug.** v1.2 is genuinely the MLP agent.
- **Lethal discipline: perfect** (218/218 KOs taken, 0 missed, 0 in lost games) — v1.2 keeps
  v1's discipline.
- **MLP-vs-linear divergence: benign.** The two scorers disagree on 7.6% of ladder MAIN states
  vs 6.6% on held-out teacher states — a 1.16× ratio, below the 1.3× OOD-risk threshold. The
  MLP does not diverge from the linear model materially more on ladder opponents than on
  training states.
- **Leave-archetype-out CV (`scratchpad/train_residual_main.py`).** Cluster the teacher's games
  by opponent archetype (Lucario/Cinderace/Alakazam/Archaludon/other); hold one out entirely,
  train on the rest. The **pure MLP is the OOD-best** (mean held-out accuracy 0.9524 vs linear
  0.8732), winning every fold. A "residual" safety-net architecture (linear backbone + bounded
  MLP correction) was built and **failed its gate G1-OOD** (0.9420, −0.0104 vs the pure MLP):
  the pure MLP is already OOD-robust, so bounding it only costs accuracy.

**M12 verdict at the time:** all three instruments said v1.2 was healthy and its architecture
generalizes; the low rating looked like small-sample/schedule noise. This later proved to be an
instrument failure, not a true reading — see M14.

## M13 — a strong-opponent harness + a paused MLP-extension track

**Strong gauntlet (`scratchpad/strong_gauntlet.py`).** The greedy gauntlet saturates (~90%),
so M13 built a paired harness vs CLONE-piloted opponents (competitive matchups, not 0.9). Fixes
over the M12 draft: the mirror row (candidate vs v1-piloted TR) is the direct head-to-head vs
v1, with the v1 baseline = 0.5 by construction; pooled two-proportion SE over matchups; a
post-hoc saturation guard; ASCII-only output (Windows cp1252 can't encode `⇒`/`—`).

**Phase A result (4 hand-authored clone foes, n=300):** pooled delta v1.2 − v1 = **+0.019**,
95% CI [−0.018, +0.056], sign 2/4 — inconclusive but positive-leaning. Head-to-head (mirror)
v1.2 beat v1 **0.590** (delta +0.090, significant). Read at the time as mild support for v1.2.

**Phase B (TO_HAND MLP + 5-seed MAIN) was built but PAUSED and never run as a ship track.**
Scripts exist and compile (`scratchpad/train_mlp_v4.py` with the Plackett-Luce stage-expansion
trick for multi-pick TO_HAND; `scratchpad/cv_archetype_tohand.py`). It was paused because a
critical re-analysis of v1.2's now-larger ladder sample (below) undercut its premise, and its
gates depend on the very instrument M14 then falsified.

## M14 — the ladder verdict + the falsified instrument (the decisive milestone)

**1. v1.2's 76-game ladder record, read via the Kaggle episode API (`tools/kaggle_api.py`,
`list_episodes` with `initialScore`/`updatedScore`):**

| | v1.2 (sub 54555926) | v1 (sub 54461489) |
|---|---|---|
| games | 76 | 91 |
| rating trajectory | 702 → **converged ~597** (plateau 45+ games) | ~686, stable from game 1, never < 660 |
| record | 38W-37L (0.507), last-30 = 12W-17L | steady |
| WR vs opponents in 500–700 band | **0.492** (32/65) | **0.632** (36/57) |

Both ratings are converged (not small-sample). Since v1.2 changes ONLY the MAIN scorer (same
deck, same 4 linear non-MAIN contexts), **the MLP is the cause. v1.2 is genuinely ~90 elo worse
than v1.** The +9.3-pt offline accuracy gain is fidelity to a 650-elo teacher — imitating a
suboptimal player *more precisely* means playing more like a 650 player, not better. The linear
model's deviations from the teacher were regression toward simpler, more robust play. **Higher
BC fidelity ≠ better play when the teacher is suboptimal.**

**2. The calibration test — is the strong gauntlet a valid ship gate?** With ground truth now
known (v1 − v1.2 ≈ +0.13 expected score), the gauntlet was expanded to 8 diverse clone foes
(the 4 hand-authored + 4 auto-built "screen" clones of fresh teachers: budew,
eduardorochadeandrade, legendbrothers, m093jp — registered in-process via
`deck_profiles.build_generic_profile`, the `quick_screen.py` pattern) and re-run at n=300
(4,500 games, `scratchpad/results/m14_calibration.txt`):

```
POOLED delta (v1.2 - v1) over 8 matchups: -0.003   95% CI [-0.028, +0.023]
sign consistency: 4/8 matchups favor v1.2
```

The instrument reads the ~0.13 gap as **−0.003 (a coin flip)** — it underestimates the true
effect ~40× and can't get the sign right. **Falsified.** Expanding 4→8 diverse opponents moved
the pooled estimate the *wrong* way (+0.019 → −0.003).

**Why no clone-vs-clone gauntlet can work (structural, not fixable by more opponents):** v1 and
v1.2 pilot the same deck and differ only in the MAIN scorer; they agree on 92% of decisions.
Against any *fixed* opponent their win rates are near-identical. The 0.13 gap emerges from the
aggregate over the ladder's large, diverse, adaptive field — not from any single matchup. Paired
evaluation over a handful of correlated matchups averages that signal to zero. This generalizes
M11's saturation finding: it is not just that the greedy field is weak; **offline paired
evaluation fundamentally cannot resolve a sub-0.15 pilot difference.**

## Consequences (why the whole tactic closes, not just this candidate)

- **Offline verification of a pilot improvement is impossible with our instruments.** Any
  candidate — a better scorer, a TO_HAND MLP (Phase B), a self-play-RL fine-tune (the unbuilt
  Phase C) — could only be validated on the ladder, at ~50 games each. Shipping on an offline
  gate is exactly what produced v1.2's regression.
- **The specific MLP lever hurts.** Phase B doubles down on MLP scorers; the evidence says that
  is a bad bet, and unverifiable regardless.
- **imitation-v1 (linear, ~686) is the maximum verifiable result** and remains champion. The
  prior findings still hold and compound: clonability > teacher elo (M8/M9); cheap search
  degrades strong play (M6/M10); v1's edge is an exceptional clone-lift on a simple deck, not
  replicable on demand (no clonable strong teacher exists — 3 human teachers + a screen of
  1000+-elo bots all failed the clonability bar).

## What remains open (for a future session to weigh with the user)

1. **Ladder-as-instrument free-rolls.** Kaggle keeps your best score, so shipping a candidate
   costs only ladder time, no downside to the standing ~686. Viable ONLY for bets with a strong
   independent prior AND that are *genuinely different* from v1's play (not another
   fidelity-to-650 tweak, which we now know regresses). Slow (~50 games ≈ 1–2 days each) and a
   poor track record.
2. **A different, clonable strong teacher** — **update (M15, 2026-07-17): now exhausted for every
   candidate screened so far.** The closest candidate, m093jp (generic-profile MAIN 0.732), was
   retrained with the already-authored `BELLIBOLT_940` profile (m093jp plays kenN2439's identical
   deck) and reached only 0.771 — a statistical tie with kenN2439's own 0.770 (M9), who already lost
   the field gauntlet −0.214 vs v1. Gate failed; Phases 1-2 (arena/gauntlet) were not run. See
   [m15_findings.md](m15_findings.md). All 4 screened candidates (budew 0.48, eduardorochadeandrade
   0.61, legendbrothers 0.62, m093jp 0.77) are now rejected. Would need a **genuinely new**,
   simple-deck ~700–900 teacher not yet sourced — not a re-check of anything screened so far.
3. **Self-play RL to exceed the teacher ceiling — update (M16, 2026-07-17): BUILT and gives a
   real, positive result, still below ship threshold.** `scratchpad/rl_selfplay.py`
   (REINFORCE from the v1.2 init, trust-region anchor, critic refresh). n=300 confirm: iter8
   beats BOTH v1 and v1.2 on the strong gauntlet (+0.027 / +0.024, 7/8 opponents, held-out
   +0.110) — the first lever to move the win objective the right way, opposite of v1.2's
   fidelity-hurts-play finding. But plateaued below the +0.05 ship bar (mechanism: the
   trust-region anchor creates a fixed equilibrium point that more epochs/iterations can't
   move past — see [m16_findings.md](m16_findings.md)) and, per M14, still unverifiable
   offline at this magnitude. `build/imitation-v2rl.tar.gz` built + verified, a ladder
   free-roll bet pending upload. **Not exhausted** — annealing the anchor down, or extending
   RL to the TO_HAND context, are untried follow-ups if pursued further.
4. **Deck-level change** rather than pilot-level — untested since M6, likely still blocked by
   the piloting-skill wall, but a different axis than everything M7–M14 explored.
5. **Domain-expertise-driven heuristics/search (M16, new) — the likely actual gap vs.
   >1000-elo agents on the ladder.** M6 already showed the wall is piloting sophistication,
   not deck/algorithm; neither the user nor the assistant has deep TCG expertise (a known
   constraint since project start). Plan: source real strategy/archetype guides for the decks
   already in play (Bellibolt ex, Lucario ex, Team Rocket swarm, etc.) and translate concrete
   sequencing/energy-priority/damage-math knowledge into `DeckProfile`'s `wants_fn`/`damage_fn`
   (the same slots already hand-authored per deck) and into a better `decision/evaluator.py` V
   for a future, better-informed search attempt (M6/M10's search failures were attributed to
   weak evaluators, not search being infeasible in this game). User is gathering source
   material; not yet started.

## Reusable assets produced

- `scratchpad/strong_gauntlet.py` — the strong harness (now permanent infra; validated as
  falsified for *subtle* gaps, still valid for LARGE ones like v1≫v3, and it correctly re-scored
  v1.2 as ≈v1 offline = the proof that offline ≠ ladder).
- `scratchpad/diagnose_v1.py` (folder-arg) + `diagnose_v12_parity.py` — ladder-log diagnostics.
- `scratchpad/train_residual_main.py` — leave-archetype-out CV harness (OOD validation).
- `scratchpad/train_mlp_v4.py` + `cv_archetype_tohand.py` — built, compile-checked, NOT run for
  ship (Phase B paused). The Plackett-Luce stage-expansion for multi-pick MLP training is reusable.
- `tools/kaggle_api.py` — reads live ladder ratings/schedule per submission (cookies valid).
- No new shipped weights. `bc_650_v3.json` was never written (residual failed G1-OOD).
  `bc_650_v1.json`/`bc_650_v2.json` unchanged.
