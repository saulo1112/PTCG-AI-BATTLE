# M17 findings — real TCG domain knowledge → critic features (2026-07-18)

**Bottom line: the user sourced competitive Pokémon TCG strategy guides to translate into
domain knowledge. Fact-checking every concrete number against the engine's own card data
killed most specific claims (several were Pokémon TCG *Pocket* cards — a different game — or
cards absent here), but isolated ONE real modeling gap: the RL critic's `threat` feature uses
structured damage only, blind to the prose-scaling attacks (Rocket Rush, Voltaic Chain) that
decide our matchups. Fase 1 (offline AUC gate, 222k cached states) cleanly confirmed it — the
prose-aware fix (T3) lifts held-out AUC +0.014/+0.017, while the two "strategic weighting"
terms (prize-weighted threat, board prize pool) add ~0 / fit a confound and were rejected.
Fase 2 (rerun RL with the improved critic `v_rl_v2.json` + λ-anneal — the two mechanisms M16
named as the way past its +0.02 plateau) came back NEGATIVE: the sharper critic left the RL
outcome unchanged (iter20 +0.010 ≈ M16 iter2 +0.011) and annealing λ down drifted the policy
into a K2 kill rather than breaking through. No M17 checkpoint beats M16's iter8 (+0.027).
Net: a clean, evidence-backed negative on the mechanism, on top of genuinely useful fact-
checking and a validated-but-non-transferring critic improvement — the THIRD independent
confirmation (after M11, M14) that offline gains on a subtle component don't move the ladder
objective when applied to the RL critic. The routing target was the critic, NOT
`decision/evaluator.py` (which never ships) nor shipped `DeckProfile` fns (editing them silently
corrupts trained BC weights). BUT — Fase 3 (below) then tested the SAME T3 fix inside the M10 search
mechanism (where threat feeds action selection directly, not just a value baseline) and it is
POSITIVE and significant: it closes M10's entire −0.16 gap to the BC champion (search 0.34→0.50 vs v1,
z=2.3), reopening determinized search as an avenue M10 had closed. The T3 fix is real; whether it helps
depends entirely on WHERE it is applied — inert in the critic, materially useful in search.**

This continues option 5 of [m12_findings.md](m12_findings.md) and the "new direction" of
[m16_findings.md](m16_findings.md).

---

## Why the critic, not the shipped agent (the routing decision)

The prompt asked to encode knowledge into `deck_profiles.py` (`wants_fn`/`damage_fn`) and
`decision/evaluator.py`. Exploration found this must be re-routed:

- **`evaluator.py` never ships.** `submission/builder.py:43-57` (`_AGENT_MODULES`) excludes
  it; the shipped entrypoint dispatches imitation → greedy → safe-random and none import it.
  Its only runtime consumer is `SearchPolicy` (rung 5), which M6/M10 showed plays *worse*
  than the shipped imitation agent. Hand-tuning a V that nothing ships is dead-end work.
- **Editing a shipped `DeckProfile`'s `wants_fn`/`damage_fn` silently corrupts trained
  weights.** The BC feature vector writes damage/wants into fixed slots; changing their
  semantics keeps the vector *length* (so `load_weights` still passes its dim check,
  `policy.py:40-44`) but makes every `bc_*.json` mis-score (train/serve skew). This is why
  new behavior gets a *new* profile name (e.g. `LUCARIO_800_V2`). Not a place for in-place
  edits against the champion.
- **The RL critic uses the same 7 features and is the live lever.** `scratchpad/train_value.py`
  defines the 7 antisymmetric features; the RL advantage is `A = win/loss − V(state)`
  (`rl_selfplay.py:322`), and `refresh_critic` (`rl_selfplay.py:435`) already refits V on
  self-play states gated by held-out AUC. A better critic sharpens credit assignment — the
  M16-identified path to break the plateau. The critic never ships, so zero risk to v1.

## Fase 0 — answers to Tasks A–D (fact-checked against the engine)

### Task A — cross-cutting rules (evaluator / critic features)

The 7 features and weights (`evaluator.py:26-38`, replicated in `train_value.py:36-64`):
`prize` 0.42, `threat` 0.20, `reserve` 0.12, `survival` 0.12, `energy` 0.06, `hand` 0.04,
`deck_out` 0.04. Each is a normalized `(me − opp)` difference.

1. **Prize trading favorable — PARTIALLY present, and FLAT.** The `prize` term exists but
   differences prize-*pile lengths* (`(opp_prizes_left − my_prizes_left)/6`), so a KO of a
   2-prize ex and a 1-prize basic look identical at decision time. `GameState` *already
   computes* per-Pokémon prize value (`_prize_value` → megaEx 3 / ex 2 / else 1,
   `game_state.py:269-280`) and exposes `my/opp_active_prize_value` (`:57-58`) — but **nothing
   reads them; they are dead code.** → **New term T1 (prize-weighted threat).**
2. **Prize mapping — ABSENT.** No term plans KO sequences or ranks targets by prize
   efficiency. The state-form of the idea is "how much prize value each side exposes on
   board." → **New term T2 (board prize pool diff).**
3. **Card sequencing / information — ABSENT, and correctly out of scope for a state value.**
   Draw-before-search ordering is a property of the *action sequence*, not of an observed
   state; the BC policy already learns it implicitly from the teacher. It would only matter
   to a revived search (expansion order). Documented, not encoded.
4. **Hard rules checklist — engine matches physical Standard.** 6 prizes, weakness ×2,
   resistance −30, one Supporter/turn (`game_state.py:30,146-149`; `feature_inventory.md`).
   `damage_fn` and `attack_damage` already apply ×2/−30. No rule bugs found.

**A second, deeper gap in the `threat` feature:** `attack_damage` (`game_state.py:139`) uses
**structured base damage only** — its docstring says "conditional +N effect text is prose and
not modelled at rung 3." So `max_threat` (which feeds the `threat` feature) *underestimates
every prose-scaling attack*: Rocket Rush (30×TR-in-play), Voltaic Chain (20+20×board-{L}),
etc. Measured on 3000 iter8 states, the prose correction is nonzero on **44%** of states and
is *always positive on our side* (our TR swarm is far more lethal than structured damage
shows). → **New term T3 (prose-aware threat delta).**

### Task B — Bellibolt ex

- **"High-Voltage Cannon 70 → 140 with 4+ energy" does NOT exist in this engine.** That is a
  Pokémon TCG **Pocket** card (a different game). Our engine's `269 Iono's Bellibolt ex` has
  **Thunderous Bolt, cost LLLC, flat 230** (EN_Card_Data.csv:521-522), no energy-count
  condition. `damage_940` correctly leaves it as flat structured damage.
- The only prose-scaled Bellibolt-deck attack is Voltorb's **Voltaic Chain = 20 + 20×board-{L}**
  (linear, already correct in `damage_940:427-428`) — *not* a threshold.
- The "4" the guide gestures at is real but is an **attack-cost** heuristic, already encoded:
  `wants_940` wants energy on Bellibolt until it can pay LLLC (`deck_profiles.py:521-522`).
  Nothing to change from this source. (The deploy order Tadbulb-active/Wattrel-bench/Voltorb-late
  is learned by the scorer, not an explicit rule — and correctly so.)

### Task C — Lucario ex

- **Rocky Energy does not exist in the engine** (grep of EN_Card_Data.csv: no match) — the
  Alakazam counter-tech is moot.
- `678 Mega Lucario ex` = **Aura Jab {F} 130 / Mega Brave {F}{F} 270** (flat, structured,
  already captured), with "can't use Mega Brave next turn." **The lock is a real gap but NOT
  observable:** `observation/models.py` exposes only special conditions
  (poisoned/burned/asleep/paralyzed/confused), never an attack-lock flag → **T4 DROPPED** (a
  feature can't read state the observation doesn't surface).
- The evaluator does **not** distinguish damage *sources* (attack HP-damage vs ability/attack
  damage-counter placement). Alakazam *does* exist (`245 TWM` moves counters, `743 MEG`
  Powerful Hand places 2 per hand-card) — a genuine `max_threat` blind spot — but no gauntlet
  opponent runs it, so it is logged as a known limitation, not encoded now.
- Barbaracle (`1052`, Stone Arms) and Cynthia's Garchomp ex (`381`) exist; the intra-turn
  energy-accel / 780-combo lines are policy-sequencing tricks the BC scorer would have to
  learn — out of scope for a state critic.

### Task D — Team Rocket swarm

- **The shipped TR_650 deck is Tarountula/Spidops (Rocket Rush), NOT Koffing/Weezing.** The
  Koffing (`461`, Smog Signals) / Weezing (`462`, Explode Together Now 40×) line *exists in the
  engine* but is in no shipped profile. So the "deploy Koffings even into a KO" concern does
  not apply, and there is **no bug**: `wants_650` is a pure energy-attach preference
  (`deck_profiles.py:184-186`); nothing "abandons" the setup, and bench deployment is scored by
  the learned MAIN scorer.
- **The transferable principle is real and it becomes T3:** swarm damage scales with board
  width (Rocket Rush 30×), and the critic's structured `threat` blinds it to exactly that. The
  fix is prose-aware threat (T3), which covers Rocket Rush for both our side and the tr_mirror
  opponent.

### Source E — Limitless (live tournament data)

Deprioritized to a weak sanity prior only. The engine's card pool is competition-specific and
M9 already showed deck strength is not our bottleneck; over-fitting rules to external meta win
rates would risk the K4 failure mode.

## Fase 1 — the three domain terms and the offline gate

Built (dev-only, no `src/` change, champion untouched):
- `scratchpad/value_features_v2.py` — base 7 + **T1 pw_threat** (structured threat scaled by
  Active prize value 1/2/3, using the dead `*_active_prize_value` fields) + **T2 board_prize**
  (Σ prize value over each side's in-play Pokémon) + **T3 prose_threat** (prose-aware minus
  structured threat, reusing the exact Rocket Rush / Voltaic Chain formulas from
  `deck_profiles`). T4 dropped (not observable).
- `scratchpad/critic_v2_gate.py` — refits the logistic critic per ablation (base / +T1 / +T2 /
  +T3 / +ALL) on the 222k cached self-play states, two temporal splits (v_rl-match tr1-3/va4;
  large tr1-6/va7-8). **GATE: adopt iff best combo AUC ≥ base-7 + 0.010 on both splits.**

**Gate result (`data/rl/m17_critic_gate.log`) — PASSED, with a clean, informative
attribution: of the three domain terms, only T3 earned its place.**

| combo | val AUC (v_rl-match, va4) | vs base | val AUC (large, va7-8) | vs base |
|---|---|---|---|---|
| base-7 | 0.8249 | — | 0.8190 | — |
| base + T1 (pw_threat) | 0.8249 | +0.0001 | 0.8192 | +0.0002 |
| base + T2 (board_prize) | 0.8258 | +0.0010 | 0.8194 | +0.0004 |
| **base + T3 (prose_threat)** | **0.8388** | **+0.0139** | **0.8359** | **+0.0169** |
| base + ALL | 0.8399 | +0.0150 | 0.8369 | +0.0178 |

- **T3 (prose-aware threat) is the entire signal.** Alone it lifts AUC +0.0139/+0.0169 —
  above the +0.010 gate on both splits — and its learned weight (+0.739/+0.704) is the
  second-largest in the model, behind only `prize`. This is the one term that fixes a *known
  engine approximation* (`attack_damage` structured-only), and it earns its place.
- **T1 (prize-weighted threat) adds ~0** (+0.0001/+0.0002). The flat `prize` term plus
  structured `threat` already capture prize trading; weighting threat by Active prize value is
  redundant. Rejected.
- **T2 (board prize pool) is negligible AND wrong-signed** (learned weight −0.42/−0.39). A
  negative weight on "opponent exposes more prize value than me" means it is fitting a confound
  (running ex correlates with stronger decks / bigger board threats), not the intended prize-
  mapping signal. Rejected — putting a confounded term into the advantage baseline would only
  add variance.
- `base+ALL` beats `base+T3` by only +0.0010/+0.0011 (within noise, and inconsistent across
  splits), so the **principled critic is base-7 + T3** — leaner and free of the T2 confound.

**Adopted for Fase 2: `data/models/v_rl_v2.json` = logistic critic on base-7 + T3.** The two
"strategic weighting" terms (T1, T2) are a documented negative result: the flat prize term is
not the bottleneck; the structured-damage approximation is.

## Fase 2 — rerun RL with the v2 critic + λ-anneal: NEGATIVE (the mechanism did not pan out)

Machinery built (`scratchpad/rl_selfplay.py`, backward-compatible): the critic feature fn is now
routed by the loaded payload's `features` list (`_critic_vec`), so the same code runs the 7-feature
v_rl and the 8-feature v_rl_v2; and `iterate(..., lam_final=...)` anneals λ linearly across surviving
iterations (safe by construction — a kill breaks the loop, so reaching iteration t means every prior
probe/K2/K3 passed). Smoke re-verified: sampler parity 300/300, gradient check 8.4e-08.

Run: from the v1.2 init (`bc_650_v2`, same as M16), critic `v_rl_v2.json`, λ 0.1→0.03, 3 iters
(20-22, offset to NOT overwrite M16's iter2-8 artifacts), eval every iter (`data/rl/m17_iterate.log`):

| iter | λ | collect WR avg | pooled d(v1.2), n=160 | d(v1) | held-out cinderace | greedy anchor |
|---|---|---|---|---|---|---|
| 20 | 0.10 | ~0.65 | **+0.010** | +0.008 | +0.041 | 0.794 |
| 21 | 0.065 | ~0.65 | +0.004 | +0.001 | +0.060 | 0.756 |
| 22 | 0.03 | ~0.65 | — | — | — | **KILL K2** (bc_acc 0.825<0.93, agree 0.850) |

**Two decisive reads:**

1. **The sharper critic did not move the RL outcome.** iter20 (first update from the v1.2 init,
   λ=0.1) is **+0.010** — statistically identical to M16's iter2 (**+0.011**, same init, same λ, base
   critic). The critic's +0.017 offline AUC gain translated into ≈0 change in the actual win-
   objective delta. The advantage estimate `A = win/loss − V(state)` was already good enough that
   sharpening V did not sharpen the gradient's usefulness. Collect-WR (the most reliable telemetry)
   stayed flat at ~0.65 — the same plateau signal as M16.
2. **Annealing λ down drifted the policy instead of breaking through.** As λ dropped 0.1→0.065→0.03,
   the pooled delta *decayed* (+0.010→+0.004) and then K2 fired at λ=0.03 (BC accuracy collapsed to
   0.825). This is exactly the M10 failure mode M16 warned that lowering the anchor re-invites: moving
   the equilibrium off the BC init found "play differently," not "win more." The kill worked as
   designed.

**No M17 checkpoint beats M16's iter8 (+0.027).** iter20's +0.010 is already below it at n=160, so the
n=300 confirm (`strong_gauntlet.py 300`) was **skipped** — there is no ship candidate to confirm; it
would only sharpen a number already worse than the existing best. M16's iter8 remains the best RL
checkpoint; imitation-v1 (linear, ~686) remains the proven champion.

## Conclusion — a clean negative on top of a clean positive, reinforcing M14

Fase 0/1 are genuinely valuable: fact-checking killed a batch of false leads (Pocket cards, absent
cards, wrong archetype), and the offline gate cleanly isolated ONE real modeling gap — the critic's
structured-damage threat blindness — fixed it in `v_rl_v2.json`, and rejected the two strategic-
weighting terms with evidence. **But Fase 2 shows, for the third time in this project (M11 offline
accuracy, M14 calibration, now M17 critic AUC), that an offline improvement to a subtle component does
NOT translate to the win objective at this resolution.** The +0.017 AUC critic gain and the anchor-
anneal — the two mechanisms M16 named as the most direct ways past the plateau — did not move it. The
plateau is more robust than the "poor critic + fixed λ" diagnosis suggested; the binding constraint is
likely elsewhere (the MAIN-only fine-tuning scope, or that REINFORCE from a 650-elo BC init simply
cannot reach the >1000-elo band regardless of critic/anchor). Extending RL to the TO_HAND context
(the untried structural fix) is the remaining lever, but its expected value is now lower given that the
two mechanistic fixes attempted here both failed.

**Reusable assets that survive:** `v_rl_v2.json` (a genuinely better offline critic, should RL ever be
re-run at larger scale), the payload-routed critic + λ-anneal machinery in `rl_selfplay.py`, and
`value_features_v2.py` (the prose-aware threat helper, reusable in any future search/evaluator work —
where, unlike the critic, a threat that sees Rocket Rush's real lethality could matter more directly).

## Fase 3 — search pilot: the corrected leaf V DOES improve the search mechanism (POSITIVE)

M16/M17 above tested T3 inside the RL *critic* (advantage baseline) and it did not move the outcome.
But `value_features_v2.py` flagged that a prose-aware threat "could matter more directly in search,
where threat feeds action selection." This pilot tests exactly that: does the M10 determinized-search
mechanism — which lost to plain BC and was blamed on a weak leaf V — improve when the leaf V is the
T3-corrected one?

Controlled A/B (`scratchpad/search_v2_pilot.py`), everything held fixed except the leaf V — same search
plumbing (`ImitationSearchPolicy`), same BC seed/fallback/rollout (v1), same opponent (imitation-v1,
mirror deck), same n, same `greedy_bias=0.05`. The two leaf V's are teacher-trained on the SAME dataset
and fitter (`scratchpad/fit_v650_v2.py`), differing only by the prose_threat column, so any delta is T3
alone (a fresh `v_650_v2.json`, teacher-fit AUC 0.7955→**0.8013** with T3 — the fix helps on the teacher
distribution too, not just self-play). Leaf features are root-seated (`build_for`), matching M10.

| leaf V | vs imitation-v1 (n=100, swapped) | 95% CI |
|---|---|---|
| v_650 (base-7) — M10's evaluator | 0.340 | [0.255, 0.437] |
| **v_650_v2 (base-7 + T3)** | **0.500** | [0.404, 0.596] |

**T3 effect on the search mechanism: +0.16, two-proportion z = 2.32 (p ≈ 0.02) — significant.** (The
overlapping 95% CIs are the well-known misleading heuristic; the difference test is the correct read.)
The n=40 pre-check agreed directionally (0.375 → 0.575, +0.20). **This confirms M10's "weak evaluator"
diagnosis directly: a better leaf V closes the entire −0.16 gap to the BC champion.** search-v1
replicated M10's loss (0.34); search-v2 reaches parity (0.50).

**Honest bounds:** search-v2 *ties* the champion, it does not beat it (0.50, CI still includes but does
not clear 0.5), and it costs ~2× wall time (63 vs 28 s/game — the corrected V drives longer, more
competitive games). Single opponent / single greedy_bias / mirror matchup — a broader confirmation
(opponent variety, gb sweep) is the prerequisite before real investment.

**Why this is the most encouraging M17 result:** unlike the RL plateau (a lever that moves ~0 and won't
compound) and the closed teacher search, this identifies a lever that moved the objective a LOT (−0.16
→ 0.00) via a mechanistically-justified fix — and the ceiling M10 thought was structural turns out to
be the evaluator, which is improvable. Plausible next levers to push search PAST BC: richer leaf
features beyond T3, the self-play-trained critic distribution, or a stronger rollout policy. Search is
**reopened**, not closed — the opposite disposition from M10. New files: `scratchpad/fit_v650_v2.py`,
`scratchpad/search_v2_pilot.py`, `data/models/v_650_v2.json`, logs `data/rl/m17_search_pilot*.log`.

## Ladder note

`build/imitation-v2rl.tar.gz` (iter8) was uploaded as a free-roll: at ~5h it read **561.4** vs
imitation-v1.2's **635.9** — still converging (M16: the ladder needs ~1–2 days). Not yet a
verdict; monitored in parallel.
