# M18 findings — the search revival, closed: the corrected V helps search but is matchup-narrow, net-neutral across the field (2026-07-18)

**Bottom line: M17 Fase 3 reopened determinized search by showing a T3-corrected leaf V closes
M10's −0.16 gap to the BC champion in the mirror. M18 tuned it (a better self-play-trained V +
lower greedy_bias reached 0.600 in the mirror, Gate A pass) and then tested generalization
against three archetype-diverse opponents — and it FAILS: pooled delta vs v1 is −0.012 (Gate B,
K-B). The mirror advantage does NOT generalize. Search is strongly matchup-dependent: it beats v1
against Bellibolt (+0.084, clears 0.5) but loses to Lucario (−0.053) and Cinderace (−0.067). The
corrected V is well-calibrated for TR-like dynamics (mirror + Bellibolt) and mis-calibrated
elsewhere, and the low greedy_bias that maximized the mirror amplifies the bad overrides too. As a
single shippable agent, search-with-better-V is a wash. The ship path (Fase C: bundle wiring +
Kaggle sandbox probe) does NOT open. imitation-v1 (linear, ~686) remains the champion.**

Continues [m17_findings.md](m17_findings.md) Fase 3. The engine-ships-in-the-bundle feasibility
finding (search CAN physically ship — `cg/` + `libcg.so` are already mandatory bundle members) is
confirmed and recorded, but moot for now since search doesn't beat v1 to justify a slot.

## Fase A — leaf V + greedy_bias sweep (mirror, n=60): Gate A PASS

Controlled matrix, all vs the imitation-v1 mirror on the TR deck, only the leaf V / greedy_bias
differ (`scratchpad/search_v2_pilot.py sweepA`):

| config | mirror score | 95% CI | s/game |
|---|---|---|---|
| v_650_v2 (teacher V) @ gb0.05 | 0.533 | [0.409, 0.654] | 57 |
| v_rl_v2 (self-play V) @ gb0.05 | 0.567 | [0.441, 0.684] | 187* |
| **v_rl_v2 @ gb0.02 (winner)** | **0.600** | [0.474, 0.714] | 37 |
| v_rl_v2 @ gb0.10 | 0.583 | [0.457, 0.699] | 32 |

Two signals: (1) the higher-AUC self-play critic `v_rl_v2` (0.836) beats the teacher-fit `v_650_v2`
(0.801) as a leaf V at equal gb; (2) lower greedy_bias (more search override of BC) nominally helps.
All three v_rl_v2 configs land ~0.57–0.60 — within n=60 noise of each other, none clears 0.5. Gate A
was a point-estimate screen (best ≥ 0.52); it passed. (*The 187 s/game was a batch outlier of long
deck-out grinds; normal search cost is ~35 s/game — comfortably under the ~600 s Kaggle episode
budget, so cost was never the blocker.)

## Fase B — generalization to archetype-diverse opponents (n=60): Gate B FAIL (K-B)

Winner config (`v_rl_v2` @ gb0.02) vs three opponents whose v1 baselines are the M16 n=300 numbers
(`rl_selfplay.V1_BASELINE`):

| opponent | search | v1 base | delta |
|---|---|---|---|
| lucario | 0.267 | 0.320 | −0.053 |
| kenn (bellibolt) | 0.867 | 0.783 | **+0.084** |
| cinderace | 0.300 | 0.367 | −0.067 |
| **pooled** | | | **−0.012** |

Gate B (pooled ≥ +0.05) → **FAIL, K-B fired.** Even generously adding the mirror as a 4th point
(delta +0.100, search's best case) lifts the pooled only to ~+0.016 — still a wash, far below +0.05.

## Diagnosis — why it helps in the mirror/Bellibolt but hurts vs Lucario/Cinderace

Search overrides BC's action whenever the leaf V ranks a different line higher (by more than
greedy_bias). The leaf V is the prose-aware value trained on **TR self-play states** — so it is
well-calibrated for TR-vs-TR tempo (the mirror) and matchups with similar dynamics (Bellibolt, also
an energy-tempo deck: +0.084). Against Lucario (a Fighting tank — high-HP, different prize/threat
geometry) and Cinderace, the V is mis-calibrated, so search **confidently makes worse overrides**
than BC would have chosen. The low greedy_bias (0.02) that maximized the mirror is exactly what
amplifies these bad overrides — it lets the mis-calibrated V override BC more often. So the single
knob that looked best in Fase A is actively harmful out of distribution. This is a cleaner, more
complete version of M10's "cheap search degrades strong play": search degrades play *specifically
where its evaluator is out of distribution*, and helps where the evaluator is in distribution.

## Verdict and disposition

- **Ship path CLOSED.** Search-with-better-V does not beat v1 across a diverse field (pooled −0.012),
  so Fase C (bundle wiring + the 1-slot Kaggle sandbox probe) does not run. No slot spent.
- **M17's "search reopened" was real but narrow.** The mechanism works and the corrected V genuinely
  helps — but only within the leaf V's training distribution. This refines, not contradicts, M17
  Fase 3: the mirror win was true and the −0.16 gap really closed; it just doesn't generalize.
- **The one remaining search lever** is a leaf V calibrated ACROSS matchups (not TR self-play only) —
  e.g. trained on states from all archetypes, or a per-matchup V. That is a large investment against
  a low prior: given the recurring offline↛ladder gap (M11/M14/M17) and that this is still
  incremental over a 650-imitation base, EV is low. Not recommended without a new reason.
- **imitation-v1 (linear, ~686) remains the proven champion and the maximum verifiable result.**

## Reusable assets

`scratchpad/search_v2_pilot.py` (now a general search-arena harness: modes `ab`/`sweepA`/`oppB`,
parametrized leaf V / greedy_bias / opponent), `data/models/v_650_v2.json` (teacher base-7+T3 V),
`data/models/v_rl_v2.json` (self-play base-7+T3 V — the best leaf V we have). Engine-in-bundle ship
feasibility is documented and stands if any future search variant ever beats v1. Logs:
`data/rl/m18_faseA_sweep.log`, `data/rl/m18_faseB_generalize.log`.
