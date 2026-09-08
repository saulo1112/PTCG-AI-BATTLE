# Roadmap

Anchor: final submission **2026-08-16**; ladder runs ~2 more weeks after.
Today: 2026-07-05. The plan is algorithm-agnostic until the Phase 2
benchmarks (ADR-0006); MCTS-family search is the *leading candidate*, not a
commitment.

## Phase 0 — Foundation (this session) ✅

- **Goal**: research framework + safe random submission pipeline.
- **Delivered**: package scaffold (13 modules), ADRs 0001–0009, config
  profiles, observability layer, bench harness, fixtures + 64 tests,
  submission builder/validator, docs suite.
- **Success criteria**: `pytest` green; full local battles run traced;
  tarball builds, validates, and smoke-tests. All met.
- **STOP line**: no greedy/rule-based/search/learning code exists. Further
  phases require explicit approval.

## Phase 1 — Research platform & game understanding (redefined; executed 2026-07-05) ✅

Redefined by the user after Phase 0: before any decision algorithms, make
the framework able to answer questions about the game **with data**
(observability, instrumentation, analytics — explicitly NOT playing
strength). The safe-random submission had already passed Kaggle validation.

- **Delivered**: episode format v2 (terminal obs, per-decision timing,
  duration, gzip — ADR-0011); validated event-counting + zone-sampling
  conventions; structured action resolution (`observation/resolve.py`);
  `analytics/` (extract → aggregate → report with CIs); experiment pipeline
  `ptcg collect`/`analyze` with manifests (ADR-0012); rich rendering
  (`battle --verbose`, `show-episode`); 1000-game baseline dataset analyzed;
  docs: feature_inventory (canonical field reference), methodology,
  game_analysis; 90+ tests.
- **Key findings** (details: [game_analysis.md](game_analysis.md)): only 11
  of ~49 decision contexts appear with the sample deck; MAIN/MAIN = 70.4% of
  decisions, mean branching 7.8 (max 50) — the action space is small; random
  play loses 90% of games to bench collapse; ~1000 games resolve headline
  proportions to ±1–2 pp.
- **Q1 resolved (2026-07-05)**: analyzed the validated submission's episode
  logs (`Logs/Submission (1) (5-07)/`) — no per-move timeout (`actTimeout=0`);
  2000 s per episode (`runTimeout`), shared by both agents + engine. Removes
  the timeout risk that gated ADR-0010; see docs/competition_analysis.md and
  docs/benchmarking.md. **Still pending for Phase 2**: `bench all` under
  ~2 vCPU throttling for the benchmarking.md baseline table (now a
  CPU/RAM-viability question, not a timeout one).

## Phase 2 — Competitive baseline & the search decision (weeks 2–3: Jul 13–26)

**Architecture is designed** (2026-07-05): the full engineering blueprint is
[phase2_blueprint.md](phase2_blueprint.md), with ADR-0013 (shared decision
pipeline + reused evaluator) and ADR-0014 (submission packaging). Phase 2 is
now execution of the milestones below; no further architecture decisions are
owed except ADR-0010 (search family/scope), which M5 writes from the M0
numbers.

- **Goal**: a rule-based agent that clearly beats random, a better deck, and
  an evidence-based verdict on search.
- **Milestones** (ROI-ordered; full table in the blueprint §8):
  - **M0** ✅ `bench all` @ 2 vCPU (benchmarking.md filled: `search_begin`
    0.34 ms, `search_step` 0.39 ms — ~30× headroom); RSS memory probe (Q8)
    still owed.
  - **M1** ✅ (2026-07-06) ADR-0014 packaging. Ships a **pruned, verbatim
    `ptcg_ai/` subtree** (7 modules: observation/{models,parser,resolve},
    cards/database, decision/{base,greedy}, state/game_state — empty package
    inits, no config/yaml/environment) + `card_data.json` (624 KiB, all 1267
    cards via `CardDatabase.to_records`/`from_records`). `main.py` runs greedy
    with a per-decision try/except → inline safe-random fallback and an
    `_AGENT_READY` flag. `validate-submission` now confirms greedy actually
    engages (clean-env smoke, `PYTHONPATH` stripped, `_AGENT_READY` + a MAIN
    decision). Real tarball `build/greedy-v1.tar.gz` (2.0 MiB) builds, validates,
    and runs card-aware greedy end to end locally. (Decoupled `cards.database`
    from `environment.sdk` via `TYPE_CHECKING`; +9 tests, 106 green.)
    **`greedy-v1` failed Kaggle's validation episode** (`NameError:
    __file__` — Kaggle `exec()`-loads `main.py` with no `__file__` in scope;
    see the ADR-0014 hotfix note and `Logs/Submission 2 - (06-07)/`). Fixed:
    guarded `__file__` lookup + smoke test now loads `main.py` the same way
    Kaggle does (`exec`, no `__file__`) so this bug class is caught locally
    going forward. `build/greedy-v2.tar.gz` supersedes v1. **Ready to
    upload** once v2 passes the exec-based validation below.
  - **M2** ✅ (2026-07-06) `state/game_state.py` + `decision/greedy.py`
    (rung 3), 14 new tests (104 total green), 0 SafePolicy interventions.
    **Arena gate cleared**: score rate **0.738** vs safe-random (95% Wilson
    CI 0.692–0.778, n=400 swapped). Analytics vs the random baseline: attacks/
    turn 0.46→0.62, attach/turn 0.84→0.92, games 51→26 decisions (decisive).
    Two findings: (a) greedy is **card-dependent** — without the `CardDatabase`
    it degrades to ~0.44 (fixed `analytics/collect.py` to pass cards; this
    makes **M1 packaging the critical path** since Kaggle needs the bundled
    card data); (b) the 0.90 aspiration was missed because the **sample deck's
    consistency (58% energy, no draw) caps the win rate** — 98.5% of games are
    still decided by bench collapse, raising M3's priority. Not yet submitted
    (needs M1).
  - **M3** ✅ (2026-07-06) **Trainer heuristics, not a deck swap.** The 37
    greedy-v2 ladder replays (`Logs/Submission 2 - (06-07)/`,
    docs/replay_analysis.md) showed 17/18 losses were still bench collapse —
    but now because the sample deck has only 6 Basics under 35 energy and rung
    3 played **0 Trainers** (ignored a playable Trainer in 90% of MAIN
    decisions). Redesigned M3 from that evidence: added a **card-ID-whitelisted
    Trainer path** to `decision/greedy.py` (search Items Poké Pad/Dusk Ball/
    Fighting Gong + draw Supporter Lillie's Determination + a TO_HAND
    Basic/energy-preference handler; board-reading Trainers deferred to M4).
    Arena (n=300 swapped): the whitelist alone lifted greedy(sample) from
    **0.738 → 0.893 vs safe-random** (non-overlapping CIs). The candidate deck
    swap (Mega Lucario ex) was measured and **rejected**: only 0.530 vs the
    sample deck under our myopic pilot (CI 0.474–0.586, doesn't clear 0.5) — it
    loses the prize race on scarce energy, a piloting gap M4 must close first.
    Shipped `build/greedy-v3.tar.gz` (trainers + **unchanged** sample deck) —
    validated locally, ready to upload. `decks/meta_lucario.csv` +
    `meta_dragapult.csv` kept for the M4 revisit. 5 new greedy tests (113 green).
  - **M4** — reframed by the 33 greedy-v3 replays (`Logs/Submission 3`,
    docs/replay_analysis.md) into three ROI-ordered workstreams:
    - **W2** ✅ (2026-07-06) expanded the Trainer whitelist (Mega Signal item
      + Cyrano/Waitress fetch-Supporters + a mid-effect ATTACH_FROM handler,
      all constructor-injectable for ablation). Arena: new whitelist **0.597
      vs the M3 whitelist on the same deck** (95% CI 0.540–0.651, clears 0.5),
      Mega-on-board 48%→67%, 0 interventions. Shipped **`build/greedy-v5.tar.gz`**
      (W2 policy + unchanged sample deck) — validated, ready to upload.
    - **W1** deck swap — **deferred to W3** (thrice-confirmed): under the
      myopic pilot, three candidate decks all lose the prize race to the
      35-energy sample deck (Lucario 0.530, water-v2 0.283, sample+basics
      0.457 — none clears 0.5) because greedy can't plan energy across turns.
      Decks kept in `decks/` as W3 validation targets.
    - **W3** the rung-4 evaluator `V` + `decision/rule_based.py`: cross-turn
      energy planning (unlocks the deferred decks), attack/attach selection by
      KO math, promote/retreat, board-reading Trainers. Beats the W2 agent in
      the arena, then re-opens the deck question with a competent pilot.
  - **M5** **ADR-0010** from M0 numbers (expected: targeted determinized
    search on MAIN; only after this may search implementation start).
- **Dependencies**: Phase 1 numbers (have them). **Complexity**: medium.
- **Risks**: rule-based ceiling too low (mitigate: rules focus on the
  decisions that dominate outcomes — bench development, attack selection,
  energy allocation); search verdict negative (fallback: stronger rules +
  targeted shallow lookahead).
- **Success criteria**: greedy/rule-based ≥ 90% score rate vs random
  (Wilson-CI cleared); ADR-0010 written; a rung-3-or-better agent submitted.

## Phase 3 — The chosen strengthening path (weeks 3–5: Jul 27 – Aug 9)

- **Goal**: implement what ADR-0010 selected.
- **Deliverables** (menu depends on the ADR):
  - Search path (blueprint M6): `planning/` (SearchSession + Determinizer
    extracted from the bench), `state/beliefs.py`, search policy rung 5, time
    governor keyed on the ~600 s/agent budget, tuning; gate vs rung 4.
  - Learning path (needs its own ADR with benchmark arithmetic):
    `features/`, `training/`, parallel self-play, rung 6.
  - Either way: continued deck iteration; weekly submission of the best
    arena-validated agent.
- **Dependencies**: ADR-0010. **Complexity**: high.
- **Risks**: this is the phase where time runs out — the arena-validated
  rule-based agent from Phase 2 is the permanent fallback submission.
- **Success criteria**: new rung beats rule-based in arena with CI cleared;
  leaderboard rating improves over the Phase 2 agent.

## Phase 4 — Freeze & endgame (weeks 5–6: Aug 10–16)

- **Goal**: maximize final rating, minimize variance.
- **Deliverables**: feature freeze (~Aug 10); robustness hardening
  (SafePolicy cumulative-episode-budget watchdog per the 2000 s/episode
  limit — Q1, docs/competition_analysis.md — memory checks, long soak runs);
  final deck choice; **final two submissions by Aug 14–15**
  (buffer for validation failures), chosen per the last-2-active rule.
- **Success criteria**: zero validation errors; both active agents are the
  two strongest we own; no submission burned on deadline day.

## Standing rules

- Every phase ships its best agent — the leaderboard is the integration
  test.
- Any architectural commitment mid-phase gets an ADR.
- Deferred modules are created only when their trigger fires
  (architecture.md Growth Plan).
