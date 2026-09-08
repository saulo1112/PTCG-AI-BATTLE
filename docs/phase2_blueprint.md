# Phase 2 Blueprint — Competitive Agent Architecture

**Status**: design deliverable (no agent code). This is the engineering
blueprint the roadmap's Phase 2 promised: every major architectural decision
made and justified, so implementation is engineering, not architecture.
Companion ADRs: [0013](decisions/0013-decision-architecture.md) (decision
architecture), [0014](decisions/0014-submission-packaging.md) (packaging).
ADR-0010 (search family/scope) is written from the benchmark numbers in
milestone M5, not here.

Everything below is grounded in one of three evidence sources, cited inline:
- **A1** synthetic analytics ([game_analysis.md](game_analysis.md), N=1000
  random self-play);
- **A2** real ladder replays ([replay_analysis.md](replay_analysis.md), 10
  competitive episodes at μ≈600);
- **A3** measured throughput ([benchmarking.md](benchmarking.md), 2-vCPU run).

---

## 1. Situation and strategic thesis

We hold one live safe-random submission at μ≈600, ~7W–13L. The replays (A2)
show **all five recoverable losses are turn-5–7 bench collapse, four with an
empty bench** — mechanically identical to synthetic random play (A1: 89.8%
bench-collapse). At 600 the pool is fast heuristics/random agents that never
spend more than ~13 s of a ~600 s/agent budget (A2). The organizers state
pure rule-based play won't top the leaderboard, and no opponent at 600
searches.

**Thesis**: the rating curve has two regimes and we build for both.
1. *Escape 600* by not losing to ourselves — bench development, energy
   attachment to a real attacker, attacking when it helps. A one-ply greedy
   agent (rung 3) fixes 100% of observed losses (A2). This is days of work and
   the single largest expected jump.
2. *Climb above the heuristic pack* with **engine-backed determinized search**
   on the main-phase decision (rung 5). It is the only approach that reads the
   prose-only card-effect semantics "for free" (the engine resolves them), the
   branching factor is tiny (A1: MAIN mean 7.8), and throughput has ~30×
   headroom under budget (A3, §7). Search is our differentiator precisely
   because the pool doesn't use it.

Rules are the floor and the permanent fallback; search is the ceiling bid. The
ladder (Random→Safe→Greedy→Rule-based→+Search→Learning) and its Wilson-CI
arena gates ([decision_system.md](decision_system.md)) are unchanged.

---

## 2. Architecture comparison

Scored for *our* problem: small action space, hidden information, prose-only
effect semantics, an engine-backed determinized simulator available at
runtime, ~600 s/agent, one engineer, ~5 weeks, no GPU. (H/M/L = high/med/low.)

| Candidate | Playing strength | Impl. cost | Debug cost | Extensibility | Handles hidden info | Uses card effects | Verdict |
|---|---|---|---|---|---|---|---|
| **Priority rule engine** | L–M (fixes our losses; low ceiling) | **L** | **L** | M | via hand rules | via hand math only | **Adopt as rung 3** |
| **Utility AI (shared evaluator)** | M | M | M | **H** (evaluator reused by search) | via features | via evaluator terms | **Adopt as rung 4** |
| Behaviour tree | ~ rule engine | M | M | M | no gain | no gain | Reject (control-flow sugar) |
| Finite state machine | < rule engine | L | L | L | no gain | no gain | Reject (too rigid for MAIN) |
| GOAP / classical planning | M (if action models existed) | H | H | M | no | **needs an action model we lack** | Reject (no forward model in our own code) |
| **Engine determinized search** | **M–H** | H | H (nondeterministic bias) | **H** | determinization + beliefs | **engine resolves effects** | **Adopt as rung 5 (ADR-0010)** |
| RL policy / value net | H *if trained* | **H** | H | M | learned | learned | Reject for now (no GPU, runway, variance) |
| Value net + shallow search | H | **H** | H | M | learned | learned | Defer (needs the net first) |

**Why the winners win and losers lose**:
- Behaviour trees and FSMs are *organisation* of rules, not new capability;
  against hidden information and unstructured effects they buy nothing a
  priority/utility scorer doesn't. Rejected to avoid ceremony.
- GOAP needs a symbolic action-effect model. We have **no** hand-written
  forward model of card effects (they are English prose, A1/feature_inventory
  §9) — the only forward model we possess is the *engine itself*, which is
  exactly what search uses. So "planning" for us = engine search, not GOAP.
- RL is the highest ceiling but the worst ROI here: no GPU, ~5 weeks, the
  single-battle-per-process SDK (sdk_analysis) throttles self-play, and
  opponent-pool overfitting (R12) is real. Kept alive only in the cheap form
  of *offline weight-tuning of the linear evaluator* via arena (§5.4) — not a
  neural net.
- **Utility AI over "just more rules"** because the utility evaluator is the
  same function search needs at its leaves. Building it once as rung 4 makes
  rung 5 an integration, not a rewrite. That reuse is the crux of the whole
  design (ADR-0013).

**Rejected-alternatives ledger** (so we don't relitigate): pure BT/FSM/GOAP
(no capability gain), neural RL as the Phase-2/3 core (ROI under the runway),
committing to a search family before A3-style numbers (ADR-0006 forbids it),
and hand-encoding 1,267 cards' effect text (astronomically expensive vs
letting the engine resolve them).

---

## 3. Strategic knowledge the agent must encode

The concepts that separate competent from random play, and where each enters
the design:

| Concept | Meaning here | Enters via |
|---|---|---|
| **Board development / bench insurance** | Always have a Basic in reserve to promote after a KO. **The** lesson from A2 (empty-bench losses). | Greedy rule + evaluator term (saturating bench count) |
| **Tempo** | Each turn you may attach exactly one energy and (usually) attack once — wasting either is unrecoverable time. A1: random forgoes attach 21% of turns, attacks 0.46/turn. | Evaluator: energy-on-board vs attack costs; "don't END with a free improvement" |
| **Prize race** | First to take 6 prizes wins; ex/mega give the opponent 2/3 prizes when KO'd. Trading a 1-prize attacker into their 3-prize Mega is good math. | Evaluator: prize differential weighted by ex/mega prize value of what's threatened |
| **Attacker selection** | Pick the attack that KOs (weakness ×2, resistance −30) or sets up the KO, not just the biggest number. | Greedy KO check; evaluator "can-KO-now" term |
| **Energy economy** | Energy is the scarce accelerator; attach toward the Pokémon that will actually attack, not spread thinly (A1: random wastes attaches re-arranging). | Attach handler targets the best attacker; evaluator rewards attack-readiness |
| **Evolution timing** | Evolving raises HP/damage but costs the turn's development; `appearThisTurn` blocks same-turn evolve. | Evolve handler + legality from `appearThisTurn` |
| **Retreat management** | Retreat costs energy and the once-per-turn retreat flag; retreat a stranded/dying active only when it saves tempo or an attacker. | Rule-based retreat handler + evaluator |
| **Supporter management** | One Supporter per turn — usually your draw/consistency engine; spend it unless holding for a specific effect. | Rule-based MAIN ordering |
| **Resource conservation** | Don't discard Basics/energy you need; don't over-draw into deck-out. | Discard/draw handlers; evaluator deck-out horizon |
| **Risk management** | Coin-flip attacks, sleep/confusion, whiff chances — prefer lines that don't hinge on a flip when a safe line exists. | Search chance-node handling (`manual_coin`); evaluator variance-aversion (late) |

---

## 4. Decision pipeline (shared by all rungs)

```
raw obs ─▶ ObservationParser ─▶ DecisionContext{raw, observation, cards}
        ─▶ GameState.build(ctx)              # NEW: perspective-normalized, derived once
        ─▶ ContextRouter(select.type/context)
             ├─ MAIN            ─▶ MainHandler   ─▶ candidate actions ─▶ Scorer ─▶ pick
             ├─ SETUP_ACTIVE/BENCH, SWITCH, …    ─▶ context Handler   ─▶ Scorer ─▶ pick
             └─ unhandled (~38 contexts)         ─▶ SafeFallback (first/random legal)
        ─▶ SafePolicy wrap (always on)
```

The router, handlers, and `GameState` are shared; **only the Scorer changes
per rung**. That is what makes the ladder a sequence of drop-in replacements
rather than rewrites.

### 4.1 GameState (`state/game_state.py`, new in Phase 2)

A frozen, perspective-normalised snapshot built once per decision from
`ParsedObservation` + `CardDatabase`. Grows additively — add a field only when
a rule/evaluator consumes it. Initial fields (all derivable from ★★★-R
features, feature_inventory §11):

- **Prize race**: `my_prizes_left`, `opp_prizes_left`; `opp_active_prize_value`
  and `my_active_prize_value` ∈ {1,2,3} from `ex`/`megaEx`.
- **Per in-play Pokémon**: attached energy multiset vs each attack's cost →
  `ready_attacks`, `attacks_one_energy_away`; `hp`, `max_hp`.
- **KO math**: `damage_to_ko(attacker, defender)` applying weakness ×2 /
  resistance −30; `can_ko_opp_active_now`, `opp_can_ko_my_active_next`.
- **Retreat**: `retreat_affordable(pokemon)` vs attached energy and the
  once-per-turn `retreated` flag.
- **Development**: `my_bench_count`, `has_bench_insurance` (≥1 promotable
  Basic in play or hand).
- **Resources**: `hand_size`, `deck_count`, `deck_out_in` (turns), tempo flags
  `energy_attached`, `supporter_played`, `retreated`, `stadium_played`.
- **Threat summary**: opponent's best known next-turn damage (upper bound from
  card statics over their in-play attackers).

`GameState` is SDK-free and unit-testable from parsed fixtures (import rule 4,
architecture.md). It does **not** own hidden-info inference — that is the
future `BeliefTracker`; `GameState` reports only what is visible + derived.

### 4.2 Evaluator `V(GameState) → [-1, 1]` (the shared leaf function)

A linear weighted sum of ~15–25 normalised terms, hand-initialised, arena-
tuned later (§5.4). Terminal states return ±1 (win/loss) / 0 (draw). Sign
convention: positive = good for the acting player. Rung 4 scores each
candidate action by `V(predict(state, action))`; rung 5 uses `V` at search
leaves.

Initial term set and rationale (weights are starting guesses, to be tuned):

| Term | Normalisation | Init weight | Why (evidence) |
|---|---|---|---|
| prize differential `(opp_left − my_left)/6` | [-1,1] | **1.00** | the win condition; margin ignored by rating but prizes are the proxy for winning |
| can-KO-opp-active-now × its prize value | {0,1}×{1,2,3}/3 | 0.45 | tempo + prize math (§3 attacker selection) |
| my-active-dies-next-turn (opp can KO) | {0,1} | −0.40 | avoid walking into KOs; drives retreat/switch |
| bench insurance (saturating: min(bench,2)/2) | [0,1] | 0.35 | **A2**: empty bench = the loss |
| board attack-readiness (Σ ready attackers, capped) | [0,1] | 0.30 | tempo; convert attach→attack (A1: 0.46 atk/turn) |
| energy-on-board vs needs (fraction of costs met) | [0,1] | 0.20 | energy economy |
| hand size (draw health, saturating) | [0,1] | 0.10 | consistency; avoid decking into nothing |
| deck-out horizon (penalty as `deck_count`→0) | [-1,0] | 0.15 | endgame safety |
| status-condition penalty on my active | [-1,0] | 0.10 | asleep/paralysed = lost tempo |

Weights live in one config/table so tuning never touches logic. Normalisation
keeps every term in a bounded range so weights are comparable and the sum
stays in `[-1,1]` after a final squash. Early/mid/late modulation (§4.4) is a
small multiplier on a subset of terms, not a second code path.

### 4.3 Per-context heuristic framework

For the 11 contexts that actually occur (A1 §1); everything else falls through
`SafeFallback`. Each handler produces candidate actions; the Scorer ranks
them.

| Context (share) | Objective | Key info | Heuristic (rung 3 → rung 4) | Failure modes to guard |
|---|---|---|---|---|
| **MAIN/MAIN** (70.4%) | best turn action | full GameState | 3: if lethal attack → do it; else attach to best attacker if not yet attached; play Basics to bench if thin; evolve if it improves attacker; only END when no scored improvement. 4: enumerate legal MAIN actions, score by `V(predict(·))`, pick argmax; END scored like any action. | never END with a free attach/attack (A1: 22% do); don't loop non-progress abilities; respect once-per-turn flags |
| CARD/TO_HAND (8.9%) | keep useful cards | hand/deck contents | prefer Basics, evolution pieces, energy for the active attacker; avoid keeping dead cards | discarding a needed Basic (A1 §5) |
| ENERGY/DISCARD_ENERGY (4.9%) | pay cost / discard cheaply | attached energy by type | discard energy off benched/non-attackers first; keep the active's attack energy | stripping the attacker's power |
| CARD/SETUP_ACTIVE (3.9%) | choose opening active | Basics in hand | pick the intended attacker line's Basic with best HP/attack, not a fragile filler | opening with a dead-end Basic |
| CARD/ATTACH_TO (3.1%) | choose attach target | attackers & costs | attach to the Pokémon closest to a usable attack (esp. the active) | spreading energy thin (A1) |
| CARD/TO_ACTIVE (2.1%) | promote after KO | benched Pokémon | promote the best available attacker / least-valuable sacrifice per game state | promoting a card you can't power |
| YES_NO/IS_FIRST (2.0%) | going first? | deck archetype | default per archetype (setup-hungry decks often take first for tempo); A/B test (open Q) | untested guess — measure |
| CARD/ATTACH_FROM (1.5%) | source of attach | zones | prefer energy from discard/deck effects that keep hand options | — |
| COUNT/DRAW_COUNT (1.2%) | how many to draw | deck_count | draw the max that doesn't risk imminent deck-out | over-drawing into deck-out |
| CARD/SETUP_BENCH (1.1%) | fill bench in setup | Basics in hand | bench every legal Basic (development first) | leaving Basics in hand |
| CARD/SWITCH (0.9%) | switch active | GameState | switch a dying/stranded active for an attacker when it nets tempo | needless switch wasting energy |

Context coverage is **deck-dependent** (A1 §1) — when we change decks or climb
into effect-heavy matchups, new contexts appear and route to `SafeFallback`
until handled. That is the ADR-0004/0005 safety contract, not a bug.

### 4.4 Conflict resolution, tie-breaking, safety, fallback

- **Scoring conflict** (rung 3 rules): fixed priority order lethal-attack >
  develop-bench-if-thin > attach-best-attacker > evolve-improvement >
  setup-play > END. Rung 4 replaces the order with `argmax V`.
- **Tie-break**: highest `V`; on equal `V`, deterministic stable option order
  (reproducibility). Never random at rung 3+/4 except inside SafeFallback.
- **Dynamic weights**: a phase multiplier keyed on `turn`/prizes — early game
  up-weights development & attack-readiness; late game up-weights prize
  differential & deck-out. One multiplier table, no branching logic.
- **Safety**: `SafePolicy` wraps every rung (unchanged). An intervention rate
  > ~0 in arena is a bug in the wrapped policy, not the wrapper's job.
- **Fallback policy**: unhandled contexts and any handler that returns no
  candidate → first legal / minCount answer (never crash; A2: no terminal obs,
  local terminal-obs quirk — clamp counts).

---

## 5. Search integration (rung 5) and its interfaces

Designed now so rung 5 is a Scorer swap plus a thin `planning/` module, with
**no refactor** of rungs 3–4. Implementation waits for ADR-0010 (M5).

### 5.1 What the SDK gives us (sdk_analysis)

`search_begin(agent_obs, your_deck, your_prize, opponent_deck, opponent_prize,
opponent_hand, opponent_active, manual_coin)` forks the true engine state into
a **determinized** copy where we supply concrete guesses for every hidden
zone; `search_step(id, select)` advances a node; nodes form a **persistent
tree** (backtracking is free). `manual_coin=True` turns coin flips into
YES/NO nodes we control. Deterministic in search mode. `DecisionContext.raw`
already carries `search_begin_input`; `bench_search.py` already demonstrates
the full loop with filler determinization.

### 5.2 Interfaces to build (contracts fixed now)

- **`Determinizer.sample(state, beliefs, rng) -> HiddenAssignment`** — concrete
  IDs for `your_deck / your_prize / opponent_deck / opponent_prize /
  opponent_hand / opponent_active`, matching the counts `search_begin`
  requires. **v0**: sample uniformly from the *unseen multiset*. The
  bookkeeping half is exact and cheap: my remaining deck = my 60 − hand −
  board − discard − seen prizes; opponent's *revealed* cards accumulate from
  their `PLAY/ATTACH/EVOLVE`/discard logs. Only the split of the opponent's
  unseen pool across hand/deck/prizes is guessed. This is "BeliefTracker-lite"
  — pure counting, no modelling (feature_inventory §10, state_representation).
- **`SearchSession`** — extracted from `bench_search.py`: owns `search_begin`/
  `search_step`/`search_release`/`search_end`, the node tree, and cleanup.
- **In-tree sub-decisions** (our own mid-effect selects, the opponent's
  replies): resolved by the **rung-4 policy** as the rollout/inner policy — so
  the evaluator does double duty as leaf value *and* in-tree action selector.
- **Chance nodes**: with `manual_coin=True`, enumerate both coin branches at
  shallow depth (≤2); beyond that, sample. Weight branches by their known
  probabilities from card text where structured, else 0.5.

### 5.3 Search scope (decided in ADR-0010; strong prior from A3)

A3 measured `search_begin` ≈ 0.34 ms, `search_step` ≈ 0.39 ms on 2 cores.
With a corrected per-decision budget `B ≈ 6 s` (½ of ~600 s/agent over ~50
decisions), affordable step-work `≈ 15,000` expansions/decision — ~30× above
the `D×N ≥ 500` "full search on the table" bar (benchmarking.md). **Latency is
not the constraint.** The prior for ADR-0010:
- Search **only on MAIN/MAIN** (70% of decisions, the widest and most
  consequential — A1); rules elsewhere.
- Determinized **expectimax / flat-MCTS**, depth 1–2 ply of MAIN, `D`≈3–8
  determinized worlds averaged, `V` at leaves.
- Binding risks are **memory** (native tree not visible to Python — RSS probe,
  Q8) and **quality** (determinization bias / strategy fusion, Q4/R6), which
  ADR-0010 addresses by arena-validating rung 5 vs rung 4 before trusting it.
- **Time governor**: per-decision budget = f(`remainingOverageTime`, expected
  remaining decisions); hard floor — if the pool drops below ~120 s, drop to
  rules only. This pulls the Phase-4 watchdog (R2) into rung 5's design.

### 5.4 Optional: offline evaluator tuning (the only "learning" we keep)

Because `V` is linear, its weights can be tuned by black-box optimisation
(coordinate descent / SPSA) against the arena win-rate vs rung 4 — cheap, CPU-
only, no GPU, no net. This is the entire learning surface for the competition
unless evidence demands more (rung 6 keeps its own ADR). It is optional and
never on the critical path.

---

## 6. Submission packaging (ADR-0014)

Today the tarball ships only stdlib `main.py` + `deck.csv` + `cg/`; **no
`ptcg_ai` code ships**, so a smart policy cannot run on Kaggle as built
(architecture.md, builder.py). ADR-0014 decides:

- Bundle a **stdlib-only** package `agentpkg/` (parsed models + card statics +
  `GameState` + handlers + evaluator; **no config/yaml/pytest deps**) into the
  tarball. Rung 5 additionally imports the already-bundled `cg/` (native
  `libcg.so` is present for Linux Kaggle).
- `main.py` adds its own dir to `sys.path`, imports `agentpkg` **inside
  try/except**, and **falls back to the current inline safe-random** on any
  import or runtime error — so R1 (crash = forfeit) safety is preserved
  unconditionally.
- Amends ADR-0001's "entrypoint imports nothing from the research stack": it
  may import the *shipped* `agentpkg`, never the dev-only `ptcg_ai`.
  `validate-submission` gains a check that the staged tree imports with only
  `cg/` + stdlib on the path (catches a stray dev-only import before upload).

This is milestone M1 and a prerequisite for *every* rating gain — sequence it
first.

---

## 7. Benchmark, experiment, and Kaggle strategy

### 7.1 Promotion gates (unchanged mechanism)

Each rung must beat the previous in `evaluation.run_matchup` with a 95% Wilson
interval clearing 0.5 (`MatchStats.wilson_interval`, decision_system.md).
Raise `evaluation.n_games` from the dev default (20) to **≥400** in the
benchmark profile so a 55% edge is separable (Wilson half-width ≈ ±0.049 at
n=400). Swap sides on (cancels first-player advantage). Re-run the Phase-1
analytics pipeline at each rung as a **regression watch**: attacks/turn,
attach/turn, end-reason mix, game length — competent play must move all three
away from the random yardsticks (A1 §7–8).

### 7.2 Every improvement is an experiment

Recorded in the `research_questions.md` convention: hypothesis, expected
effect, metric, acceptance criterion, rollback criterion. E.g. rung 3:
*hypothesis* "greedy bench-development + attack-when-lethal ≥ 90% score vs
safe-random"; *metric* arena score rate + bench-collapse-loss share;
*acceptance* Wilson CI clears 0.5 (target ≥0.9); *rollback* if it regresses vs
safe-random on any deck.

### 7.3 Kaggle cadence

- **Submit at every rung promotion** (greedy → deck → rule-based → search) and
  keep the two active slots as two *different* strong agents (last-2 rule,
  competition_analysis).
- New submissions start at μ₀=600 and need episode runway to climb — upload
  the final agent by **Aug 14–15** and lean on the ~2-week post-deadline
  ladder for convergence (roadmap Phase 4).
- Never upload unvalidated: `validate-submission` + a local self-match first.
- Watch **leaderboard rating vs arena prediction divergence** (R12); a large
  gap means the arena opponent (rung 4) doesn't represent the ladder — pull
  fresh replays and re-tune.

### 7.4 Honest expectation

A2 says rung 3 removes every *observed* loss cause at 600, but the magnitude
of the climb and the metagame above ~800 are unknown until we get there.
Treat Kaggle rating as the integration test and the arena as the unit test;
do not over-fit to n=10 replays.

---

## 8. ROI-ranked milestones

| # | Work | Effort | Expected leaderboard impact | Complexity | Debug difficulty | Risk |
|---|---|---|---|---|---|---|
| **M0** | ✅ `bench all` @ 2 vCPU + (todo) RSS probe; benchmarking.md filled | 0.5d | unblocks ADR-0010 | L | L | none |
| **M1** | Packaging ADR-0014: `agentpkg/` + fallback entrypoint + import validation | 1–2d | **prerequisite for all gains** | M | M | low (fallback preserves R1) |
| **M2** | `state/game_state.py` + greedy rung 3 + tests + arena gate + **submit** | 2–3d | **largest single jump** (kills bench-collapse losses, A2) | M | L | low |
| **M3** | Deck arena (sample-trimmed vs meta Lucario vs one pool archetype) + adopt winner + **submit** | 1–2d | large, multiplicative with M2 | L | L | low |
| **M4** | Evaluator `V` + rule-based rung 4 (target/supporter/retreat handlers) + gate + **submit** | 3–4d | medium–large | M | M | low–med |
| **M5** | **ADR-0010** from M0 numbers (expected: targeted determinized search on MAIN) | 0.5d | decision quality | L | L | none |
| **M6** | `planning/` + `Determinizer` v0 + rung-5 search on MAIN + time governor + gate + **submit** | 4–6d | our top-quartile bid | H | H | med (R6 quality trap; rung 4 fallback) |
| **M7** | Continuous: evaluator weight tuning (§5.4), belief v1 (log bookkeeping), deck v2, Q4 experiment | ongoing | steady | M | M | low |

Phase 4 (freeze ~Aug 10, final two submissions Aug 14–15) unchanged. If time
runs out at any point, the last arena-validated rung is already submitted —
the ladder guarantees a submittable agent at every step (R15).

---

## 9. Feature prioritisation (from feature_inventory.md, R columns)

- **Critical** (rung 3 consumes today): `select.option` + structured
  resolution (`resolve.py`); tempo flags `energyAttached`/`supporterPlayed`/
  `retreated`; both boards' `active`/`bench` with `hp`/`energies`/`tools`/
  stages/`ex`/`megaEx`; `prize` counts; own `hand`, opponent `handCount`/
  `deckCount`; card statics (damage/cost/HP/weakness/resistance/retreat/
  stages); `search_begin_input` (rung 5).
- **Important** (rung 4 / belief v1): `discard` piles, `logs` incl.
  `*_REVERSE`, special conditions, `appearThisTurn`, `stadium`, `benchMax`.
- **Optional**: `looking` peeks, `serial` entity tracking, SKILL ordering,
  COIN history, `preEvolution`.
- **Ignore (this phase)**: prose effect-text NLP, tool/energy sub-identity
  beyond counts, spectator `visualize_data`.

---

## 10. Risks (delta to risk_analysis.md)

- **New**: bundled-`agentpkg` import failure on Kaggle → silent degrade to
  safe-random. *Mitigation*: try/except fallback + `validate-submission`
  import check (ADR-0014); a live game logs the fallback. (Add as R16.)
- **Carried, now sharper**: search-quality trap (R6) — determinization bias
  can make search play *worse* than rung 4 despite throughput; gate rung 5 in
  the arena before trusting it, and keep rung 4 as the active fallback slot.
- **Carried**: native search-tree memory (R14/Q8) — RSS probe owed at M0/M6.
- **Carried**: premature commitment (R4) — ADR-0010 stays gated on M0 numbers.

---

## 11. Open questions carried into implementation

- **Q4** determinization-accuracy sensitivity — arena experiment (ground-truth
  vs uniform vs belief-tracked) at M6/M7.
- **Q6** draw frequency under competent play — measure at rung 3+.
- **Q8** native search-tree memory ceiling — RSS probe.
- **Q11 (new)** does `actTimeout=0` + ~600 s/agent `remainingOverageTime` hold
  on *competitive* ladder episodes? Confirmed on 10 replays (A2); re-check
  every new drop — search time-budgeting depends on it.
- **IS_FIRST** preference by archetype — A/B on the ladder.
- **Metagame above ~800** — unknown until we climb; the deck pipeline (M3/M7)
  stays warm to respond.

---

## 12. Summary recommendation

Build one shared decision pipeline (`GameState` → context handlers → Scorer →
SafePolicy). Ship it in ladder order: greedy rung 3 (fixes our actual losses),
a trimmed/better deck, a utility rung 4 whose evaluator is reused as the search
leaf, then — gated by ADR-0010 on the already-measured throughput headroom —
targeted determinized engine search on the main-phase decision. Reject
behaviour trees, FSMs, GOAP, and neural RL as poor ROI under the constraints;
keep only cheap linear-evaluator tuning as "learning". Package a stdlib
`agentpkg/` with an unconditional safe-random fallback so play-strength gains
never cost submission safety. Submit at every rung; the arena gates each
promotion; the leaderboard is the final integration test.
