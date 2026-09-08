# ADR-0010: Determinized all-contexts receding-horizon search (rung 5)

- **Status**: Accepted
- **Date**: 2026-07-07

## Context

This is the ADR that ADR-0006 reserved: no search algorithm could be chosen
until the benchmark suite had measured throughput, memory, and the decision
statistics of real play. All gates have now been measured:

- **Throughput** ([benchmarking.md](../benchmarking.md)): `search_begin`
  0.34 ms p50, `search_step` 0.39 ms p50 on 2 throttled cores; the per-agent
  budget is ~600 s/episode with **no per-move timeout** — ~15,000
  step-expansions affordable per decision. Latency does not bind.
- **Memory (Q8, resolved 2026-07-07)**: running the production loop shape
  (D=4 determinizations × per-option candidates × rollouts to actor-flip,
  `search_end` between decisions) peaks at **~33 MiB RSS over a 27 MiB
  baseline** (~8 KiB/node) against a 12.2 GiB cap. Memory does not bind.
- **Begin-anywhere (probed 2026-07-07)**: `search_begin` succeeds at every
  live select context we encounter — MAIN, mid-effect sub-selects
  (TO_HAND with a deck view, ATTACH_TO, TO_ACTIVE, DRAW_COUNT), setup, and
  IS_FIRST — and the sim resolves each back to MAIN correctly.
- **Coin nodes (probed 2026-07-07)**: zero COIN_HEAD nodes in ~2,000 my-turn
  rollout steps with the sample deck under either `manual_coin` setting.
- **Strategic evidence** (M4/M5, replay analyses): hand-heuristics plateaued —
  the rung-4 pilot is neutral-or-worse on our best deck; our biggest single
  gain ever was making *more cards playable* (Trainer whitelist expansion,
  +0.097); ladder losses are bench-outs with the prize race even; top players
  play 2.2–2.6 cards/turn to our ~0.8 and win via setup-then-sweep. Heavy
  searchers exist on the ladder and lose to fast heuristics — compute is not
  the edge; decision quality per second is.

## Problem

What search architecture does rung 5 commit to, now that the gates are
measured?

## Alternatives

1. **MAIN-only 1–2-ply expectimax / flat MCTS** (the blueprint §5 prior) —
   searches the widest context but hands every sub-select back to greedy's
   handlers; for non-whitelisted trainers those are `_safe_default` (first N
   options), so search could choose to play a card whose effect greedy then
   butchers. Also "flatten sub-selects into MAIN candidates" fails across
   determinized worlds: the same flattened candidate is a *different action*
   in different worlds (the fetch target may be in the deck in one world and
   in the prizes in another), so cross-world averages are meaningless.
2. **Whole-turn plan caching** — plan once per turn, replay the plan across
   decisions. Buys speed we do not need (50× headroom) at the price of
   plan-vs-reality divergence bugs. Rejected.
3. **Depth-2 with full opponent rollout** — simulating the opponent's turn
   requires their hidden hand, which is filler; a greedy opponent rollout
   would "play" our filler guesses (pure strategy-fusion noise, risk R6).
   Rejected for v0; a *restricted* opponent model (attack/retreat/END from
   their **visible** board only) is retained as a flagged v1 experiment.
4. **All-contexts receding-horizon search** — search EVERY live decision
   (MAIN and sub-selects) with the same loop; depth = the rest of my turn;
   re-plan at the next real decision. No plan is ever carried, so the
   sim-vs-live mismatch of alternative 1 is structurally impossible.

## Decision

Adopt alternative 4. Rung 5 = `decision/search.py` `SearchPolicy`, per
decision (any context):

1. **Fast-paths (no search)**: single legal option; a KO-math lethal that
   takes my *last* prize(s); time pool below the floor or SDK unavailable →
   greedy. (Other lethals are top-priority candidates, sim-confirmed — base-
   damage KO math has both false positives and false negatives.)
2. **Candidates** = the raw legal options of the current select (cap 16;
   unknown PLAY/ABILITY options rank above redundant attach targets when the
   cap binds). Multi-count selects (TO_HAND "up to 3", DISCARD "choose 2")
   use ~6–8 candidate *sets* (greedy's ranked set, single-swap perturbations,
   min-count, themed sets) — never full enumeration.
3. **Determinization** (`planning/determinize.py`): MY zones are *exact
   multiset bookkeeping* — my 60-list minus every visible card (hand, board
   incl. `energyCards`/`tools`/`preEvolution`, discard, revealed prizes,
   `looking`/`select.deck`); shuffle the unseen pool and split into deck
   order + hidden prizes; assert `pool == deckCount + hidden-prize count`,
   else `DeterminizeError` → greedy for this decision. OPPONENT zones are
   mechanically-valid filler weighted toward their observed cards —
   legitimate because a my-turn sim almost never consumes their hidden info.
   D = 4 paired worlds (every candidate scored in the same worlds).
4. **Rollout**: step the candidate, then play out the rest of MY turn with
   greedy as the rollout policy; leaf at actor-flip / terminal / 40-step
   cap. `manual_coin=False` (coin nodes measured absent from my-turn lines);
   lines whose logs show SHUFFLE/COIN get R=2 replicates (the engine's
   shared RNG stream makes each re-step a fresh sample).
5. **Leaf value** = `V(state, root_index)` (`decision/evaluator.py`): linear
   sum of ≤8 normalized features anchored to measured loss modes (prize
   diff, reserve attackers, survival, threat-on-me/them, energy development,
   hand, deck-out), terminal ±1 short-circuit. The root player index is
   fixed at search start — never "negate by leaf perspective" (the leaf's
   `yourIndex` flips at opponent-owned nodes; naive negation double-flips).
   V must be total on `None` actives (post-KO promotion leaves are the
   decisive ones).
6. **Score** = mean over worlds; play argmax. Any exception → greedy answer;
   `SafePolicy` wraps everything; the engine session (`planning/session.py`
   `SearchSession`) is a context manager that calls `search_end()` on every
   exit path.

Time governor: spend cap ~420 s of the ~600 s pool, per-decision cap 6 s,
floor 120 s → pure greedy. The only cross-decision state is the time ledger
(reset in `on_battle_start`).

## Justification

Search's measured value here is (1) the **engine-as-card-interpreter** — the
sim resolves prose effects, unlocking the non-whitelisted trainers/abilities
that account for the largest behavioral gap to top players, and (2)
**survival-aware line planning** against the bench-out loss mode. Both
require the sub-select decisions of unlocked cards to be *searched*, not
delegated to `_safe_default` — hence all-contexts. Receding-horizon
re-planning is affordable (probed ~50–250 steps/decision ≈ 16–100 ms;
~22 searched decisions/game ≈ 1–5 s of a 600 s pool) and eliminates the
cross-world candidate-identity and plan-execution-mismatch failure modes.
Everything heavier (beam over sequences, opponent-turn simulation, belief-
tracked opponent hands) is deferred until the arena shows this design's
specific blind spot.

## Consequences

- Creates `planning/determinize.py`, `planning/session.py` (including the
  cg-dataclass→dict observation converter our parser needs),
  `decision/evaluator.py`, `decision/search.py`; extends `GameState` with
  `build_for(obs, cards, player_index)`, `reserve_attackers`, `max_threat`;
  adds `SearchConfig` to the config schema.
- Promotion gate (ADR-0013 ladder): safe-search(sample) vs safe-greedy
  (sample) = the shipped v5 config, n=300 swapped sides; ship bar = score
  ≥0.65 with the Wilson CI clearing 0.5, plus meta-gauntlet no-regression.
  The deck question is re-opened *under the search pilot* before shipping.
- Whitelists demote from knowledge-gates to candidate-ordering priors.
- Bundle: the four new modules join `_AGENT_MODULES`; the entrypoint imports
  the bundled `cg.api` lazily on the first real decision (`import cg.api`
  runs `GameInitialize` — must not run at module exec), with `_SEARCH_READY`
  tiered under `_AGENT_READY` and the chain search → greedy → random.
- Out of scope (each needs a new ADR or a flagged experiment under this
  one): opponent-turn simulation beyond the restricted model, belief-tracked
  determinization of opponent zones, learned evaluators (rung 6).
