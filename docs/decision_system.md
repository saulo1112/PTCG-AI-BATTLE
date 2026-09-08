# Decision System

One interface, many engines (ADR-0004): every decision the engine can ask —
deck submission, going first, mulligan, setup, main phase, mid-effect
targets, coin calls — flows through
`Policy.choose(DecisionContext) -> list[int]`.

## The pipeline per decision

```
raw obs ─▶ ObservationParser ─▶ DecisionContext{raw, observation, cards}
        ─▶ Policy.choose ─▶ [option indices] ─▶ engine
```

`PTCGAgent` (agent/agent.py) is the glue and contains zero strategy.
`DecisionContext` carries the raw dict too, because the SDK search API
requires the untouched observation.

## Baseline ladder

Each rung must beat the previous one in the arena
(`ptcg_ai.evaluation.run_matchup`) before it earns a submission slot.
Promotion criterion: score rate whose 95% Wilson interval clears 0.5
against the previous rung (see `MatchStats.wilson_interval`).

The rungs share one pipeline (ADR-0013): `GameState.build → ContextRouter →
per-context Handler → Scorer → SafePolicy`. **Only the Scorer changes per
rung** — the router, `GameState`, and handlers are reused, so each rung is a
drop-in, not a rewrite. Full design: [phase2_blueprint.md](phase2_blueprint.md).

| Rung | Policy | Status | Scorer / notes |
|---|---|---|---|
| 1 | `random` | **done** | uniform legal choice; the floor |
| 2 | `safe-random` | **done** | rung 1 + SafePolicy wrapper; the Phase 0/1 submission |
| 3 | `greedy` | **done (M2→M3→M4/W2, 2026-07-06)** | fixed-priority rules: lethal > search Item > develop bench > attach active > evolve > fetch Supporter > draw Supporter > any attack > END. **M2**: 0.738 vs safe-random. **M3**: card-ID-whitelisted Trainer path (search Items + Lillie's + TO_HAND handler) → 0.893 vs safe-random, shipped greedy-v3. **M4/W2**: expanded whitelist (Mega Signal, Cyrano, Waitress + `ATTACH_FROM` handler; whitelists constructor-injectable for ablation) → **0.597 vs the M3 whitelist head-to-head** (95% CI 0.540–0.651, n=300), Mega-on-board 48%→67%; shipped greedy-v5. Board-reading Trainers (Boss's Orders, Switch) + cross-turn energy planning deferred to rung 4. `state/game_state.py` + `decision/greedy.py`. **Card-dependent**: needs the bundled `CardDatabase` (M1) to identify Basics/Trainers and compute KOs |
| 4 | `rule-based` | Phase 2 (M4) | Scorer = `argmax V(predict(state, action))` over a linear evaluator `V(GameState)→[-1,1]`; adds target/supporter/retreat handlers. The benchmark opponent for everything after; **`V` is reused as the search leaf** |
| 5 | `search` | **implemented (M6, 2026-07-07); NOT shipped — underperformed greedy** | determinized **all-contexts** receding-horizon search (ADR-0010): per decision, sample D=4 worlds (my zones exact — `planning/determinize.py`; opponent zones filler), score every candidate by stepping it + greedy rollout to the actor-flip leaf, evaluate with `V(observation, root_index)` (`decision/evaluator.py`), play the world-averaged argmax. Composes a `GreedyPolicy` as rollout + fallback; `SafePolicy` outside. **G1 result: 0.44 vs greedy on the sample deck (worse); weight-tuning 0.35–0.39.** On a deck greedy already pilots near-optimally, a linear `V` deviates and loses; kept as research, greedy-v5 stays shipped. Full analysis: [m6_findings.md](m6_findings.md). `decision/search.py`, `planning/` |
| 6 | learning | Phase 3, own ADR | only if evaluation beats rung 4/5 within resource limits; near-term "learning" = offline coordinate-descent/SPSA tuning of `V`'s weights against arena win-rate (no GPU/net) |

## Candidate decision paradigms (all *candidates* until ADR-0010 — ADR-0006)

| Paradigm | Pros | Cons |
|---|---|---|
| Rule-based / heuristic | cheap, predictable, debuggable, no timeout risk | ceiling limited by author insight; brittle vs unseen lines; organizers say insufficient for the top |
| Greedy evaluation (1-ply) | almost as cheap; uses card math | myopic: no setup planning, no resource husbandry |
| Determinized tree search (MCTS family) | native fit with the SDK's persistent search tree + `manual_coin`; strong tactically | cost per decision (must pass benchmark gate); determinization bias (strategy fusion); belief quality matters |
| RL policy network | fast at inference; can exceed hand-written strategy | training cost + 6-week runway; stability; feature design; opponent-pool overfitting |
| Value network + shallow search | balances both; distillable from self-play | needs both a trained net *and* search budget |
| Hybrid rules + targeted search | spends budget only on hard decisions | complexity of deciding *when* to search |

## Safety layer (always on for submissions)

`SafePolicy` wraps any rung: validates count/uniqueness/range, catches
exceptions, degrades to uniform random, logs every intervention, counts them
(`interventions`). Phase 4 adds a cumulative-episode time-budget watchdog
(Q1 resolved: no per-move limit, 2000 s/episode — competition_analysis.md).
An intervention rate above ~0 in arena play is a bug in the wrapped policy —
the wrapper is insurance, not a crutch.

## Deck decisions

The deck is also a decision (`Policy.choose_deck`), though in practice
Phase 0–2 use a fixed `deck.csv`. Deck iteration happens through arena
evaluation of deck variants (Phase 2+); any learned deck construction is
far-future and out of scope for this competition unless evidence demands it.

## Notes for policy authors

- Handle *every* `SelectKind`, at minimum by falling through to a sane
  default (first option / random) — new contexts can appear mid-competition.
- Never crash on the terminal-observation quirk (see
  [battle_flow.md](battle_flow.md)).
- Set `last_note` with your reasoning at every non-obvious choice
  (ADR-0009) — traces are the primary debugging surface.
- Statelessness between decisions is safest; if you keep per-battle state,
  reset it in `on_battle_start`.
