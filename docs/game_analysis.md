# Game Analysis — Findings

What the data says about the Pokémon TCG environment, one section per
research question. **Dataset**: experiment `random-baseline-1k` — 1000
safe-random vs safe-random games, 51,007 decisions, 0 aborted, collected in
61 s on 2026-07-05 (14 MB gzipped). Methodology and the random-play bias
caveat: [methodology.md](methodology.md). Re-run the identical pipeline at
every ladder rung; this page then gains per-policy columns.

> **Read every number through the caveat**: random play explores the
> decision space but plays terribly — e.g. 90% of its games end by bench
> collapse, which will NOT survive competent play.

## 1. Which decision contexts appear, and how often?

Only **11 distinct contexts** (of ~49 defined in the SDK) appear under
random play with the sample deck:

| context | count | share of decisions |
|---|---:|---:|
| MAIN/MAIN | 35,934 | 70.4% |
| CARD/TO_HAND | 4,515 | 8.9% |
| ENERGY/DISCARD_ENERGY | 2,513 | 4.9% |
| CARD/SETUP_ACTIVE_POKEMON | 2,000 | 3.9% |
| CARD/ATTACH_TO | 1,569 | 3.1% |
| CARD/TO_ACTIVE | 1,093 | 2.1% |
| YES_NO/IS_FIRST | 1,000 | 2.0% |
| CARD/ATTACH_FROM | 764 | 1.5% |
| COUNT/DRAW_COUNT | 630 | 1.2% |
| CARD/SETUP_BENCH_POKEMON | 550 | 1.1% |
| CARD/SWITCH | 439 | 0.9% |

**Implications**: (a) the main-phase decision dominates everything — a
policy that only improves MAIN/MAIN already touches 70% of decisions;
(b) the other ~38 contexts (mulligan, coin calls, damage-counter placement,
tool/special-condition selects…) come from card effects the sample deck
doesn't trigger — **context coverage is deck-dependent**, so fixture and
handler coverage must be re-checked per deck archetype (and the universal
Policy interface + safe fallback is what protects us against unhandled
ones, ADR-0004).

## 2. Branching factor (legal actions per decision)

- Per-game mean: **5.80** (95% CI 5.73–5.86, n=1000 games).
- MAIN/MAIN: mean **7.76**, median 7, **max 50**.
- Everything else is narrow: TO_HAND mean 3.8 (max 6), DISCARD_ENERGY 3.1
  (max 15), setup ≤4.

**Implications**: the action space is *small* — this is not Go. A 1-ply
lookahead over ~8 main-phase options is trivially affordable; even
exhaustive 2–3 ply expansion of MAIN decisions is plausible within budget
(pending ADR-0010 benchmarks). The max-50 tail says option-list handling
must never assume single digits.

## 3. What does the random baseline actually choose?

Within MAIN decisions (chosen counts; uniform ⇒ tracks availability):

| option kind | available (per-option) | chosen | share of MAIN choices |
|---|---:|---:|---:|
| ATTACH | 138,393 | 11,712 | 32.6% |
| END | 35,934 | 8,026 | 22.3% |
| PLAY | 49,189 | 7,319 | 20.4% |
| ATTACK | 39,781 | 6,858 | 19.1% |
| EVOLVE | 12,255 | 1,580 | 4.4% |
| RETREAT | 3,442 | 439 | 1.2% |

**Implications**: availability is dominated by ATTACH options (~3.9 per
MAIN decision — every energy × every target), so uniform random wastes most
of its turns re-arranging energy. Ending the turn is chosen in 22% of MAIN
decisions despite better options existing — the clearest single rule-based
win ("don't END when a free improvement exists"). ATTACK options average
1.1 per MAIN decision — attack *choice* matters less than attack *timing*
under this deck.

## 4. How long do games last?

| metric | mean (95% CI) | median | max |
|---|---|---:|---:|
| decisions | 51.0 (48.8–53.2) | 40 | 213 |
| turns | 14.9 (14.3–15.6) | 12 | 68 |
| wall-clock | 0.04 s | 0.03 s | 0.25 s |

Also: games can end in **under 30 decisions** (observed during bench
warmups), and agent-side decision latency is p50 **0.29 ms** / p95 0.80 ms
(parse + random policy).

**Implications**: ~50 decisions/game bounds per-move budgets (a 1 s/move
policy costs ~50 s/game — fine for any plausible Kaggle timeout); the
engine is never the bottleneck (25 games/s through the full stack,
single-process).

## 5. How do games end?

| reason | share (95% CI) |
|---|---|
| no active Pokémon (bench collapse) | **89.8%** (87.8–91.5) |
| prizes taken | 8.7% (7.1–10.6) |
| deck-out | 1.5% (0.9–2.5) |
| draws | 0 in 1000 games |

**Implications**: random players die by failing to keep a bench — they
discard/never develop, then lose on a KO with nothing to promote. First
rule-based priorities fall out directly: (1) always develop bench
insurance, (2) don't discard needed basics. Prize-race finishes will
dominate once both sides play competently — re-measure at rung 3+. Zero
draws in 1000 games says engine draw caps are practically unreachable under
this deck (Q6 partially answered).

## 6. Which cards appear where? (sample deck; 14,884 turn samples)

- **Hand**: dominated by Basic {W} Energy (67,772 appearances) — the deck
  is energy-heavy and random play doesn't spend them; then Mega Abomasnow ex
  (11,346 — a Stage-2-line top that random play rarely manages to evolve
  into play: it appears on board only ~7.3k times vs 11.3k stuck in hand).
- **Active**: Snover (12,272) > Kyogre (10,169) > Mega Abomasnow ex (7,327).
- **Discard**: 43,603 energy + all four trainers heavily (Mega Signal
  21,352; Lillie's Determination 19,120; Waitress 18,096) — random play
  burns trainers with no plan.

**Implications**: these are deck-composition mirrors, not metagame facts.
The useful readout is the *machinery*: per-zone tracking works, names
resolve, and hand-vs-board deltas expose development failures (a metric
worth formalizing when tuning rule-based play).

## 7–8. Energy attachment and attack rates (per turn; 14,899 turns)

- Manual ATTACH chosen in MAIN: 11,712 → **0.79/turn** (random forgoes its
  free attachment in ~21% of turns).
- ATTACH engine events (incl. effect-driven): 12,476 → **0.84/turn**.
- ATTACK chosen: 6,858 → **0.46/turn**; every attack produced exactly one
  HP_CHANGE event (6,858 — no misses under this deck's attack set).

**Implications**: barely attacking every other turn is the mechanical
signature of random play; attach-rate → attack-rate conversion is a cheap
policy-quality metric to watch climb up the ladder.

## 9. What does the simulator ask for?

By select kind: MAIN 70.4%, CARD 21.4%, ENERGY 4.9%, YES_NO 2.0%,
COUNT 1.2% — SKILL, ATTACK-select, EVOLVE-select, ATTACHED_CARD and
SPECIAL_CONDITION requests never occurred (deck-dependent; see §1).

## 10. How many games are enough?

At N=1000: end-reason shares ±1–2 pp; mean decisions ±2.2; rare contexts
(SWITCH: 0.44/game) still noisy. Guidance table + formula in
[methodology.md](methodology.md#sample-size-guidance-user-question-10).
Default: **1000 games per question**, more only when a specific CI is still
too wide. Collection cost is ~1 min per 1000 random games.

## What Phase 2 takes from this

1. Rule-based effort concentrates on MAIN/MAIN (70% of decisions, widest
   branching): attach-planning, don't-END-early, bench development.
2. The branching factor (~8 in MAIN) makes shallow search cheap — a strong
   prior for ADR-0010, and Q1 (resolved: no per-move limit, 2000 s/episode)
   removes the timeout risk that used to gate that decision.
3. Context coverage must be revalidated per deck archetype; unhandled
   contexts default safely through the universal Policy interface.
4. Baseline yardsticks for the ladder: 0.46 attacks/turn, 0.79
   attaches/turn, 90% bench-collapse losses — competent play must move all
   three dramatically, and the identical pipeline will measure it.

## Open follow-ups

- ~~Q1 (Kaggle per-move budget)~~ — **resolved**: no per-move limit
  (`actTimeout=0`); 2000 s per episode (`runTimeout`). See
  docs/competition_analysis.md and docs/research_questions.md.
- Re-run at rung 3 (greedy) to get first non-random game-length and
  end-reason distributions.
- Formalize a "development failure" metric (hand-stuck evolutions) if
  rule-based tuning needs it.
