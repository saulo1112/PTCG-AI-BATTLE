# Replay Analysis — Real Ladder Episodes

Empirical evidence from real Kaggle games played by our live safe-random
submission (μ≈600). This is the counterpart to
[game_analysis.md](game_analysis.md) (which analyses *synthetic* random-vs-
random self-play): here the opponents are real ladder agents, so the patterns
reflect what actually loses and wins games at our current rating.

**Dataset**: `Logs/Submission 1 - (05-07)/` — 10 distinct competitive
episodes with recoverable outcomes (files `84146602.json` … `84153791.json`)
plus the self-play validation episode (`Self Match/Replay.json`). Analysed
2026-07-05 with read-only scripts against the vendored enum tables
(`cg/api.py`) and card names (`pokemon-tcg-ai-battle/EN_Card_Data.csv`).
Re-run this analysis on every new submission's downloaded logs.

> **Caveat**: n=10 is small — treat proportions as direction, not precision.
> Where a pattern appears in *all* losses it is safe to act on; where it is a
> minority it is a hypothesis for the arena.

---

## Submission 2 — greedy-v2 (37 competitive replays, 2026-07-06)

**Dataset**: `Logs/Submission 2 - (06-07)/` (37 episodes; the
`Submission error (fixed)/` subfolder — greedy-v1's failed validation — is
excluded). This is our first *improved* agent (rung 3 greedy) on the ladder.

**Methodology fix (applies to the Submission-1 tables above too)**: in the
Kaggle replay format, `steps[t][agent].action` is the response to the
observation stored at `steps[t-1][agent]`, not `steps[t]`. The Submission-1
outcome/board/timing columns were correct, but the earlier per-action *rate*
figures (e.g. "attacks/turn 0.23") were computed on the wrong pairing. All
greedy-v2 numbers below use the corrected action↔observation pairing.

**Record: 19W–18L (~51%).** The policy executed exactly as coded — every
observed decision matches `decision/greedy.py`; **zero fallbacks, zero
SafePolicy interventions**. Behaviour vs the random baseline improved sharply:
we attack on **84% of our turns** (opponents 46%), attach to the Active 100%
of the time, and go first whenever offered (19/19). Going first correlated
with winning (15W–9L first vs 4W–9L second).

**But 17 of 18 losses are still bench collapse, and at the last decision of
every single loss our state was: bench = 0 AND Basics-in-hand = 0.** The
policy had nothing to bench — the deck physically ran out of Pokémon:

- The vendor sample deck holds **6 Basic Pokémon in 60 cards** (4 Snover,
  2 Kyogre) buried under **35 energy**. When the opening Basic is KO'd and no
  second Basic has been drawn, the game is lost regardless of policy quality.
- **We played 0 Trainer cards all 37 games** (rung 3 had no Trainer logic),
  ignoring a playable Trainer in **90% of MAIN decisions** — including the
  deck's own 4 Lillie's Determination that could have dug for Basics.

**Root-cause shift**: bench collapse is *still* the dominant loss mode, but the
cause moved from "policy never benches" (random era) to "policy has nothing to
bench + never draws to find one" (greedy era). The fix is **draw/consistency,
not more combat heuristics** — see the M3 result below.

**Opponent pool (unchanged thesis)**: still nobody searches (max 13 s of
~600 s used by 35/37; two opponents burned 113 s / 189 s — we beat both).
Meta decks recovered in full from `action` arrays: Mega Lucario ex (item-dense,
lean), Dragapult ex, Mega Gardevoir ex, plus sample-deck mirrors.

### M3 experiment result (arena, 2026-07-06)

The Submission-2 evidence pointed at "swap to a better deck". Measured under
the greedy policy (`ptcg_ai.evaluation` hosting, n=300 swapped, Wilson gate):

| Arena | Matchup | Score rate (95% CI) | Reading |
|---|---|---|---|
| A | old-greedy(Lucario) vs greedy(sample) | 0.253 (0.207–0.305) | meta deck **loses** without Trainer play — it is engine-dependent (13 energy vs 35) |
| B1 | greedy+Trainers(Lucario) vs greedy(sample) | 0.530 (0.474–0.586) | Trainer logic rescues Lucario to a **wash** — does **not** clear 0.5; it loses the prize race (scarce energy → can't power attackers) |
| B2 | greedy+Trainers(sample) vs safe-random | **0.893 (0.853–0.923)** | vs the M2-recorded **0.738 (0.692–0.778)** for old-greedy(sample) — **the Trainer whitelist alone is +0.155 on the deck we already ship**, CIs non-overlapping |
| B3 | greedy+Trainers(Lucario) vs safe-random | 0.977 (0.953–0.989) | Lucario's ceiling is higher vs weak play, but locked behind energy planning (M4) |

**Decision**: ship the **Trainer heuristics on the unchanged sample deck**
(greedy-v3); **defer the Lucario deck swap to M4**. The deck swap does not
clear the promotion bar under a myopic pilot, and it adds Kaggle-robustness
risk (unvalidated card contexts); the policy change is a clean, low-risk,
already-Kaggle-validated win. Decks kept in `decks/` for the M4 revisit.

---

## Submission 3 — greedy-v3 (33 replays, 2026-07-06) → M4

**Dataset**: `Logs/Submission 3 - (06-07)/` (33 episodes). First replays of the
M3 Trainer-enabled agent. **Record 14W–19L**; the chronological run climbed a
mid-streak into a visibly stronger pool (opponents now bench 3–5, play full
Trainer suites — 314 PLAYs / 58 abilities / 13 stadiums vs our 61 PLAYs — and a
new Lillie's Clefairy ex archetype appears).

**M3 worked as designed, and it wasn't enough.** The Trainer path is live on
Kaggle (Lillie's played 29×; IS_FIRST YES 15/15; **zero fallbacks/
interventions**); attack tempo is healthy (~0.8–1.0/turn vs opp 0.2–1.0); dead
turns are gone (END ≈9% of MAIN). **But 19/19 losses still ended with bench≤1
AND zero Basics in hand** — now *with* the draw engine running (only 3/19 died
holding an unplayed Lillie's). Conclusion: **draw-6 cannot beat a 6-Basic/60
skeleton.** Two derived findings:

1. **The deck is a hard ceiling.** 6 Basics under 35 energy; when the opener is
   KO'd before a second Basic is drawn, no policy recovers.
2. **Conversion fails vs the stronger pool.** Several long losses attacking
   every turn for **0 prizes taken** (e.g. 84440373: 19 turns, 0.78 atk/turn,
   0 prizes) — Snover/Kyogre chip damage vs 130–350 HP bodies. Mega Abomasnow
   reached the board only 16/33 games (its fetchers Mega Signal/Cyrano sat dead
   — not whitelisted in M3).

**Search is still premature**: none of the 33 losses trace to a decision a
lookahead would change; they are structural (no Basic exists) or conversion
(need real attackers + KO-aware selection), and the opponent pool still doesn't
search (max 45 s of ~600 s). Fix order = trainers, deck, evaluator — then
reassess search at the M5 gate (ADR-0006/0010).

### M4/W2 result (arena, 2026-07-06)

Expanded the whitelist with the deck's own dead value-Trainers (verified to
resolve safely: Mega Signal/Cyrano take no sub-select; Waitress triggers an
`ATTACH_FROM` handled by preferring the Active):

| Test (n=300 swapped) | Score rate (95% CI) | Reading |
|---|---|---|
| new whitelist vs **M3 whitelist**, sample deck | **0.597 (0.540–0.651)** | W2 beats the shipped greedy-v3 config head-to-head; Mega-on-board 48%→67% |
| new whitelist vs safe-random, sample deck | 0.893 (0.853–0.923) | at ceiling vs the weak floor (unchanged) |

**Deck swap re-tested and rejected a third time** (all under the W2 policy):
Lucario 0.530 (M3), water-v2 **0.283**, sample+basics **0.457** vs the sample
deck — none clears 0.5. The mechanism is now unambiguous: water-v2 gets its
Mega out *more* (90%) and crushes random (0.983) but **loses the prize race
201/300** because its lean 17 energy leaves attackers unpowered under a pilot
that attaches only to the Active and never pre-charges the bench. Adding energy
back (sample+basics, 29 energy) recovers most of the gap (0.283→0.457),
confirming **greedy is energy-hungry; deck quality is blocked on W3's
cross-turn energy planning.**

**Decision**: ship **greedy-v5 = W2 policy + unchanged sample deck** (a
measured +0.097 over greedy-v3, CI clears 0.5, zero deck risk). Deck swap and
board-reading Trainers move to **W3** (rung-4 evaluator). `decks/` holds three
validated candidates (Lucario, water-v2, sample+basics) as W3 targets.

---

## Replay JSON format (Kaggle `cabt` env)

Top-level keys: `configuration`, `specification`, `info`, `rewards`,
`statuses`, `steps`, plus metadata. What matters:

- `info.TeamNames` / `info.Agents` — player names; `info.EpisodeId`.
- `configuration` — `{actTimeout: 0, episodeSteps: 10000000, runTimeout: 2000,
  seed: …}` (confirms the Q1 timeout facts on *competitive* episodes, not just
  the validation one).
- `rewards` — `[r0, r1]`, each `+1`/`-1`/`0`; the winner has the higher
  reward. `statuses` — `["DONE","DONE"]`.
- `steps` — list of turns; each is a 2-element list (one entry per agent).
  Per entry: `action` (the indices the agent returned — **its decklist of 60
  IDs on the first decision**), `status` (`ACTIVE`/`INACTIVE`/`DONE`),
  `reward`, and `observation` = the exact dict our `agent()` received:
  `{current, logs, select, search_begin_input, remainingOverageTime, step}`.
- The acting agent in a step has `status == "ACTIVE"` and a non-empty
  `select.option`; the other is `INACTIVE`.

**Two format facts that constrain our tooling** (both now handled/known):

1. **`remainingOverageTime`** is present on every Kaggle observation and sits
   near **600.0 at game start**, counting down as that agent consumes wall
   clock. This is a **per-agent** budget of ~600 s, distinct from the 2000 s
   whole-episode `runTimeout`. Our safe-random agent finished with 597–600 s
   remaining (≤13 s used) in every game. **This is the real compute budget a
   search policy spends against** — see the correction in
   [competition_analysis.md](competition_analysis.md) and Q1/Q11 in
   [research_questions.md](research_questions.md).
2. **No terminal observation is delivered.** The winning move ends the game;
   the loser's last stored observation is a mid-game `select` it never got to
   answer with `result != -1`. End reasons must be **inferred** from the
   winner's final board (prizes remaining, active/bench emptiness, deckCount),
   not read from a `RESULT` log. Analytics extractors on Kaggle logs must not
   assume the terminal-obs shape that local episodes provide (contrast the
   local terminal-obs quirk in [battle_flow.md](battle_flow.md)).

## Outcomes and per-episode signals

Record over the 10 competitive replays: **5W–5L** (consistent with the
reported ~7W–13L over the full ~20; this subset is the recoverable half).

| episode | opponent (deck) | result | end | turns | our max bench | opp max bench | our atk/turn | opp atk/turn | opp time used |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 84147263 | Mega Lucario ex | **L** | bench-out | 5 | **0** | 5 | 0.0 | 0.67 | 11.3 s |
| 84147950 | (Fighting/Solrock) | **L** | bench-out | 5 | **0** | 5 | 0.5 | 0.33 | 0.7 s |
| 84149419 | Mega Lucario ex | **L** | bench-out | 6 | **0** | 5 | 0.0 | 0.0 | 9.4 s |
| 84146602 | Mega Lucario ex | **L** | bench-out | 7 | 1 | 1 | 0.33 | 0.0 | 0.7 s |
| 84151535 | (Dragapult ex) | **L** | bench-out | 5 | **0** | 0 | 0.0 | 0.33 | 0.6 s |
| 84148794 | Mega Lucario ex | **W** | opp bench-out | 16 | 1 | 4 | 0.67 | 0.0 | 9.8 s |
| 84150263 | Mega Lucario ex | **W** | prizes/late | 29 | 4 | 5 | 0.27 | 0.0 | 13.0 s |
| 84150891 | (Water mirror-ish) | **W** | opp bench-out | 44 | 3 | 1 | 0.33 | 0.0 | 11.5 s |
| 84153153 | Mega Gardevoir ex | **W** | prizes/late | 39 | 1 | 4 | 0.12 | 0.13 | 2.7 s |
| 84153791 | (mirror) | **W** | opp bench-out | 21 | 2 | 0 | 0.1 | 0.0 | 0.7 s |

(Prize counts and exact reasons reconstructed from the winner's final view;
"bench-out" = loser had 0 Active and 0 Bench = SDK finish reason 3, "no Active
Pokémon".)

## The loss pattern — one cause dominates

**All 5 losses are bench collapse on turns 5–7, and in 4 of 5 we never put a
single Pokémon on the bench** (our mean max-bench across losses ≈ 0.2 vs
opponents' ≈ 3.0). The failure mode is mechanical and identical to synthetic
random play ([game_analysis.md](game_analysis.md) §5: 89.8% bench-collapse),
so real ladder play at 600 has **not** yet punished us for anything subtler
than "keep Pokémon in play":

1. We open with one Basic, never develop the bench, take one or two KOs, and
   have nothing to promote → instant loss (finish reason 3).
2. Random target/attach selection means our lone attacker is usually not even
   powered to attack: **0.23 attacks/turn for us** across all games. Half our
   games we never attack at all.
3. Games we *win* are the ones that run long (16–44 turns) — i.e. games where
   the **opponent** also fails to close, and our random play stumbles into a
   prize lead or the opponent benches out first. We are not winning by playing
   well; we are winning coin-flips of mutual incompetence.

**Every one of these is fixed by the rung-3 greedy heuristics already named in
[decision_system.md](decision_system.md)**: develop bench insurance, don't
discard needed Basics, attach toward a viable attacker, attack/END correctly.
The replays are direct evidence that rung 3 removes 100% of our *observed*
loss causes at this rating. What it will *not* tell us is the ceiling above
600 — see open questions.

## Opponent behaviour at μ≈600

- **Nobody searches.** Max wall-clock consumed by any opponent in any episode
  was **13.0 s of ~600 s**; most used < 1 s. The 600-pool is entirely fast
  heuristic/random agents. A search policy has the entire budget to itself.
- **Skill is bimodal.** Some opponents are random-like (END with an attack
  available, 0.0 attacks/turn, e.g. 84149419). Others are clearly
  heuristic: they bench 4–5 Pokémon every game and attach ~1 energy/turn
  (the Mega Lucario ex players), which is exactly why they beat our
  non-developing random agent so cleanly on turn 5.
- **No crashes or timeouts** observed on either side in this sample.

## Metagame intel (recovered decklists)

Every replay exposes **both** full 60-card decklists (each agent's first
`action` is its deck). Recovered lists at 600:

- **Ours (vendored sample)**: Mega Abomasnow ex line (Snover x4 → Mega
  Abomasnow ex x4), Kyogre x2; **35× Basic {W} Energy**; trainers Mega Signal
  x4, Lillie's Determination x4, Waitress x4, Cyrano x2; Maximum Belt x1 (ACE
  SPEC). **No draw-power items, 58% of the deck is energy** — a structural
  consistency handicap (why we brick and fail to develop).
- **Dominant ladder deck — Mega Lucario ex** (4 of 10 opponents, several
  identical lists): Riolu → Mega Lucario ex, Solrock; Basic {F} Energy ×13;
  item-heavy consistency (Dusk Ball, Poké Pad, Premium Power Pro, Fighting
  Gong) + Carmine/Lillie's draw supporters. A lean, fast, low-energy build —
  the opposite of ours.
- **Others seen**: Dragapult ex (Dreepy→Drakloak→Dragapult ex, multi-type
  energy, Crushing Hammer disruption + Ultra Ball/Buddy-Buddy Poffin draw),
  Mega Gardevoir ex (Ralts line + Rare Candy + Salvatore/Smoochum), and Water
  mirrors of our own list.

**Takeaway for the deck track**: the sample deck's energy glut and absent draw
engine are a measurable disadvantage against the item-dense meta lists. Cutting
energy to ~12–15 and adding consistency draw is a cheap, high-leverage change
(blueprint deck milestone M3). We can legally rebuild the observed meta lists
from the public pool for arena sparring.

## What Phase 2 takes from this

1. **Bench development + attach-a-viable-attacker + attack-when-lethal is the
   entire near-term rating story at 600** — rung 3 is not a warmup, it is the
   fix for our actual losses.
2. **Search has the whole ~600 s/agent budget** and no competition from the
   600-pool — the differentiator once rung 3/4 saturate the easy wins.
3. **Deck quality is a first-class, cheap lever**: our own deck is a
   consistency liability vs the meta.
4. **Tooling**: Kaggle-log analytics must infer end reasons (no terminal obs)
   and can mine `remainingOverageTime` for opponent compute profiling and both
   decklists for metagame tracking — fold into the analytics subsystem when we
   next pull logs.

## Open follow-ups (replay-specific)

- Re-pull logs after each submission; grow n and track whether higher-rated
  opponents introduce loss causes beyond bench collapse (attack selection,
  prize-race timing, disruption like Crushing Hammer).
- Confirm the ~600 s `remainingOverageTime` per-agent budget holds as we climb
  (Q11) — it is the number search time-budgeting depends on.
- Formalise a "development failure" metric (turns-to-first-bench,
  attach→attack conversion) as a per-rung regression watch.
