# M26 findings — diagnosing Kanga's real ladder play + clone-vs-teacher gap (2026-07-23)

**Bottom line: ran the M10-Phase-0 loss diagnosis on Kanga's two real ladder uploads (139 games)
for the first time, plus a clone-vs-teacher per-turn behavioral comparison. Result is a clean,
evidence-backed NEGATIVE for the "fix a decision leak from the logs" lever, exactly like v1's M10 —
but with one genuinely new structural fact. (1) **No bug-class leak exists**: the shipped bundle
reproduces the live decisions at **100.0% parity (5981/5981), 0 SafePolicy interventions**; lethal
discipline is **98.8% with 0 losing-game misses** (cleaner than the teacher's own 92.8%); routing is
normal (94% learned) with no loss-concentration in a greedy-routed context. (2) **The clone already
matches the teacher on every measurable behavioral axis** — same first-attack turn (5), similar
attacks/turn (clone slightly MORE aggressive), and it benches slightly DEEPER (3.09 vs 2.88) with
identical bench-insurance (~89% of turns). (3) The one new fact: **Kanga loses differently than v1 —
55% of losses are bench-outs (board swept), not prize races** — but the benching probe REFUTES the
"clone under-develops its board" hypothesis, so the bench-outs are deck-structural (the all-in Mega
Kangaskhan build collapses once its main attacker is answered), not a decision defect. No accountable
decision lever surfaced; nothing was built or shipped; no `src/` change.**

Continues [m10_findings.md](m10_findings.md) (the same diagnosis on v1) and the M22–M25 Kanga arc.
New tool: `scratchpad/diagnose_kanga.py` (adapts `diagnose_v1.py` to two live folders + the
`KANGASKHAN_1052` profile + the teacher's dataset for an instrument sanity and the behavioral gap).

## Data (all on disk, nothing downloaded)

- `replays/54911514` (kanga-new, 71 files) and `replays/54893233` (the prior upload, 68) — owner
  "Saulo Quiñones Góngora", ≈139 real ladder games pooled.
- `data/imitation/懒惰的金枪鱼_screen.jsonl.gz` — the teacher's dataset (461 games / 25,478 decisions
  with `raw_observation`), used for the lethal-audit instrument check and the behavioral comparison.
- `data/models/bc_kangaskhan_1052.json` (dim 596; MAIN fidelity 0.652) — the shipped weights.

## Fase 0 — record & loss shape

| folder | record | dominant loss reason | prize margin |
|---|---|---|---|
| 54911514 (kanga-new) | 41W–29L | **bench-out 17**, prize-race 6, deck-out 6 | mostly 5–6 (blowouts) |
| 54893233 (prior) | 41W–26L | **bench-out 13**, prize-race 10, deck-out 3 | mostly 6 |
| **pooled** | **82W–55L (59.9%)** | **bench-out 30/55 = 55%** | 1–2 prizes: only 5 losses; 5–6: 46 |

Loss opponents are diverse (Alakazam 9, Mega Lucario 8, Archaludon 9, Cinderace 3, Bellibolt 1,
Team Rocket 2, other 23) — no single villain, same field thesis as M8.1/M10. **The headline shift
from v1: v1 lost by prize-race 38 / bench-out 2; Kanga loses by bench-out 30 / prize-race 16.** And
the losses are overwhelmingly blowouts (46 of 55 needed ≥5 more prizes = we scored 0–1), only ~5
close losses. Per M10's framing, blowouts are matchup/variance, not decision-fixable.

## Fase 1 — bug-class leak audit (the only directly-accountable lever)

- **Bundle parity: 100.0% (5981/5981), `bc_failures`=0, `bc_used`=5648.** The ladder agent IS the
  trained clone — no serving/deployment bug, no SafePolicy degradation. Rules out the entire
  bundle-bug class outright.
- **Lethal discipline (turn-level, `KANGASKHAN_1052.damage_fn` = Rapid-Fire Combo at its guaranteed
  200): 418/423 = 98.8% taken, 5 missed, and 0 of those in games we lost.** No game-losing lethal
  miss exists — the same decisive result as v1 (0/256). No cheap lethal patch.
- **Instrument sanity — teacher lethal: 1140/1229 = 92.8% taken (7.2% "missed").** The teacher
  appears to take FEWER lethals than the clone. This is partly `damage_fn` over-flagging (it models
  Rapid-Fire at a flat guaranteed 200) and partly the teacher genuinely playing for position — but
  it confirms the clone is if anything MORE lethal-aggressive than its teacher, not leaking KOs.
- **Routing census: 6510 learned / 394 greedy (94% learned).** The context mix in lost games is
  proportional to won games (MAIN dominates both); losses are not concentrated in a greedy-routed
  or unlearned context. Nothing to "learn" that would target the losses.

## Fase 2 — clone (live, 139 g) vs teacher (dataset, 461 g), per-turn axes

| axis | kanga-live | teacher | read |
|---|---|---|---|
| attacks / game | 4.50 | 4.12 | clone slightly more aggressive |
| first-attack turn (median) | 5 | 5 | identical tempo |
| MAIN turns / game | 8.96 | 8.82 | ~same game length |
| attack-turn fraction | 0.502 | 0.466 | clone attacks a bit more often |
| games that ever attacked | 88.5% | 90.9% | ~same |
| **bench depth (max, avg)** | **3.09** | **2.88** | **clone benches MORE, not less** |
| **bench-insurance turn frac** | **0.892** | **0.896** | **identical — a charged bencher ~89% of turns** |

**The clone matches — even slightly exceeds — the teacher on every axis.** Most important, the two
bench metrics **refute the natural hypothesis** that the 55% bench-out losses come from the clone
under-developing its board: it develops a deeper bench than the teacher and has an attack-ready
bencher just as often (~89%). The board is there; the deck simply loses the damage race once the
Mega Kangaskhan is answered. The 0.652 MAIN-fidelity gap to the teacher is therefore **diffuse**
(M20's conclusion), not a concentrated behavioral axis we can point a fix at.

## Answering the user's two questions

1. **"Can we improve decision-making from the logs?"** — No accountable decision leak exists. Parity
   is perfect, interventions zero, lethal discipline clean (0 game-losing misses, cleaner than the
   teacher), routing normal. This is the M10 negative repeating on Kanga: the clone plays as built,
   faithfully; there is no cheap, verifiable decision fix to make.
2. **"Can comparing to the teacher close the gap?"** — The clone already reproduces the teacher's
   tempo, aggression, and board development; it is not visibly under-playing anywhere. The residual
   fidelity gap is diffuse, and per M11/M14 pushing fidelity higher is not a reliable ladder gain
   (it made v1 worse). So the teacher is a reference the clone has effectively already reached on
   every measurable behavioral axis — there is no concentrated gap to close.

## The one structural insight (not a decision fix)

Kanga's losses are **board-sweep blowouts ending in bench-out**, a signature of the all-in Mega
Kangaskhan build (one 300-HP attacker + Crustle; no secondary threat once it falls). v1's TR swarm
lost prize races instead. This is a property of the **deck**, not the pilot — and it is exactly the
clonability-vs-resilience trade the teacher chose. Making Kanga more bench-out-resistant means a
different, more resilient build, i.e. a different teacher/deck — which loops back to the M23/M25
clone hunt (sweet spot ~900–1050 clone-friendly decks), not a decision-policy edit here.

## Verdict

- **No decision-level improvement is available from Kanga's logs** — the accountable levers (parity,
  lethal, routing) are all clean, and the behavioral gap to the teacher is diffuse and already
  effectively closed on measurable axes.
- **Nothing built or shipped; no `src/` change; no upload.** 184 tests pass.
- Standing recommendation holds (M25): the bottleneck is ladder-test slots and deck resilience, not
  a fixable pilot defect. Let Kanga ride at ~760–770; if pushing further, it is a deck/teacher
  question, not a decision-policy one.

## Addendum — is there a MORE FAITHFUL deck than Kanga? (the fidelity hunt, closed)

Prompted by the question "maybe a higher-fidelity clone is the key", we ranked the MAIN
fidelity of all ~38 screened clones and filtered for the full Kanga profile (**fidelity on
FULL data ≥ Kanga's 0.652, teacher ≥ ~1000 elo, abundant data**). Findings:

- **Fidelity alone is disproven as the lever, in our own agents.** v1 (TR_650) has 0.862
  MAIN fidelity — far above Kanga's 0.652 — yet is a *weaker* ladder agent (~686 vs ~760),
  because its teacher is only ~650 elo. Every candidate that beats Kanga's fidelity number
  does so on either thin/subsampled data (optimistic, M21) or a weaker teacher, and **every
  higher-fidelity candidate we actually dueled lost** (akihironomura 0.764 → −0.228,
  alnajafi 0.697 → −0.382, twshin 0.692 → −0.151, Bellibolt/m093jp 0.732 → −0.214).
- The strict filter returned **only Kanga's own teacher**. The one lateral lead —
  **Eduardo Rocha de Andrade** — resolved to a clean negative: its big-data submission
  (54909624, 180 g) is a **DISCARD_DECK (deck strength 0.224, unpilotable)**; its pilotable
  submission (54771566, 78 g) clones at only **0.582** and, at n=12, **ties kanga (−0.007)
  but LOSES to v1 (−0.039)**. The "0.610 fidelity / 3445-game" number that made Eduardo look
  promising was a **stale mismatched `_screen.json`** from an older, no-longer-available
  Eduardo submission — it matched neither testable deck.

**The real formula is fidelity × teacher-strength × clonability-of-skilled-play × data.**
Kanga is the unique sweet spot in our mined pool; no untested candidate dominates it.

## Addendum — backstona & vvs ARE the ITF deck (why they are NOT good candidates)

The two M25 "beat kanga at n=6" candidates were re-examined. Deck extraction is decisive:
**`itfesyskasu`, `backstona`, and `vvs` run the BYTE-IDENTICAL 60-card deck** (18/18 unique
cards, identical counts) — three different pilots of ITF's exact **Mega Starmie ex +
Cinderace + Crushing Hammer** energy-denial/disruption deck (only 6/18 cards shared with
Kanga). Their offline signature is identical to ITF's (tie kanga, beat v1: backstona
+0.018/+0.025, vvs +0.013/+0.032 at n=12) — and **ITF, a pilot of this exact deck, is
losing on the live ladder.** This is the M23 "Where is my orbit" lesson again: cloning the
same deck from a different pilot does not help; the build clones poorly (~0.53) because it
is a high-skill disruption/tempo archetype (like the M8 Lucario), and the gauntlet over-
rates it because the Starmie deck is strong *vs greedy* specifically (no prose-scaling
damage → the generic profile is accidentally correct for it). **Recommendation revised:
do NOT spend a ladder slot on backstona/vvs — they re-run the failing ITF experiment.** Of
the tie-kanga candidates, only **THIRD** (a Team Rocket variant, a genuinely different and
clonable archetype) is a sensible ladder bet.

## Next milestone (M27, decided)

**Try to improve Kanga directly** — either via **strategy search** (domain-knowledge into
the search leaf-V, the channel that worked in M17: the T3 prose-damage fix closed M10's gap
0.34→0.50 vs v1) or **pure self-play RL** (extending REINFORCE, e.g. to the TO_HAND context
named-but-untried since M16). The BC-clone-hunt lever is exhausted for now (no deck beats
Kanga's profile; the offline gauntlet cannot rank the ±0.05 band). Every card claim must
still be verified against `EN_Card_Data.csv` first, and any leaf-V work must target
cross-matchup generalization (the M18 wall).

## Files

New (deletable, none in production): `docs/m26_findings.md`, `scratchpad/diagnose_kanga.py`,
`candidates_eduardo*.txt`, memory. No `src/` change; no weights/profile/bundle touched;
nothing uploaded. Note: `decks/eduardorochadeandrade.csv` and its `_screen.json` were
overwritten across Eduardo submissions during the fidelity investigation (scratch artifacts,
not production).
