# Feature Inventory — Canonical Observable-Field Reference

Every field the SDK exposes to an agent, classified and rated for future
phases. **This is the canonical reference**: when designing rules, search
evaluations, or (later) feature vectors, start here.

Sources: vendored `cg/api.py` (authoritative field list + inline docs),
engine C++ (`ToJson.h` hidden-info rules), and our captured fixtures
(empirical shapes). Field names appear exactly as in the raw JSON /
`observation/models.py` mirrors.

**Visibility classes**
- **public** — always present and truthful for both players.
- **self-only** — truthful for the acting player; hidden (`None`/count-only)
  for the opponent.
- **hidden** — never directly visible; must be *inferred* (beliefs) or
  *guessed* (determinization for `search_begin`).
- **derived** — not a wire field; computable from public/self-only data plus
  the static card database (these are the future `GameState`/feature
  candidates).

**Usefulness ratings** (per consumer): ★★★ core signal | ★★ situational |
★ marginal/bookkeeping. R = rule-based, S = search, L = learning (RL/NN).

---

## 1. Observation top level (`Observation`)

| Field | Type | Visibility | Meaning | R | S | L |
|---|---|---|---|---|---|---|
| `select` | SelectData \| None | public | The decision request; None only on Kaggle's initial deck call. **Terminal obs carries a stale select — never a decision** (battle_flow.md) | ★★★ | ★★★ | ★★★ |
| `logs` | list[Log] | viewer-relative | Events since THIS viewer's last selection (opponent's hidden moves appear as `*_REVERSE`) | ★★ | ★★ | ★★ |
| `current` | State \| None | mixed (below) | Board snapshot for the acting player | ★★★ | ★★★ | ★★★ |
| `search_begin_input` | str | public (opaque) | Serialized engine state (opponent-hidden info already erased server-side); the ONLY key to `search_begin` | — | ★★★ | — |

## 2. Game state (`State` / `current`)

| Field | Type | Visibility | Meaning | R | S | L |
|---|---|---|---|---|---|---|
| `turn` | int | public | 1 = first player's 1st turn, 2 = second player's 1st, … (0 = setup). Parity identifies the acting side | ★★★ | ★★★ | ★★★ |
| `turnActionCount` | int | public | Actions taken this turn (engine caps runaway turns) | ★ | ★ | ★ |
| `yourIndex` | int | public | Whose decision this observation requests (0/1) — the perspective anchor | ★★★ | ★★★ | ★★★ |
| `firstPlayer` | int | public | Starting player; -1 before the coin | ★★ | ★★ | ★★ |
| `supporterPlayed` | bool | public | Supporter already used this turn (one per turn) | ★★★ | ★★★ | ★★★ |
| `stadiumPlayed` | bool | public | Stadium already played this turn | ★★ | ★★ | ★★ |
| `energyAttached` | bool | public | Manual energy attachment used this turn — THE tempo resource | ★★★ | ★★★ | ★★★ |
| `retreated` | bool | public | Retreat already used this turn | ★★ | ★★ | ★★ |
| `result` | int | public | -1 ongoing / 0,1 winner / 2 draw | ★★★ | ★★★ | ★★★ (terminal reward) |
| `stadium` | list[Card] (0–1) | public | Stadium in play (affects both players) | ★★ | ★★ | ★★ |
| `looking` | list[Card\|None] \| None | self-only | Cards currently revealed to you mid-effect (deck peeks, opponent-hand reveals). Transient but occasionally gold (e.g. seeing opponent's hand) | ★★ | ★★ | ★ |
| `players` | list[PlayerState] ×2 | mixed | Both players' zones (below) | ★★★ | ★★★ | ★★★ |

## 3. Per-player zones (`PlayerState`)

| Field | Type | Visibility | Meaning | R | S | L |
|---|---|---|---|---|---|---|
| `active` | list[Pokemon\|None] (0–1) | public (None if face-down during setup) | The battling Pokémon; empty after a KO until replaced | ★★★ | ★★★ | ★★★ |
| `bench` | list[Pokemon] (≤ benchMax) | public | Benched Pokémon, fully visible incl. attachments | ★★★ | ★★★ | ★★★ |
| `benchMax` | int | public | Usually 5; card effects can change it | ★★ | ★★ | ★★ |
| `hand` | list[Card] \| None | **self-only** | Your cards; `None` for the opponent — the core hidden information | ★★★ | ★★★ | ★★★ |
| `handCount` | int | public | Opponent hand SIZE is public — disruption/draw signal | ★★ | ★★★ | ★★★ |
| `deckCount` | int | public | Cards left; deck-out horizon (loss at 0 on draw) | ★★★ | ★★★ | ★★★ |
| `discard` | list[Card] | public | Fully visible — the belief tracker's best friend (played resources, KO'd attackers, spent energy) | ★★ | ★★★ | ★★★ |
| `prize` | list[Card\|None] (≤6) | **hidden** (own AND opponent's are face-down; length = prizes remaining) | Prize race state; contents unknown (6 of your own 60 are locked away — deck-composition uncertainty about YOUR own deck too) | ★★★ (count) | ★★★ (count + guess) | ★★★ |
| `poisoned/burned/asleep/paralyzed/confused` | bool ×5 | public | Active Pokémon's special conditions (asleep/paralyzed ⇒ can't attack/retreat; confused ⇒ coin risk) | ★★★ | ★★★ | ★★ |

## 4. In-play Pokémon (`Pokemon`)

| Field | Type | Visibility | Meaning | R | S | L |
|---|---|---|---|---|---|---|
| `id` | int | public | CardData ID → full static profile via `all_card_data()` | ★★★ | ★★★ | ★★★ |
| `serial` | int | public | Unique physical-card identity within the match (tracks THIS Kyogre across zones/logs) | ★★ | ★★★ | ★★ |
| `hp` / `maxHp` | int | public | Current/max HP (maxHp reflects tools/effects) — KO math | ★★★ | ★★★ | ★★★ |
| `appearThisTurn` | bool | public | Entered play this turn (evolution timing, some attack conditions) | ★★★ | ★★ | ★★ |
| `energies` | list[EnergyType] | public | Attached energy as TYPES (bitmask enum incl. RAINBOW/TEAM_ROCKET) — attack-cost math | ★★★ | ★★★ | ★★★ |
| `energyCards` | list[Card] | public | The actual attached cards (special energies have card effects) | ★★ | ★★ | ★★ |
| `tools` | list[Card] | public | Attached Pokémon Tools | ★★ | ★★ | ★★ |
| `preEvolution` | list[Card] | public | Evolution stack underneath (devolution effects, Rare-Candy detection) | ★ | ★★ | ★ |

## 5. Cards (`Card`)

| Field | Type | Visibility | Meaning | R | S | L |
|---|---|---|---|---|---|---|
| `id` | int | public where card is visible | Card TYPE identity (join key to CardData) | ★★★ | ★★★ | ★★★ |
| `serial` | int | public where visible | Physical identity within match | ★ | ★★ | ★ |
| `playerIndex` | int | public | Owner | ★ | ★ | ★ |

## 6. Decision request (`SelectData`)

| Field | Type | Visibility | Meaning | R | S | L |
|---|---|---|---|---|---|---|
| `type` | SelectType (11 values) | public | Coarse decision kind (MAIN, CARD, ENERGY, YES_NO, COUNT, …) | ★★★ | ★★★ | ★★★ |
| `context` | SelectContext (49 values, may grow) | public | Fine-grained WHY (SETUP_ACTIVE, MULLIGAN, IS_FIRST, ATTACH_TO, DISCARD, COIN_HEAD…) — the semantic router for specialized handling | ★★★ | ★★★ | ★★★ |
| `minCount` / `maxCount` | int | public | Selection arity bounds (minCount 0 ⇒ optional). Invariant `maxCount ≤ len(option)` holds ONLY at real decisions | ★★★ | ★★★ | ★★★ |
| `remainDamageCounter` | int | public | Damage counters still to place (spread-damage effects) | ★★ | ★★ | ★ |
| `remainEnergyCost` | int | public | Energy still to pick (ENERGY selects) | ★★ | ★★ | ★ |
| `option` | list[Option] | public | THE legal-action list — the engine guarantees legality; agents only ever pick indices | ★★★ | ★★★ | ★★★ |
| `deck` | list[Card] \| None | self-only | Your deck contents, revealed ONLY during deck-search effects (a free peek at your remaining deck — belief goldmine) | ★★ | ★★★ | ★★ |
| `contextCard` | Card \| None | public | Card the ACTIVATE question is about | ★★ | ★★ | ★ |
| `effect` | Card \| None | public | Card whose effect is resolving (why am I discarding?) | ★★★ | ★★ | ★★ |

## 7. Options (`Option`, tagged union on `OptionType` — 17 kinds)

Populated fields per kind (verbatim from `cg/api.py`; resolution to card
identities implemented in `observation/resolve.py`):

| OptionType | Fields | Meaning | R | S | L |
|---|---|---|---|---|---|
| `NUMBER` | number | Pick a quantity (draw counts, damage counts) | ★★ | ★★ | ★★ |
| `YES` / `NO` | — | Binary answers (mulligan, going first, ACTIVATE, COIN_HEAD) | ★★★ | ★★★ | ★★★ |
| `CARD` | area, index, playerIndex | Select a card in a zone (targets, setup, tutors) | ★★★ | ★★★ | ★★★ |
| `TOOL_CARD` | + toolIndex | Select an attached tool | ★ | ★ | ★ |
| `ENERGY_CARD` | + energyIndex | Select an attached energy card | ★★ | ★★ | ★ |
| `ENERGY` | + count | Select energy units (cost payments) | ★★ | ★★ | ★ |
| `PLAY` | index (hand) | Play a card from hand | ★★★ | ★★★ | ★★★ |
| `ATTACH` | area/index → inPlayArea/inPlayIndex | Attach energy/tool to a Pokémon | ★★★ | ★★★ | ★★★ |
| `EVOLVE` | area/index → inPlayArea/inPlayIndex | Evolve a Pokémon | ★★★ | ★★★ | ★★★ |
| `ABILITY` | area, index | Use an ability | ★★★ | ★★★ | ★★★ |
| `DISCARD` | area, index | Discard a card in play | ★★ | ★★ | ★★ |
| `RETREAT` | — | Retreat the active | ★★★ | ★★★ | ★★★ |
| `ATTACK` | attackId | Use an attack (join to `all_attack()` for damage/cost/text) | ★★★ | ★★★ | ★★★ |
| `END` | — | End turn (always present in MAIN) | ★★★ | ★★★ | ★★★ |
| `SKILL` | cardId, serial | Order simultaneous effect resolutions | ★ | ★★ | ★ |
| `SPECIAL_CONDITION` | specialConditionType | Choose a condition to apply/recover | ★ | ★★ | ★ |

## 8. Event log (`Log`, 24 `LogType`s — may grow)

Per-viewer stream; fields vary by type (see `cg/api.py` for the exact field
sets). Grouped by research value:

| Group | Types | Visibility | Use | R | S | L |
|---|---|---|---|---|---|---|
| Turn structure | TURN_START, TURN_END | public | Turn accounting (validated: TURN_START count == final turn number) | ★ | ★ | ★★ |
| Own/visible card flow | DRAW, MOVE_CARD, PLAY, ATTACH, EVOLVE, DEVOLVE, SWITCH, CHANGE, MOVE_ATTACHED | public-for-owner | What happened between your decisions; the belief tracker's input for OPPONENT plays (their PLAY/ATTACH/EVOLVE logs are visible with card IDs once cards go public) | ★★ | ★★★ | ★★ |
| Hidden flow | DRAW_REVERSE, MOVE_CARD_REVERSE | opponent's hidden moves | Counts only (hand size deltas, deck movements) — negative information | ★ | ★★★ | ★★ |
| Combat | ATTACK, HP_CHANGE | public | Damage accounting; KO detection (HP_CHANGE + MOVE_CARD to discard) | ★★ | ★★ | ★★★ (reward shaping) |
| Conditions | POISONED…CONFUSED (+isRecover) | public | Status timeline | ★ | ★ | ★ |
| Randomness | SHUFFLE, COIN (head: bool) | public | Coin outcomes (attack effects, sleep checks); shuffles invalidate deck-order beliefs | ★ | ★★★ (chance nodes) | ★★ |
| Meta | HAS_BASIC_POKEMON, RESULT (result, reason) | public | Mulligan reveals (free info about opponent's hand!); game end + reason | ★★ | ★★ | ★★★ |

## 9. Static card knowledge (`CardData` / `Attack` — via `all_card_data()` / `all_attack()`)

Not part of the observation but joined to it by `id`/`attackId`; loaded once
(`cards/database.py`). All **public**.

| Field | Meaning | R | S | L |
|---|---|---|---|---|
| `cardType` | POKEMON / ITEM / TOOL / SUPPORTER / STADIUM / BASIC_ENERGY / SPECIAL_ENERGY | ★★★ | ★★★ | ★★★ |
| `hp`, `weakness`, `resistance`, `retreatCost` | Combat math (weakness ×2, resistance −30 per rulebook) | ★★★ | ★★★ | ★★★ |
| `energyType` | Pokémon's type (weakness matchups) | ★★★ | ★★★ | ★★★ |
| `basic/stage1/stage2`, `evolvesFrom` | Evolution lines | ★★★ | ★★★ | ★★ |
| `ex`, `megaEx` | Multi-prize Pokémon (2 / 3 prizes on KO) — prize-race math | ★★★ | ★★★ | ★★★ |
| `tera` | No damage while benched | ★★ | ★★ | ★ |
| `aceSpec` | Deck-building constraint (≤1) | ★ (deck) | ★ | ★ |
| `skills` (name+text) | Ability TEXT — English prose; machine-usable only via hand-encoding or NLP (a real Phase 2+ gap: ability semantics are NOT structured) | ★★ | ★★ | ★★ |
| `attacks` → `Attack{damage, energies, text}` | Attack cost/base damage structured; EFFECT text is prose (same gap: conditional damage, coin effects live only in text) | ★★★ | ★★★ | ★★★ |

## 10. Hidden information & inference targets (the belief layer's spec)

What is **knowable by bookkeeping** (future `BeliefTracker`, Growth Plan):

| Inferred quantity | From | Value |
|---|---|---|
| Your own remaining deck multiset | your 60 − hand − board − discard − seen prizes | ★★★ for search determinization of YOUR deck |
| Opponent's revealed-cards multiset | their PLAY/ATTACH/EVOLVE logs + discard + board | ★★★ — shrinks their unseen pool |
| Opponent's unseen pool split | unseen = deck + hand + prizes (counts public) | THE determinization problem (`search_begin` needs concrete guesses for all three) |
| Deck-out horizons | deckCount + draw obligations | ★★★ endgame rules |
| Prize-race state | prize lengths + ex/megaEx on board | ★★★ everywhere |
| Opponent archetype/decklist prior | revealed cards vs metagame priors (Q5) | ★★ → ★★★ as meta data accrues |

**Structural blind spots** (no amount of inference recovers these):
own+opponent prize contents (random 6-card subsets), opponent hand identity
beyond reveals, deck ORDER (shuffle-fresh), future coin flips, and the
prose-only semantics of ability/attack effect text (§9).

## 11. Guidance for future phases

- **Rule-based (Phase 2)**: everything ★★★ in R columns is available today
  through `ParsedObservation` + `CardDatabase`; the derived quantities in
  §10 rows 1/4/5 are the first `GameState` fields to implement.
- **Search (post ADR-0010)**: `search_begin_input` + the §10 determinization
  split is the entire integration surface; COIN logs + `manual_coin` handle
  chance nodes.
- **Learning (Phase 3)**: L-★★★ fields define the observation encoding's
  core; `serial` gives entity persistence for set/graph encodings; effect
  TEXT prose is the open encoding problem — start with id-embeddings and
  structured fields only.
