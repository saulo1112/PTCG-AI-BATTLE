# State Representation

Status: the **raw → parsed** stage is implemented; everything past it is
design, gated on Phase 2+ (Growth Plan in
[architecture.md](architecture.md)). No code should be written from this
document until its phase starts.

## Pipeline

```
raw dict  ──ObservationParser──▶  ParsedObservation  ──(Phase 2)──▶  GameState  ──(Phase 3)──▶  features
(engine)      tolerant, extras       typed, verbatim      perspective-        vectors for
              (ADR-0005)             field names          normalized          learned policies
```

Why a parsed layer at all: vendored types are outside our control (ADR-0002)
and the schema grows mid-competition (ADR-0005). Why a *separate* GameState
later: parsed models mirror the wire format (player arrays indexed 0/1,
counts spread across zones); policies want "me vs opponent" views and
derived quantities computed once per decision, not re-derived in every rule.

## Parsed layer (implemented)

See `observation/models.py`. Key properties: SDK field names verbatim;
`extras` catch-all per model; tolerant enums (`UNKNOWN_<n>`); frozen
dataclasses; `ParsedState.me` / `.opponent` convenience views.

## GameState design (Phase 2, with the first heuristic policy)

A frozen snapshot built from `ParsedObservation`, adding only what rules
actually consult (grow additively, driven by real policy needs):

- prize race: prizes left for each side; whether opponent's active is a
  multi-prize Pokémon (ex / mega ex).
- board math per Pokémon: attached energy by type vs attack costs (which
  attacks are one energy away?), damage needed to KO opposing active
  (weakness/resistance applied), retreat affordability.
- resource state: hand size, deck count vs deck-out horizon, energy
  attachment used, supporter used.
- threat summary: opponent's known attackers and their best damage next turn
  (upper bound from card knowledge).

## Hidden information model

What is *knowable* without modeling (pure bookkeeping — the future
`BeliefTracker`):
- own deck contents = own 60 list minus own hand/board/discard/known prizes;
- opponent's *revealed* cards: everything they played, discarded, attached —
  accumulated from `logs` (their hand/deck remainder shrinks accordingly);
- zone counts for both sides (always in the observation).

What must be *estimated*: the split of the opponent's unseen cards between
hand, deck order, and prizes; and their decklist beyond revealed cards
(metagame prior — Q5).

The **Determinizer** contract (consumed by search after ADR-0010):
`sample(state, beliefs, rng) -> concrete hidden-zone assignment` matching
the counts `search_begin` requires. The naive version samples uniformly
from the unseen-card multiset; upgrades are a research track (Q4), not an
architecture change — the contract is stable.

## Feature extraction (Phase 3 design sketch)

`FeatureExtractor.extract(state, ctx) -> vector` for learned policies.
Open design questions live in
[research_questions.md](research_questions.md); candidate encodings (flat
hand-crafted vector vs per-Pokémon set encoding vs card-ID embeddings) are
compared in [training_design.md](training_design.md). Nothing is committed.
