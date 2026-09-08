# SDK Analysis

The vendored SDK lives at
`pokemon-tcg-ai-battle/sample_submission/sample_submission/cg/` (read-only,
ADR-0002). It is pure-stdlib Python over ctypes, bridging to the native
cabt engine (`cg.dll` Windows, `libcg.so` Linux x86-64, `libcg-arm64.so`,
`libcg.dylib` macOS — all bundled, so Windows dev and Linux Kaggle both work).

## Package anatomy

| File | Role |
|---|---|
| `cg/sim.py` | ctypes bindings; loads the native lib per-OS at import and calls `GameInitialize()` once; module-global `Battle` state |
| `cg/api.py` | Enums, observation dataclasses, card data API, **search API** |
| `cg/game.py` | Battle host loop: `battle_start` / `battle_select` / `battle_finish` / `visualize_data` |
| `cg/utils.py` | dict → dataclass helpers |

**Process model**: battle state lives in module globals (`cg.sim.Battle`,
plus a lazily created global `agent_ptr` for search). At most **one live
battle per process**; parallel self-play requires multiprocessing.

## Type inventory (mirrored in `ptcg_ai.observation.models`)

Enums (IntEnum, numeric in raw JSON): `AreaType` (12), `EnergyType` (12),
`CardType` (7), `SpecialConditionType` (5), `SelectType` (11),
`SelectContext` (49), `OptionType` (17), `LogType` (24).

Dataclasses: `Observation{select, logs, current, search_begin_input}`,
`SelectData{type, context, minCount, maxCount, remainDamageCounter,
remainEnergyCost, option[], deck?, contextCard?, effect?}`, `Option` (tagged
union on `OptionType`), `State`, `PlayerState`, `Pokemon`, `Card{id, serial,
playerIndex}`, `Log`, plus card metadata `CardData`/`Attack`/`Skill` and
search types `SearchState{observation, searchId}`/`ApiResult`.

Key identity concept: `id` is the card *type* (CardData ID); `serial` is the
unique physical card within a match.

## Forward-compatibility warning (drives ADR-0005)

`cg/api.py` states verbatim:

> Please note that new elements may be appended to the Enum during the
> competition.
> Please note that new attributes may be appended to each class during the
> competition.

Our parser therefore tolerates unknown enum values and unknown dict keys;
`tests/unit/test_fixtures.py::test_no_schema_drift_in_fixtures` is the drift
detector.

## Function surface

Card data (static, load once):
- `all_card_data() -> list[CardData]` — the authoritative card pool
  (~1.2k cards; the CSVs are human reference only).
- `all_attack() -> list[Attack]`.

Battle hosting (local only; wrapped by `ptcg_ai.environment.adapter`):
- `battle_start(deck0, deck1) -> (obs_dict | None, StartData)` — decks are
  60 IDs; on failure `StartData.errorPlayer/errorType` says which deck broke
  which rule (1 unknown card, 2 copy limit, 3 no Basic, 4 ACE SPEC).
- `battle_select(list[int]) -> obs_dict` — advance; error 30 = broken
  handle, other nonzero = illegal selection (`IndexError`).
- `battle_finish()` — free engine memory.
- `visualize_data() -> str` — spectator JSON (full information).

Search API (the lookahead mechanism; used by `bench/bench_search.py` now,
by search policies after ADR-0010):
- `search_begin(agent_observation, your_deck, your_prize, opponent_deck,
  opponent_prize, opponent_hand, opponent_active, manual_coin=False)
  -> SearchState` — forks the true engine state (carried opaquely in
  `observation.search_begin_input`) into a **determinized** copy where the
  caller supplies concrete guesses for all hidden zones. Guessed lists must
  match the hidden counts; IDs are validated against the card table.
  `opponent_active` is required only when the opponent's Active is
  face-down (setup).
- `search_step(search_id, select) -> SearchState` — apply a selection in the
  hypothetical line. Nodes form a **persistent tree**: stepping does not
  invalidate the parent, so branching/backtracking (MCTS-style) is natural.
- `search_release(search_id)` / `search_end()` — free one node / all nodes.

## Randomness and determinism

- **Real battles are not reproducible**: `BattleStart` enables
  `deviceRand`, so shuffles and coins draw from `std::random_device`,
  ignoring any seed.
- **Search mode is deterministic**: `AgentStart` keeps the seeded RNG, and
  `manual_coin=True` turns coin flips into explicit YES/NO decision points
  the searcher controls (`SelectContext.COIN_HEAD`).
- Our Python-side choices are seeded via `utils.seeding` — runs are as
  reproducible as the engine allows.

## Hidden-information discipline

The engine strips hidden data *server-side* before handing over an
observation (opponent hand → `None` + `handCount`; face-down active/prizes →
`None`; opponent draws/moves logged as `DRAW_REVERSE`/`MOVE_CARD_REVERSE`).
The `search_begin_input` blob is likewise erased — a client cannot recover
hidden cards from anything it receives.

## SDK update procedure (ADR-0002)

When the organizers ship a new SDK drop:
1. Replace the vendored folder wholesale (never merge by hand).
2. Run `uv run pytest` — parser/fixture tests flag schema drift.
3. Re-capture fixtures (`uv run ptcg capture-fixtures`), diff them, extend
   `observation/models.py` enums/fields for anything new.
4. Update this document and re-run `ptcg validate-submission`.
Only `environment/` and `observation/` should ever need code changes.

## Licensing

`LicenseRef-PTCG-ABC-Competition-Use-Only`: competition use only, no
redistribution or publication, delete after the competition. Consequences:
the repo stays **private**; the SDK is never copied into `src/`; the C++
engine source (`ptcg_engine/`) is reference material only.
