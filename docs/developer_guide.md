# Developer Guide

## Day one

```bash
uv sync                      # deps + editable install
uv run pytest                # 64+ tests; sdk-marked ones auto-skip without the engine
uv run ptcg battle --trace build/trace.jsonl   # watch a traced random game
uv run ptcg show-obs tests/fixtures/observations/004_MAIN_MAIN.json
```

A `Makefile` wraps the common `ptcg` invocations as short targets (`make
battle`, `make bench`, `make submit-check`, …). Run `make help` for the full
list with descriptions; every target accepts overrides, e.g.
`make battle GAMES=10 POLICY=random`. It's a convenience layer only — the
underlying `uv run ptcg ...` commands documented throughout these docs
always work directly.

Read next: [architecture.md](architecture.md), then
[battle_flow.md](battle_flow.md), then the ADR index
([decisions/](decisions/README.md)).

## Repo tour

| Path | What |
|---|---|
| `src/ptcg_ai/` | the research framework (see architecture.md) |
| `tests/` | unit (no SDK) + integration (`@pytest.mark.sdk`) + fixtures |
| `configs/` | YAML profiles: development / benchmark / submission |
| `docs/` | you are here; `docs/decisions/` = ADRs |
| `build/`, `replays/` | generated artifacts, gitignored |
| `pokemon-tcg-ai-battle/`, `References/` | **read-only vendor material** (ADR-0002) — never edit |

## Conventions

- **Types everywhere**; frozen dataclasses for data; `Protocol` for
  extension points (Policy, future FeatureExtractor/Determinizer);
  composition over inheritance.
- **Google-style docstrings** on every public class/function; comments state
  constraints, not narration.
- **Parsed models keep SDK field names verbatim** (camelCase) — deliberate,
  see ADR-0005. Everything else is PEP 8.
- **No `import cg` outside `environment/sdk.py`.** The submission entrypoint
  imports nothing from `ptcg_ai`.
- **No magic numbers** — tunables live in `config/schema.py`; genuine
  constants (deck size, size limit) are named there too.
- Logging: modules use `logging.getLogger(__name__)`; only entry points call
  `setup_logging`.

## Common tasks

**Add a policy (ladder rung)**
1. Implement in `src/ptcg_ai/decision/<name>.py` (subclass `BasePolicy` or
   satisfy the `Policy` protocol). Set `last_note` at non-obvious choices.
2. Register a factory in `decision/registry.py`.
3. Arena-gate it: `run_matchup(new, previous_rung, config)` — promotion
   criterion in [decision_system.md](decision_system.md).
4. Unit-test legality over the fixtures (copy
   `test_random_policy` / `test_fixtures.py` patterns).

**Add a config field** — edit the dataclass in `config/schema.py` (with a
default), use it, optionally override in a profile YAML. Unknown YAML keys
fail loudly by design.

**Regenerate fixtures (after an SDK drop or for more coverage)**
```bash
uv run ptcg capture-fixtures            # new battle, new signatures
uv run pytest tests/unit/test_fixtures.py   # drift check
```
Commit the JSONs; if drift tests fail, extend `observation/models.py` and
document the change in sdk_analysis.md.

**Build & validate a submission**
```bash
uv run ptcg --profile submission build-submission
uv run ptcg validate-submission build/submission.tar.gz
```
Both must pass before any upload (5/day limit — don't waste slots).

**Run benchmarks** — see [benchmarking.md](benchmarking.md).

**Collect and analyze a dataset** (Phase 1 research pipeline)
```bash
uv run ptcg collect --name my-exp --games 1000 --seed 1
uv run ptcg analyze data/experiments/my-exp        # report.md/json + tables/*.csv
uv run ptcg show-episode data/experiments/my-exp/episodes/00000.jsonl.gz --step 12
uv run ptcg battle --verbose --games 1             # watch decisions live, named
```
Conventions (event counting, zone sampling, CIs) are normative in
[methodology.md](methodology.md); findings go to
[game_analysis.md](game_analysis.md) citing the experiment name.

## Testing

- Unit tests must not require the engine; anything that does gets
  `@pytest.mark.sdk` (auto-skipped when the native lib is absent).
- Realistic observations come from committed fixtures (ADR-0007), synthetic
  ones from `tests/conftest.make_raw_obs`.
- One battle per process: never run two SDK tests' battles concurrently in
  one process (pytest's default is fine; watch out with xdist later).

## ADR process

Significant architecture decision → copy
`docs/decisions/0000-adr-template.md` to the next number, fill it, link it
in the index, reference it from code/docs. Changing an accepted ADR means a
new ADR that supersedes it. Next reserved number: **0010 (search
viability)**.
