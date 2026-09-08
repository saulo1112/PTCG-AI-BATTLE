# Environment Setup & Running Battles

## Local setup

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.10 (pinned in
`.python-version`). Then:

```bash
uv sync            # install package + dev deps into .venv
uv run pytest      # everything should pass (SDK tests run if engine loads)
uv run ptcg --help
```

Platform parity: the vendored SDK bundles native engine builds for Windows,
Linux (x86-64 + arm64), and macOS — local development on Windows exercises
the same engine logic Kaggle runs on Linux (per the official docs, the SDK
"uses the same logic as the Kaggle competition environment").

## SDK path resolution

The vendored SDK directory is a config value
(`paths.sdk_dir`, default: `pokemon-tcg-ai-battle/sample_submission/
sample_submission` resolved from the repo root). Override per profile YAML
if a new SDK drop lands elsewhere. `ptcg_ai.environment.sdk.load_sdk` is the
single load point (ADR-0002); it is idempotent and process-singleton.

## Running battles locally

```bash
# one traced game, random vs random (development profile defaults)
uv run ptcg battle --trace build/trace.jsonl

# several games, explicit policies
uv run ptcg battle --games 5 --policy safe-random --opponent random

# record an episode for replay/debugging
uv run ptcg battle --record build/episode.jsonl

# inspect any captured observation
uv run ptcg show-obs tests/fixtures/observations/004_MAIN_MAIN.json --logs
```

The local host loop is `BattleEnvironment` + `BattleRunner` (SDK adapter) —
fast and programmatic. Remember: **one live battle per process** (SDK
global state); parallel self-play will use multiprocessing (Phase 3).

## Kaggle-fidelity runs (kaggle-environments)

`kaggle-environments` is **not** a project dependency: every release
containing the `cabt` env (1.30.x+) requires Python ≥ 3.11 and
`open_spiel`, which has no Windows wheels. For full-harness fidelity runs
(including the initial deck-submission call and HTML replays):

```bash
# in WSL / Linux with Python 3.11+
python3.11 -m venv ~/.venvs/kaggle-cabt
source ~/.venvs/kaggle-cabt/bin/activate
pip install kaggle-environments
```

then use `ptcg_ai.environment.kaggle_env.run_kaggle_battle(...)`, or the
official snippet:

```python
from kaggle_environments import make
env = make("cabt", configuration={"decks": [deck, deck]})
env.run([agent, agent])
open("replay.html", "w").write(env.render(mode="html"))
```

When to use which host:

| Need | Use |
|---|---|
| Fast iteration, benchmarks, arena, fixtures | SDK adapter (this repo, any OS) |
| Deck-submission call, HTML replay, harness fidelity | kaggle-environments (Linux/WSL venv) |
| Final truth | Kaggle's validation episode after upload |

## Outputs and hygiene

Generated artifacts land in `build/` (tarballs, traces, bench reports) and
`replays/` — both gitignored. The vendored directories
`pokemon-tcg-ai-battle/` and `References/` are read-only: never edit,
move, or reformat anything inside them (ADR-0002).
