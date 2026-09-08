# ADR-0014: Bundle a stdlib agent package with a safe-random fallback

- **Status**: Accepted
- **Date**: 2026-07-05

## Context

The submission tarball currently ships `main.py` (stdlib-only, copied from
`submission/_entrypoint.py`), `deck.csv`, and the vendored `cg/` package
(builder.py). Per ADR-0001, the entrypoint imports **nothing** from the
`ptcg_ai` research stack, and the Phase-0 agent is inline safe-random. That was
correct while the agent had no strategy. Phase 2 (ADR-0013) puts the strategy
in `GameState` + context handlers + a linear evaluator — code that must run on
Kaggle but currently has no way to reach the submission. Kaggle mounts the
agent files at `/kaggle_simulations/agent/`; the runtime has stdlib + whatever
we bundle; a crash or illegal action forfeits the game (R1), and rating is
win/loss only.

## Problem

How does the Phase-2 decision code reach the Kaggle runtime without importing
the dev-only research package (with its config/yaml/pytest dependencies) and
without weakening the never-crash guarantee?

## Alternatives

1. **Ship the whole `ptcg_ai` package** — simplest to wire; drags in
   non-stdlib deps (pyyaml, config machinery) that may be absent or add size,
   and couples the submission to dev-only code paths. Rejected.
2. **Inline all strategy into `main.py`** — one giant stdlib file; no import
   risk, but unmaintainable and untestable, and diverges from the research
   code it mirrors. Rejected.
3. **Bundle a curated stdlib-only subset as `agentpkg/`** — the decision code
   (parsed models, card statics, `GameState`, handlers, evaluator; rung 5 also
   uses the already-bundled `cg/`), imported by `main.py` behind a fallback.

## Decision

Adopt alternative 3. Build a stdlib-only `agentpkg/` (no config/yaml/pytest
imports) and add it to the tarball alongside `main.py`, `deck.csv`, `cg/`.
`main.py` prepends its own directory to `sys.path`, imports `agentpkg` **inside
try/except**, and delegates to it; on **any** import-time or per-decision
exception it falls back to the current inline safe-random logic. This amends
ADR-0001: the entrypoint may import the *shipped* `agentpkg`, but still never
the dev-only `ptcg_ai`. `validate-submission` gains a check that the staged
tree imports with only `cg/` + stdlib on `sys.path` (catching a stray dev-only
import before upload), in addition to the existing size and smoke gates.

## Justification

The fallback makes play-strength strictly dominant over safety: the worst case
of a packaging or runtime bug is the current safe-random agent, never a
forfeit — R1 is preserved unconditionally. A curated stdlib subset keeps the
197.7 MiB budget and the Kaggle runtime clean, stays unit-testable, and mirrors
the research code so the two don't drift. This is milestone M1 in
[phase2_blueprint.md](../phase2_blueprint.md) and a prerequisite for every
rating gain, so it is sequenced first.

## Implementation note (2026-07-06)

Shipped as milestone M1. One deviation from the sketch above, made to
eliminate drift: instead of a renamed `agentpkg/` (which would require
rewriting every `ptcg_ai.` import), the builder ships a **pruned `ptcg_ai/`
subtree copied verbatim** — only the 7 modules the greedy path imports
(`observation/{models,parser,resolve}`, `cards/database`, `decision/{base,
greedy}`, `state/game_state`) with empty package `__init__.py` files, so the
shipped code is byte-identical to the tested code and no import rewriting is
needed. The config/yaml/environment/registry packages are simply not copied.
Card knowledge travels as `card_data.json` (`CardDatabase.to_records` at build,
`from_records` at runtime) so the agent needs no native engine. One enabling
change to the research code: `cards/database.py`'s `SdkModules` import became
`TYPE_CHECKING`-only, which is why importing the shipped `cards.database`
pulls in no `environment`/`cg`. The greedy entrypoint keeps the bundled `cg/`
available for future search rungs but does not import it.

## Hotfix (2026-07-06): Kaggle exec-loads `main.py` with no `__file__`

The first upload of the M1 artifact (`greedy-v1.tar.gz`) **failed Kaggle's
validation episode**. `Logs/Submission 2 - (06-07)/84297644-{0,1}.json` show
both self-play sides crashing at module load:

```
File "/kaggle_simulations/agent/main.py", line 24, in <module>
    _HERE = os.path.dirname(os.path.abspath(__file__))
NameError: name '__file__' is not defined
```

**Root cause**: `kaggle_environments.agent.get_last_callable` loads the agent
via `exec(code_object, env)`, and `env` has no `__file__` key — unlike a normal
`import`. The entrypoint computed `_HERE` from `__file__` as a bare
module-level statement (outside any try/except), so it crashed the whole
module before `agent()` (and therefore the safe-random fallback) ever became
callable. Confirmed by reproducing Kaggle's exact loader
(`exec(source, {"__name__": "__main__"})`, no `__file__` key) against the
extracted `greedy-v1` bundle: identical `NameError`.

**Why local validation missed it**: `smoke_test_entrypoint`'s original script
did `import main`, and a real `import` *does* define `__file__` — so the smoke
test exercised a different loading mechanism than Kaggle's and passed while
Kaggle failed.

**Fix**:
1. `_entrypoint.py` guards the `__file__` lookup: `try/except NameError`,
   falling back to `_KAGGLE_AGENT_DIR`. Standing rule: **zero unguarded
   module-level code** in the entrypoint — anything that can raise must be
   inside `_build_greedy()`'s existing try/except, or itself guarded.
   `_build_greedy()` also adds `os.getcwd()` to its `sys.path` candidates so
   path discovery doesn't depend on `_HERE` being meaningful.
2. `submission/validate.py`'s `_SMOKE_SCRIPT` now loads `main.py` the same way
   Kaggle does — `exec(compile(source, "main.py", "exec"), {"__name__":
   "__main__"})`, no `__file__` — instead of `import main`. This closes the
   local/Kaggle loading-mechanism gap that let the bug through.
3. `tests/unit/test_entrypoint_exec_load.py` adds a fast, SDK-free regression
   test that execs the entrypoint *source* the same way, independent of any
   tarball build.

This does not change the packaging design in the Decision section above —
only how the entrypoint is loaded, which the smoke test now mirrors exactly.

## Consequences

- `submission/builder.py` stages `agentpkg/`; `_entrypoint.py` grows the
  `sys.path` + try/except + delegation shim while keeping its inline
  safe-random as the except-branch.
- `submission/validate.py` gains an isolated-import check (import `agentpkg`
  with only `cg/` + stdlib visible).
- Adds risk R16 (bundled-import failure → silent degrade to safe-random),
  mitigated by the fallback itself + the validation check; a live fallback is
  logged.
- The two artifacts (ADR-0001) remain distinct: `agentpkg/` is *derived from*
  `ptcg_ai` but shipped as its own stdlib tree; nothing in the tarball imports
  the research stack.
- Rung 5 relies on importing the bundled `cg/` at runtime (native `libcg.so`
  present for Linux Kaggle) — already shipped, no new bundling.
