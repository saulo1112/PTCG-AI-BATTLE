# Competition Analysis

Source of truth: `References/Overview and data.pdf` (Kaggle competition
"The Pokémon Company - PTCG AI Battle Challenge Simulation") and the CABT
engine documentation PDF. Facts below are quoted from those documents;
anything marked **UNKNOWN** is not documented and must be measured.

## Summary

Build an autonomous agent that plays full games of the Pokémon TCG inside
the official **cabt** simulator (kaggle-environments env name `"cabt"`).
Core challenges named by the organizers: probability (shuffles, coins),
hidden information (opponent's hand/deck/prizes), and long-horizon strategic
planning. The organizers explicitly note that pure rule-based programming is
unlikely to reach the top of the leaderboard.

There are two sibling competitions (Simulation and Hackathon); this project
targets the **Simulation** competition only.

## Timeline

| Date (UTC) | Event |
|---|---|
| 2026-06-16 | Start |
| 2026-08-09 | Entry & team-merger deadline |
| **2026-08-16** | **Final submission deadline** |
| 2026-08-17 → ~08-31 | Ladder continues (~2 weeks) until convergence; then final |

As of 2026-07-05 there are ~6 weeks to the final deadline. All deadlines
11:59 PM UTC unless stated otherwise.

## Submission contract

- One `tar.gz`, built with `tar -czvf submission.tar.gz *`, uploaded via
  "My Submissions".
- **`main.py` must sit at the archive top level** and expose
  `agent(obs_dict: dict) -> list[int]`.
- **`deck.csv` must be included**: 60 card IDs, one per line. At runtime the
  agent files are mounted at `/kaggle_simulations/agent/` (our entrypoint
  falls back to that path).
- First call: `obs["select"] is None` → return the 60-card deck.
  Every later call: return `minCount..maxCount` unique indices into
  `obs["select"]["option"]`.
- Built on **kaggle-environments 1.14.10** per the Overview; note the PyPI
  releases actually containing the cabt env are 1.30.x (see
  [environment.md](environment.md)).

In this repo the tarball is produced by `ptcg build-submission` and gated by
`ptcg validate-submission` (ADR-0001) — never hand-built.

## Resource limits

| Resource | Limit |
|---|---|
| Submission size | **197.7 MiB** |
| vCPUs | **2** |
| RAM | **12.2 GiB** |
| Disk | 11.8 GiB |
| Per-move time | **No limit** (`actTimeout = 0`) — confirmed from the `cabt` env spec embedded in real replays (research question Q1, resolved 2026-07-05; reconfirmed on 10 *competitive* episodes 2026-07-05, [replay_analysis.md](replay_analysis.md)) |
| Per-agent time | **~600 s** (`remainingOverageTime`, starts ≈600 and counts down per agent) — **THE compute budget a search policy spends against**; observed across all competitive replays (our safe-random used ≤13 s; the strongest opponent used 13 s) |
| Per-episode time | **2000 s** (`runTimeout`) — whole-match wall-clock ceiling across both agents + the engine; the per-agent 600 s overage is the tighter, per-side bound to budget against |
| Episode steps | 10,000,000 (`episodeSteps`) — no practical cap |
| GPU | Not listed in submission resources (Docker image defines a GPU stage, but no GPU allocation is stated) |

**Q1 evidence**: `Logs/Submission (1) (5-07)/Replay.json` (EpisodeId
84143748, validation episode, result `DONE` / rewards `[1, -1]`) embeds the
env's `configuration` and `specification`: `actTimeout: 0` ("Maximum runtime
(seconds) to obtain an action from an agent"), `runTimeout: 2000`
("Maximum runtime (seconds) of an episode"), no `agentTimeout` field at
all. Observed per-move latency in that episode (random-safe agent):
mean 0.036 ms, max 0.067 ms across 104 move calls — nowhere near any
constraint.

**Implications**: there is no per-move timeout to budget against. The
practical budget a search policy plans against is the **per-agent
`remainingOverageTime`, ~600 s** (starts at ≈600 and counts down for that
agent), sitting inside the 2000 s whole-episode `runTimeout`. Spread over
~50 decisions/episode, that is an effective ~6–12 s per decision — orders of
magnitude above the sub-millisecond `search_begin`/`search_step` measured in
[benchmarking.md](benchmarking.md). This **removes the timeout risk that gated
ADR-0010**; the remaining constraints for search are CPU/RAM (2 vCPUs,
12.2 GiB) and staying under the 600 s per-agent overage, not individual-move
latency.

Caveats: (a) the ~600 s per-agent overage is observed on 10 competitive
episodes plus the validation one ([replay_analysis.md](replay_analysis.md)) —
keep confirming it holds as we climb (Q11); (b) budget against the **600 s
per-agent** figure, not the 2000 s shared episode ceiling. Any learned model
must still fit the submission size limit alongside the bundled `cg/`.

**Opponent-pool intel (μ≈600, 10 replays)**: nobody searches — the strongest
opponent spent 13 s of ~600 s; most < 1 s. Skill is bimodal (random-like vs
bench-developing heuristic). The dominant ladder deck is a lean Mega Lucario
ex item-toolbox; our vendored sample deck (58% energy, no draw engine) is a
consistency liability. Full breakdown and recovered decklists:
[replay_analysis.md](replay_analysis.md).

## Rating system

- TrueSkill-style: each submission's skill is a Gaussian N(μ, σ²), starting
  at **μ₀ = 600**; σ shrinks as episodes accumulate.
- Updates use only **win/loss/draw** — margin of victory is ignored. Never
  optimize for prize differential at the cost of win probability.
- On upload, a **validation episode** (self-play against a copy) must pass,
  or the submission is marked Error (agent logs are downloadable).
- Matchmaking pairs submissions of similar rating; new submissions get a
  temporarily increased episode rate.

## Submission cadence

- **5 submissions per team per day**; only the **most recent 2** are actively
  tracked for final evaluation.
- The leaderboard shows the team's best-scoring agent.
- Strategy consequences:
  - Never burn a slot on an unvalidated artifact — local validation first.
  - Late-competition submissions start at μ₀ = 600 and need episodes to
    climb: plan the final agent's upload with enough runway (the ~2-week
    post-deadline window helps convergence).
  - Keep exactly two strong, *different* agents active when experimenting
    (the last-2 rule).

## Deck rules

Bring your own 60-card deck (`deck.csv`). Card pool = the competition list
(`Card_ID List_EN.pdf` / `EN_Card_Data.csv`, IDs 1..1267 sparse; the
machine-readable source is `all_card_data()`). Engine-enforced deck rules:
max 4 copies of a same-name card (basic Energy exempt), at least one Basic
Pokémon, at most one ACE SPEC. Deck quality is a ranked variable of its own
(roadmap Phase 2+).

## What winning requires

1. **Robustness** — a crash or illegal action forfeits the game; rating
   counts it as a loss (SafePolicy, validation gates).
2. **Play quality** — the decision policy ladder (docs/decision_system.md).
3. **Deck quality** — co-evolves with the policy; a strong deck under a weak
   policy still beats a strong policy on a broken deck at this rating model.
4. **Cadence discipline** — 5/day, last-2-active, and the validation episode
   as a hard gate.
