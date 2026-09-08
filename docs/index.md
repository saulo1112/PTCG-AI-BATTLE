# Documentation Map

| Document | Read when… | Audience |
|---|---|---|
| [handoff.md](handoff.md) | **starting a new session** — current submission state, what's shipped vs experimental, what was tried | everyone |
| [competition_analysis.md](competition_analysis.md) | you need the rules of the game we're playing (Kaggle side) | everyone |
| [sdk_analysis.md](sdk_analysis.md) | you touch anything that talks to the engine | engineers |
| [battle_flow.md](battle_flow.md) | you handle observations/decisions | engineers |
| [architecture.md](architecture.md) | you add or move code | engineers |
| [environment.md](environment.md) | you set up a machine or run battles | everyone |
| [observability.md](observability.md) | you debug "why did it do that?" | engineers |
| [benchmarking.md](benchmarking.md) | before proposing any algorithm | researchers |
| [feature_inventory.md](feature_inventory.md) | you need ANY observable field's meaning/visibility/usefulness — the canonical reference | everyone |
| [methodology.md](methodology.md) | you collect data or read/write any statistic | researchers |
| [game_analysis.md](game_analysis.md) | you want what the data says about the game (synthetic random self-play) | everyone |
| [replay_analysis.md](replay_analysis.md) | you want what real ladder games show (loss patterns, opponent behaviour, metagame decklists) | everyone |
| [state_representation.md](state_representation.md) | you design state/features (Phase 2+) | researchers |
| [decision_system.md](decision_system.md) | you add a policy | researchers |
| [phase2_blueprint.md](phase2_blueprint.md) | you implement the competitive agent (the Phase 2 engineering blueprint) | engineers |
| [training_design.md](training_design.md) | you plan the learning phase (Phase 3) | researchers |
| [roadmap.md](roadmap.md) | you plan the week | everyone |
| [risk_analysis.md](risk_analysis.md) | you make a scope/priority call | everyone |
| [research_questions.md](research_questions.md) | you pick the next experiment | researchers |
| [m6_findings.md](m6_findings.md) | before proposing a new deck or pilot upgrade — what was already tried and why it failed | everyone |
| [developer_guide.md](developer_guide.md) | your first day, and for conventions | engineers |
| [decisions/](decisions/README.md) | you wonder "why is it built this way?" | everyone |

**Status legend** — *living*: updated as facts change (roadmap, risks,
research questions). *Reference*: describes the current system; update with
the code (architecture, decision_system, observability, benchmarking,
developer_guide). *Frozen facts*: change only when the competition/SDK
changes (competition_analysis, sdk_analysis, battle_flow, environment).
*Snapshot*: a dated point-in-time record, superseded rather than edited —
regenerate instead of updating in place (handoff, m6_findings).
