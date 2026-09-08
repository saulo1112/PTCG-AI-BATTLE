"""Battle throughput: how fast can we generate self-play games?

Answers (docs/benchmarking.md): expected decisions per battle (drives search
budgets), battles/hour for future self-play data generation.
"""

from __future__ import annotations

import time

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.bench.harness import BenchResult
from ptcg_ai.config.schema import AppConfig
from ptcg_ai.decision.random_policy import RandomPolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.runner import BattleRunner
from ptcg_ai.environment.sdk import load_sdk


def run_battle_bench(config: AppConfig, n_battles: int | None = None) -> BenchResult:
    """Time ``n_battles`` random-vs-random games through the full stack
    (adapter → parser → policy → runner)."""
    n_battles = n_battles if n_battles is not None else config.bench.n_battles
    sdk = load_sdk(config.paths.sdk_dir)
    deck = load_deck(config.paths.deck_path)
    env = BattleEnvironment(config, sdk=sdk)
    runner = BattleRunner(env, max_decisions=config.battle.max_decisions)

    durations: list[float] = []
    decision_counts: list[int] = []
    for i in range(n_battles):
        agent0 = PTCGAgent(RandomPolicy(seed=1000 + i))
        agent1 = PTCGAgent(RandomPolicy(seed=2000 + i))
        start = time.perf_counter()
        record = runner.run(agent0, agent1, deck.as_list(), deck.as_list())
        durations.append(time.perf_counter() - start)
        decision_counts.append(record.decisions)

    decision_counts.sort()
    return BenchResult.from_durations(
        "battle (random vs random)",
        durations,
        notes={
            "decisions/battle p50": decision_counts[len(decision_counts) // 2],
            "decisions/battle max": decision_counts[-1],
            "decisions/s": round(sum(decision_counts) / sum(durations), 1),
        },
    )
