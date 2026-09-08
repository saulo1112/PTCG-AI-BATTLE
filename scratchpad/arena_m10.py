"""M10 arena: BC-guided search (imitation_search) vs imitation-v1.

Side A = ImitationSearchPolicy (v1's weights as BC seed/fallback + learned V leaf,
determinized search via the SDK api). Side B = imitation-v1 (plain BC). Same deck
both sides (greengreenpurple), so this isolates the search layer's value. Swapped
sides, Wilson 95% CI, plus search stats (searched/fallbacks, s/game).

Run:  uv run python scratchpad/arena_m10.py [n] [greedy_bias]
"""

from __future__ import annotations

import collections
import dataclasses
import sys
import time
from pathlib import Path

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.safety import SafePolicy
from ptcg_ai.decision.search import SearchConfig
from ptcg_ai.decision.search_bc import ImitationSearchPolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.runner import BattleRunner
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.evaluation.metrics import MatchStats
from ptcg_ai.imitation.policy import ImitationPolicy

DECK = "decks/greengreenpurple.csv"
W_V1 = "data/models/bc_650_v1.json"
V_MODEL = "data/models/v_650.json"


def run(n: int, greedy_bias: float):
    cfg = load_config(profile="benchmark")
    config = dataclasses.replace(cfg, paths=dataclasses.replace(
        cfg.paths, deck_path=Path(DECK), opponent_deck_path=Path(DECK)))
    sdk = load_sdk(config.paths.sdk_dir)
    cards = CardDatabase.from_sdk(sdk)
    da = load_deck(Path(DECK))
    dids = da.as_list()

    scfg = SearchConfig(greedy_bias=greedy_bias)
    pol_a = SafePolicy(
        ImitationSearchPolicy(W_V1, V_MODEL, deck=dids, cards=cards, api=sdk.api,
                              search_cfg=scfg, rng_seed=1),
        deck=dids, seed=1)
    pol_b = SafePolicy(ImitationPolicy(W_V1, deck=dids), deck=dids, seed=2)
    agent_a = PTCGAgent(pol_a, deck=da, cards=cards)
    agent_b = PTCGAgent(pol_b, deck=da, cards=cards)
    env = BattleEnvironment(config, sdk=sdk)
    runner = BattleRunner(env, max_decisions=config.battle.max_decisions)
    stats = MatchStats()
    t0 = time.perf_counter()

    for game in range(n):
        a_side = game % 2
        pol_a.on_battle_start(); pol_b.on_battle_start()
        if a_side == 0:
            rec = runner.run(agent_a, agent_b, dids, dids)
        else:
            rec = runner.run(agent_b, agent_a, dids, dids)
        stats.add(rec.outcome.winner, a_played_as=a_side)

    wall = time.perf_counter() - t0
    lo, hi = stats.wilson_interval()
    inner = pol_a._inner
    print(f"\n=== search-BC vs v1 (n={n}, greedy_bias={greedy_bias}, swapped) ===")
    print(f"A[search]: {stats.summary()}   clears0.5? {'YES' if lo > 0.5 else 'no'}")
    print(f"  searched={inner.searched} fallbacks={inner.fallbacks}  "
          f"wall={wall:.0f}s ({wall/n:.1f}s/game)")
    return stats


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    gb = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    run(n, gb)
