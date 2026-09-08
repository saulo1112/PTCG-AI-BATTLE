"""Policy-vs-policy matchups over the local SDK host.

This is the promotion gate of the baseline ladder (docs/decision_system.md):
a new rung must beat the previous one here before it earns a submission.
"""

from __future__ import annotations

import logging

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config.schema import AppConfig
from ptcg_ai.decision.base import Policy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.runner import BattleRunner
from ptcg_ai.environment.sdk import load_sdk

logger = logging.getLogger(__name__)


def run_matchup(
    policy_a: Policy,
    policy_b: Policy,
    config: AppConfig,
    n_games: int | None = None,
    swap_sides: bool | None = None,
) -> "MatchStats":
    """Play ``n_games`` between two policies; return stats for ``policy_a``.

    With ``swap_sides`` (default from config) the policies alternate playing
    first/second player to cancel any side advantage.

    Note: both policies play with the decks from ``config.paths`` — deck A
    for ``policy_a``'s side, deck B for the opponent side.
    """
    from ptcg_ai.evaluation.metrics import MatchStats

    n_games = n_games if n_games is not None else config.evaluation.n_games
    swap_sides = swap_sides if swap_sides is not None else config.evaluation.swap_sides

    sdk = load_sdk(config.paths.sdk_dir)
    cards = CardDatabase.from_sdk(sdk)
    deck_a = load_deck(config.paths.deck_path)
    deck_b = load_deck(config.paths.opponent_deck_path)

    agent_a = PTCGAgent(policy_a, deck=deck_a, cards=cards)
    agent_b = PTCGAgent(policy_b, deck=deck_b, cards=cards)

    env = BattleEnvironment(config, sdk=sdk)
    runner = BattleRunner(env, max_decisions=config.battle.max_decisions)
    stats = MatchStats()

    for game in range(n_games):
        a_side = game % 2 if swap_sides else 0
        policy_a.on_battle_start()
        policy_b.on_battle_start()
        if a_side == 0:
            record = runner.run(agent_a, agent_b, deck_a.as_list(), deck_b.as_list())
        else:
            record = runner.run(agent_b, agent_a, deck_b.as_list(), deck_a.as_list())
        policy_a.on_battle_end(record.outcome)
        policy_b.on_battle_end(record.outcome)
        stats.add(record.outcome.winner, a_played_as=a_side)
        logger.info(
            "arena game %d/%d: winner=%s (%s as P%d) — %s",
            game + 1, n_games, record.outcome.winner, policy_a.name, a_side, stats.summary(),
        )
    return stats
