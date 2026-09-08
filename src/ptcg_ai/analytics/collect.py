"""Batch data collection: N games → experiment directory (ADR-0012).

Fault-tolerant by design: one pathological game must never kill a 10k-game
run — failures are logged, counted in the manifest, and skipped.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ptcg_ai
from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config.schema import AppConfig
from ptcg_ai.decision.registry import build_policy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.runner import BattleRunner
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.replay.episode import save_episode
from ptcg_ai.replay.recorder import EpisodeRecorder
from ptcg_ai.utils.paths import repo_root

logger = logging.getLogger(__name__)


def run_collection(
    config: AppConfig,
    name: str,
    games: int,
    policy_name: str | None = None,
    opponent_name: str | None = None,
    seed: int | None = None,
    out_root: Path | None = None,
) -> Path:
    """Run ``games`` battles and persist them as an experiment directory.

    Returns the experiment directory path. Policy seeds derive from ``seed``
    (per-game offsets); the engine's own battle RNG is NOT seedable — see
    the reproducibility contract in ADR-0012.
    """
    out_root = out_root if out_root is not None else repo_root() / "data" / "experiments"
    experiment_dir = out_root / name
    episodes_dir = experiment_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    policy_name = policy_name if policy_name is not None else config.policy.name
    opponent_name = opponent_name if opponent_name is not None else policy_name
    seed_base = seed if seed is not None else 1

    sdk = load_sdk(config.paths.sdk_dir)
    deck_a = load_deck(config.paths.deck_path)
    deck_b = load_deck(config.paths.opponent_deck_path)
    # Card-aware policies (rung 3+) need the static card database at decision
    # time; without it a GreedyPolicy silently degrades (can't tell a Basic
    # from an Item, can't do KO math). Random baselines ignore it, so Phase 1
    # datasets are unaffected.
    cards = CardDatabase.from_sdk(sdk)
    env = BattleEnvironment(config, sdk=sdk)
    runner = BattleRunner(env, max_decisions=config.battle.max_decisions)

    started = datetime.now(timezone.utc)
    completed = 0
    aborted = 0
    t0 = time.perf_counter()
    for game in range(games):
        game_seed = seed_base + 2 * game
        policy_a = build_policy(config, deck=deck_a.as_list(), name=policy_name)
        policy_b = build_policy(config, deck=deck_b.as_list(), name=opponent_name)
        _seed_policy(policy_a, game_seed)
        _seed_policy(policy_b, game_seed + 1)
        agent_a = PTCGAgent(policy_a, deck=deck_a, cards=cards)
        agent_b = PTCGAgent(policy_b, deck=deck_b, cards=cards)
        recorder = EpisodeRecorder(
            metadata={
                "game": game,
                "policy_a": policy_name,
                "policy_b": opponent_name,
                "policy_seed_a": game_seed,
                "policy_seed_b": game_seed + 1,
            }
        )
        try:
            record = runner.run(
                agent_a, agent_b, deck_a.as_list(), deck_b.as_list(), recorder=recorder
            )
        except Exception:
            logger.exception("Game %d failed; continuing.", game)
            aborted += 1
            env.close()  # ensure the engine handle is released for the next game
            continue
        assert record.episode is not None
        save_episode(record.episode, episodes_dir / f"{game:05d}.jsonl.gz")
        completed += 1
        if (game + 1) % 100 == 0:
            logger.info("collect %s: %d/%d games", name, game + 1, games)

    manifest = {
        "name": name,
        "created_utc": started.isoformat(),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "git_sha": _git_sha(),
        "package_version": ptcg_ai.__version__,
        "profile": config.profile,
        "config": _config_snapshot(config),
        "policies": {"a": policy_name, "b": opponent_name},
        "policy_seed_base": seed_base,
        "engine_rng_note": "battle RNG is std::random_device; games are NOT reproducible from seeds (ADR-0012)",
        "decks": {"a": deck_a.as_list(), "b": deck_b.as_list()},
        "games_requested": games,
        "games_completed": completed,
        "games_aborted": aborted,
        "platform": platform.platform(),
    }
    (experiment_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    logger.info(
        "collect %s done: %d completed, %d aborted -> %s", name, completed, aborted, experiment_dir
    )
    return experiment_dir


def _seed_policy(policy: Any, seed: int) -> None:
    """Best-effort reseeding of a registry-built policy for per-game variety."""
    from ptcg_ai.utils.seeding import make_rng

    for target in (policy, getattr(policy, "_inner", None), getattr(policy, "_fallback", None)):
        if target is not None and hasattr(target, "_rng"):
            target._rng = make_rng(seed)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=repo_root(), timeout=10,
        ).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _config_snapshot(config: AppConfig) -> Any:
    def encode(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: encode(v) for k, v in obj.items()}
        return obj

    return encode(config)
