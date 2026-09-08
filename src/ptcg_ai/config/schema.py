"""Configuration schema: frozen dataclasses with defaults in code (ADR-0003).

Every tunable in the project lives here — no magic numbers scattered in
modules. ``AppConfig()`` with no arguments is a fully working development
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ptcg_ai.utils import paths

#: Kaggle hard limit for the submission archive (Overview PDF).
SUBMISSION_SIZE_LIMIT_MIB = 197.7

#: Cards in a legal deck.
DECK_SIZE = 60


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations. Vendored material is read-only (ADR-0002)."""

    sdk_dir: Path = field(default_factory=paths.default_sdk_dir)
    deck_path: Path = field(default_factory=paths.default_deck_path)
    opponent_deck_path: Path = field(default_factory=paths.default_deck_path)
    build_dir: Path = field(default_factory=lambda: paths.repo_root() / "build")
    replay_dir: Path = field(default_factory=lambda: paths.repo_root() / "replays")


@dataclass(frozen=True)
class BattleConfig:
    """Local battle-running options."""

    games: int = 1
    seed: int | None = None
    #: Hard safety cap on decisions per battle (the engine has its own caps;
    #: this only guards our host loop against a hung exchange).
    max_decisions: int = 20_000
    record_episodes: bool = False
    trace: bool = False


@dataclass(frozen=True)
class PolicyConfig:
    """Which policy to build (see ``decision.registry``) and its parameters."""

    name: str = "safe-random"
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationConfig:
    """Arena matchup settings."""

    n_games: int = 20
    swap_sides: bool = True


@dataclass(frozen=True)
class BenchConfig:
    """Benchmark harness settings (ADR-0006)."""

    n_battles: int = 5
    parser_iterations: int = 200
    search_steps: int = 500
    output_dir: Path = field(default_factory=lambda: paths.repo_root() / "build" / "bench")


@dataclass(frozen=True)
class LoggingConfig:
    """Process logging."""

    level: str = "INFO"
    log_dir: Path | None = None


@dataclass(frozen=True)
class SubmissionConfig:
    """Submission tarball build settings (ADR-0001)."""

    output_name: str = "submission"
    size_limit_mib: float = SUBMISSION_SIZE_LIMIT_MIB


@dataclass(frozen=True)
class AppConfig:
    """Root configuration object passed through the framework."""

    profile: str = "development"
    paths: PathsConfig = field(default_factory=PathsConfig)
    battle: BattleConfig = field(default_factory=BattleConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    submission: SubmissionConfig = field(default_factory=SubmissionConfig)
