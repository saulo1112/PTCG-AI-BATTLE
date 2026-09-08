"""Config schema and loader (ADR-0003)."""

from pathlib import Path

import pytest

from ptcg_ai.config import AppConfig, load_config
from ptcg_ai.config.loader import ConfigError


def test_defaults_need_no_files(tmp_path: Path) -> None:
    config = load_config(profile="nonexistent", config_dir=tmp_path)
    assert config == AppConfig(profile="nonexistent")
    assert config.policy.name == "safe-random"


def test_profile_yaml_overlays_defaults(tmp_path: Path) -> None:
    (tmp_path / "test.yaml").write_text(
        "battle:\n  games: 7\nlogging:\n  level: DEBUG\n", encoding="utf-8"
    )
    config = load_config(profile="test", config_dir=tmp_path)
    assert config.battle.games == 7
    assert config.logging.level == "DEBUG"
    # Untouched sections keep defaults.
    assert config.evaluation.n_games == AppConfig().evaluation.n_games


def test_overrides_beat_yaml(tmp_path: Path) -> None:
    (tmp_path / "test.yaml").write_text("battle:\n  games: 7\n", encoding="utf-8")
    config = load_config(
        profile="test", config_dir=tmp_path, overrides={"battle": {"games": 9}}
    )
    assert config.battle.games == 9


def test_unknown_key_raises(tmp_path: Path) -> None:
    (tmp_path / "test.yaml").write_text("battle:\n  gmaes: 7\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="gmaes"):
        load_config(profile="test", config_dir=tmp_path)


def test_path_fields_are_converted(tmp_path: Path) -> None:
    (tmp_path / "test.yaml").write_text(
        "paths:\n  deck_path: some/deck.csv\n", encoding="utf-8"
    )
    config = load_config(profile="test", config_dir=tmp_path)
    assert isinstance(config.paths.deck_path, Path)


def test_repo_profiles_load() -> None:
    """The committed profiles must always be loadable."""
    for profile in ("development", "benchmark", "submission"):
        config = load_config(profile=profile)
        assert config.profile == profile
