"""Policy registry."""

import pytest

from ptcg_ai.config import AppConfig
from ptcg_ai.decision.registry import available_policies, build_policy


def test_available_baselines_registered() -> None:
    assert {"random", "safe-random"} <= set(available_policies())


def test_build_random() -> None:
    policy = build_policy(AppConfig(), deck=[3] * 60, name="random")
    assert policy.name == "random"


def test_build_default_from_config() -> None:
    policy = build_policy(AppConfig(), deck=[3] * 60)
    assert policy.name == "safe(random)"


def test_unknown_name_raises_with_choices() -> None:
    with pytest.raises(KeyError, match="Available"):
        build_policy(AppConfig(), name="does-not-exist")
