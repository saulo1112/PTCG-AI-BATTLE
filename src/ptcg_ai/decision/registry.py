"""Name → policy factory registry.

Configs and the CLI refer to policies by name; new rungs of the baseline
ladder register themselves here as they are implemented.
"""

from __future__ import annotations

from typing import Callable

from ptcg_ai.config.schema import AppConfig, PolicyConfig
from ptcg_ai.decision.base import Policy
from ptcg_ai.decision.random_policy import RandomPolicy
from ptcg_ai.decision.safety import SafePolicy

#: Factory signature: (policy config, app config, deck card IDs) -> Policy.
PolicyFactory = Callable[[PolicyConfig, AppConfig, "list[int] | None"], Policy]

_REGISTRY: dict[str, PolicyFactory] = {}


def register_policy(name: str) -> Callable[[PolicyFactory], PolicyFactory]:
    """Decorator registering a policy factory under ``name``."""

    def decorator(factory: PolicyFactory) -> PolicyFactory:
        if name in _REGISTRY:
            raise ValueError(f"Policy {name!r} is already registered.")
        _REGISTRY[name] = factory
        return factory

    return decorator


def available_policies() -> list[str]:
    return sorted(_REGISTRY)


def build_policy(
    config: AppConfig,
    deck: list[int] | None = None,
    name: str | None = None,
) -> Policy:
    """Build the policy named in ``config.policy`` (or ``name`` override)."""
    policy_name = name if name is not None else config.policy.name
    try:
        factory = _REGISTRY[policy_name]
    except KeyError:
        raise KeyError(
            f"Unknown policy {policy_name!r}. Available: {available_policies()}"
        ) from None
    return factory(config.policy, config, deck)


@register_policy("random")
def _random(policy_cfg: PolicyConfig, config: AppConfig, deck: list[int] | None) -> Policy:
    return RandomPolicy(deck=deck, seed=policy_cfg.params.get("seed", config.battle.seed))


@register_policy("safe-random")
def _safe_random(policy_cfg: PolicyConfig, config: AppConfig, deck: list[int] | None) -> Policy:
    seed = policy_cfg.params.get("seed", config.battle.seed)
    return SafePolicy(RandomPolicy(deck=deck, seed=seed), deck=deck, seed=seed)


@register_policy("greedy")
def _greedy(policy_cfg: PolicyConfig, config: AppConfig, deck: list[int] | None) -> Policy:
    from ptcg_ai.decision.greedy import GreedyPolicy

    return GreedyPolicy(deck=deck)


@register_policy("safe-greedy")
def _safe_greedy(policy_cfg: PolicyConfig, config: AppConfig, deck: list[int] | None) -> Policy:
    from ptcg_ai.decision.greedy import GreedyPolicy

    seed = policy_cfg.params.get("seed", config.battle.seed)
    return SafePolicy(GreedyPolicy(deck=deck), deck=deck, seed=seed)


@register_policy("rule-based")
def _rule_based(policy_cfg: PolicyConfig, config: AppConfig, deck: list[int] | None) -> Policy:
    from ptcg_ai.decision.rule_based import RuleBasedPolicy

    return RuleBasedPolicy(deck=deck)


@register_policy("safe-rule-based")
def _safe_rule_based(policy_cfg: PolicyConfig, config: AppConfig, deck: list[int] | None) -> Policy:
    from ptcg_ai.decision.rule_based import RuleBasedPolicy

    seed = policy_cfg.params.get("seed", config.battle.seed)
    return SafePolicy(RuleBasedPolicy(deck=deck), deck=deck, seed=seed)


def _build_search(config: AppConfig, deck: list[int] | None, seed: int | None):
    """Construct a SearchPolicy wired to the loaded SDK + card database."""
    from ptcg_ai.cards.database import CardDatabase
    from ptcg_ai.decision.search import SearchPolicy
    from ptcg_ai.environment.sdk import load_sdk

    sdk = load_sdk(config.paths.sdk_dir)
    cards = CardDatabase.from_sdk(sdk)
    return SearchPolicy(deck=deck, cards=cards, api=sdk.api, rng_seed=seed)


@register_policy("search")
def _search(policy_cfg: PolicyConfig, config: AppConfig, deck: list[int] | None) -> Policy:
    return _build_search(config, deck, policy_cfg.params.get("seed", config.battle.seed))


@register_policy("safe-search")
def _safe_search(policy_cfg: PolicyConfig, config: AppConfig, deck: list[int] | None) -> Policy:
    seed = policy_cfg.params.get("seed", config.battle.seed)
    return SafePolicy(_build_search(config, deck, seed), deck=deck, seed=seed)
