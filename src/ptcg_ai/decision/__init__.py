"""The decision layer: one universal Policy interface (ADR-0004).

Baseline ladder (docs/decision_system.md): Random → Random+Safety → Greedy →
Rule-Based → Rule-Based+Search → Learning. Only the first two rungs exist;
further rungs are added per the roadmap, gated by ADR-0006.
"""

from ptcg_ai.decision.base import BasePolicy, DecisionContext, Policy
from ptcg_ai.decision.random_policy import RandomPolicy
from ptcg_ai.decision.registry import build_policy, register_policy
from ptcg_ai.decision.safety import SafePolicy

__all__ = [
    "BasePolicy",
    "DecisionContext",
    "Policy",
    "RandomPolicy",
    "SafePolicy",
    "build_policy",
    "register_policy",
]
