"""Boundary to the vendored competition SDK (ADR-0002).

This package is the ONLY place allowed to import the vendored ``cg`` SDK.
Everything above it consumes raw observation dicts (``RawObs``) or the parsed
models from :mod:`ptcg_ai.observation`.
"""

from ptcg_ai.environment.adapter import BattleEnvironment, BattleOutcome, BattleStartError
from ptcg_ai.environment.sdk import SdkModules, SdkNotAvailableError, load_sdk, sdk_available

__all__ = [
    "BattleEnvironment",
    "BattleOutcome",
    "BattleStartError",
    "SdkModules",
    "SdkNotAvailableError",
    "load_sdk",
    "sdk_available",
]
