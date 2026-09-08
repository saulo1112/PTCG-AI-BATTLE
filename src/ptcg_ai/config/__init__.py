"""Centralized configuration (ADR-0003).

Schema and defaults live in :mod:`ptcg_ai.config.schema`; YAML profiles in
``configs/`` overlay selected fields via :func:`ptcg_ai.config.loader.load_config`.
"""

from ptcg_ai.config.loader import load_config
from ptcg_ai.config.schema import AppConfig

__all__ = ["AppConfig", "load_config"]
