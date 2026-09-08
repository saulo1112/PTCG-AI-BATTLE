"""YAML profile loader for :class:`~ptcg_ai.config.schema.AppConfig` (ADR-0003).

Precedence (lowest to highest): dataclass defaults ← profile YAML ←
programmatic ``overrides``. Unknown keys raise — typos must not silently
produce default behavior.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import UnionType
from typing import Any, Mapping, Union, get_args, get_origin, get_type_hints

import yaml

from ptcg_ai.config.schema import AppConfig
from ptcg_ai.utils import paths


class ConfigError(ValueError):
    """Raised for unknown keys or malformed profile files."""


def load_config(
    profile: str = "development",
    config_dir: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """Build an :class:`AppConfig` for the named profile.

    Args:
        profile: Profile name; resolves to ``<config_dir>/<profile>.yaml``.
            A missing file is allowed and yields pure defaults (so tests and
            fresh checkouts work without any configs).
        config_dir: Directory holding profile YAMLs; defaults to ``configs/``
            at the repository root.
        overrides: Nested mapping applied last, e.g.
            ``{"battle": {"games": 10}}``.
    """
    config_dir = config_dir if config_dir is not None else paths.configs_dir()
    data: dict[str, Any] = {"profile": profile}

    profile_path = config_dir / f"{profile}.yaml"
    if profile_path.is_file():
        loaded = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{profile_path} must contain a mapping, got {type(loaded).__name__}")
        _deep_merge(data, loaded)

    if overrides:
        _deep_merge(data, dict(overrides))

    return _build_dataclass(AppConfig, data, context="AppConfig")


def _deep_merge(base: dict[str, Any], extra: Mapping[str, Any]) -> None:
    """Recursively merge ``extra`` into ``base`` (in place)."""
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _build_dataclass(cls: type, data: Mapping[str, Any], context: str) -> Any:
    """Recursively construct a (frozen) dataclass from a nested mapping."""
    hints = get_type_hints(cls)
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ConfigError(f"Unknown config key(s) in {context}: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _convert(hints[f.name], data[f.name], f"{context}.{f.name}")
    return cls(**kwargs)


def _convert(annotation: Any, value: Any, context: str) -> Any:
    """Coerce a YAML scalar/mapping to the annotated field type."""
    if value is None:
        return None
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        # e.g. Path | None, int | None — convert against the non-None member.
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _convert(non_none[0], value, context)
        return value
    if annotation is Path or (isinstance(annotation, type) and issubclass(annotation, Path)):
        return Path(str(value)).expanduser()
    if dataclasses.is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{context} must be a mapping, got {type(value).__name__}")
        return _build_dataclass(annotation, value, context)
    return value
