"""Episode data model and JSONL persistence (ADR-0008, format v2 per ADR-0011).

Episodes store RAW observation dicts plus chosen actions so that replays
survive any evolution of our parsed models. File layout (one JSON object per
line): a ``meta`` line (with ``version``), ``step`` lines in order, an
optional ``final`` line carrying the terminal observation, and a closing
``outcome`` line. Files ending in ``.gz`` are transparently gzip-compressed.

v1 files (no version / no final line / no per-step timing) remain loadable;
missing fields default to ``None``.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, IO, Mapping

#: Current on-disk format version (ADR-0011).
EPISODE_FORMAT_VERSION = 2


@dataclass(frozen=True)
class Step:
    """One decision: the observation shown and the action returned.

    ``elapsed_ms`` is the runner-measured wall time of the agent call
    (parse overhead included); ``None`` in v1 files or unrecorded steps.
    """

    raw_obs: Mapping[str, Any]
    action: list[int]
    player: int
    elapsed_ms: float | None = None


@dataclass
class Episode:
    """A full battle from the host's perspective (both players' decisions).

    ``final_obs`` is the terminal observation (``current.result != -1``).
    It is NOT a decision — its ``select`` is stale (see docs/battle_flow.md).
    """

    steps: list[Step] = field(default_factory=list)
    final_obs: Mapping[str, Any] | None = None
    outcome: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _open(path: Path, mode: str) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def save_episode(episode: Episode, path: Path) -> None:
    """Write an episode as (optionally gzipped) JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open(path, "w") as fh:
        meta = {
            "kind": "meta",
            "version": EPISODE_FORMAT_VERSION,
            "metadata": dict(episode.metadata),
        }
        fh.write(json.dumps(meta) + "\n")
        for step in episode.steps:
            record: dict[str, Any] = {
                "kind": "step",
                "player": step.player,
                "action": step.action,
                "raw_obs": step.raw_obs,
            }
            if step.elapsed_ms is not None:
                record["elapsed_ms"] = step.elapsed_ms
            fh.write(json.dumps(record) + "\n")
        if episode.final_obs is not None:
            fh.write(json.dumps({"kind": "final", "raw_obs": episode.final_obs}) + "\n")
        fh.write(json.dumps({"kind": "outcome", "outcome": episode.outcome}) + "\n")


def load_episode(path: Path) -> Episode:
    """Read an episode written by :func:`save_episode` (v1 or v2)."""
    episode = Episode()
    with _open(path, "r") as fh:
        for line in fh:
            record = json.loads(line)
            kind = record.get("kind")
            if kind == "meta":
                episode.metadata = record.get("metadata", {})
            elif kind == "step":
                episode.steps.append(
                    Step(
                        raw_obs=record["raw_obs"],
                        action=list(record["action"]),
                        player=int(record["player"]),
                        elapsed_ms=record.get("elapsed_ms"),
                    )
                )
            elif kind == "final":
                episode.final_obs = record["raw_obs"]
            elif kind == "outcome":
                episode.outcome = record.get("outcome")
    return episode
