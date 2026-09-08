"""End-to-end analysis of an experiment directory (ADR-0012).

Streams episodes one at a time — datasets can exceed RAM; the aggregate
structures cannot.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ptcg_ai.analytics.aggregate import Aggregator
from ptcg_ai.analytics.extractors import extract_episode
from ptcg_ai.analytics.report import write_reports
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.replay.episode import load_episode

logger = logging.getLogger(__name__)


def analyze_experiment(
    experiment_dir: Path, cards: CardDatabase | None = None
) -> dict[str, Any]:
    """Aggregate every episode under ``experiment_dir`` and write reports.

    Returns the analysis dict (also persisted as ``report.json``).
    """
    episodes_dir = experiment_dir / "episodes"
    episode_paths = sorted(
        [*episodes_dir.glob("*.jsonl"), *episodes_dir.glob("*.jsonl.gz")]
    )
    if not episode_paths:
        raise FileNotFoundError(f"No episodes under {episodes_dir}")

    parser = ObservationParser()
    aggregator = Aggregator()
    for index, path in enumerate(episode_paths):
        aggregator.add(extract_episode(load_episode(path), parser))
        if (index + 1) % 200 == 0:
            logger.info("analyze: %d/%d episodes", index + 1, len(episode_paths))

    result = aggregator.result()
    header = _header_from_manifest(experiment_dir)
    write_reports(result, experiment_dir, header=header, cards=cards)
    logger.info("analyze: reports written to %s", experiment_dir)
    return result


def _header_from_manifest(experiment_dir: Path) -> dict[str, Any]:
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.is_file():
        return {"experiment": experiment_dir.name, "note": "no manifest found"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "experiment": manifest.get("name", experiment_dir.name),
        "generating policies": f"{manifest.get('policies', {}).get('a')} vs "
        f"{manifest.get('policies', {}).get('b')} "
        "(random-play caveat: numbers describe the decision space, not competent play)",
        "games": f"{manifest.get('games_completed')} completed, "
        f"{manifest.get('games_aborted')} aborted",
        "git": manifest.get("git_sha", "unknown")[:12],
        "created": manifest.get("created_utc"),
    }
