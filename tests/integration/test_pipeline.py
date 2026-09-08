"""End-to-end experiment pipeline: collect → analyze → reports (needs engine)."""

import json
from pathlib import Path

import pytest

from ptcg_ai.analytics.analyze import analyze_experiment
from ptcg_ai.analytics.collect import run_collection
from ptcg_ai.config import AppConfig

pytestmark = pytest.mark.sdk


def test_collect_then_analyze(app_config: AppConfig, tmp_path: Path) -> None:
    experiment_dir = run_collection(
        app_config, name="pipeline-test", games=3, seed=99, out_root=tmp_path
    )

    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["games_completed"] == 3
    assert manifest["games_aborted"] == 0
    assert len(manifest["decks"]["a"]) == 60
    assert "engine_rng_note" in manifest
    episodes = list((experiment_dir / "episodes").glob("*.jsonl.gz"))
    assert len(episodes) == 3

    result = analyze_experiment(experiment_dir)
    assert result["n_games"] == 3
    assert result["n_decisions"] > 0
    assert result["game_length"]["turns"]["n"] == 3  # terminal obs present in all
    assert (experiment_dir / "report.md").is_file()
    assert (experiment_dir / "report.json").is_file()
    assert (experiment_dir / "tables" / "contexts.csv").is_file()

    payload = json.loads((experiment_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["analysis"]["n_games"] == 3
    assert "MAIN/MAIN" in payload["analysis"]["contexts"]
