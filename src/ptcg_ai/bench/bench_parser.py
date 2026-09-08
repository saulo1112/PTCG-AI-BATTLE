"""Parser throughput over captured fixtures.

Answers (docs/benchmarking.md): is observation parsing ever a per-move cost
worth worrying about? (Expected: no — but measured, not assumed.)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ptcg_ai.bench.harness import BenchResult
from ptcg_ai.config.schema import AppConfig
from ptcg_ai.observation.parser import ObservationParser


def run_parser_bench(
    config: AppConfig,
    fixtures_dir: Path,
    iterations: int | None = None,
) -> BenchResult:
    """Parse every fixture ``iterations`` times; one op = one observation."""
    iterations = iterations if iterations is not None else config.bench.parser_iterations
    fixture_paths = sorted(fixtures_dir.glob("*.json"))
    if not fixture_paths:
        raise FileNotFoundError(f"No fixtures in {fixtures_dir}; run `ptcg capture-fixtures`.")
    raw_observations = [json.loads(p.read_text(encoding="utf-8")) for p in fixture_paths]

    parser = ObservationParser()
    durations: list[float] = []
    for _ in range(iterations):
        for raw in raw_observations:
            start = time.perf_counter()
            parser.parse(raw)
            durations.append(time.perf_counter() - start)

    return BenchResult.from_durations(
        "parser (fixtures)",
        durations,
        notes={"fixtures": len(raw_observations), "iterations": iterations},
    )
