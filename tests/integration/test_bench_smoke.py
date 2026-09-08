"""Each benchmark must run end-to-end with tiny op counts."""

import pytest

from ptcg_ai.bench.bench_battle import run_battle_bench
from ptcg_ai.bench.bench_parser import run_parser_bench
from ptcg_ai.bench.bench_search import run_search_bench
from ptcg_ai.config import AppConfig
from tests.conftest import OBSERVATIONS_DIR

pytestmark = pytest.mark.sdk


def test_battle_bench(app_config: AppConfig) -> None:
    result = run_battle_bench(app_config, n_battles=1)
    assert result.n_ops == 1
    assert result.ops_per_s > 0
    assert "decisions/battle p50" in result.notes


def test_parser_bench(app_config: AppConfig) -> None:
    if not OBSERVATIONS_DIR.is_dir():
        pytest.skip("fixtures not captured")
    result = run_parser_bench(app_config, OBSERVATIONS_DIR, iterations=2)
    assert result.n_ops > 0
    assert result.ops_per_s > 0


def test_search_bench(app_config: AppConfig) -> None:
    begin, step = run_search_bench(app_config, n_steps=25)
    assert begin.n_ops == 10
    assert step.n_ops > 0
    assert step.p50_ms >= 0
