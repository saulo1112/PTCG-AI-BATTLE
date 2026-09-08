"""Decision tracing (ADR-0009)."""

import json
from pathlib import Path

from ptcg_ai.debug.trace import DecisionTracer, TracingPolicy
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.decision.random_policy import RandomPolicy
from ptcg_ai.observation.parser import ObservationParser
from tests.conftest import make_raw_obs

parser = ObservationParser()


def _ctx() -> DecisionContext:
    raw = make_raw_obs(n_options=3)
    return DecisionContext(raw=raw, observation=parser.parse(raw))


def test_traces_are_recorded_and_transparent() -> None:
    tracer = DecisionTracer()
    traced = TracingPolicy(RandomPolicy(seed=5), tracer)
    action = traced.choose(_ctx())

    assert len(tracer.decisions) == 1
    decision = tracer.decisions[0]
    assert decision.chosen == action
    assert decision.select_type == "MAIN"
    assert decision.n_options == 3
    assert decision.elapsed_ms >= 0.0
    assert decision.policy == "random"
    assert decision.note is not None


def test_jsonl_and_report(tmp_path: Path) -> None:
    tracer = DecisionTracer()
    traced = TracingPolicy(RandomPolicy(seed=5), tracer)
    for _ in range(3):
        traced.choose(_ctx())

    out = tmp_path / "trace.jsonl"
    tracer.save_jsonl(out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["select_type"] == "MAIN"

    report = tracer.report()
    assert "decisions: 3" in report
