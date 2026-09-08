"""Benchmark harness (ADR-0006).

These measurements — not intuition — decide whether tree search or learning
approaches are viable within Kaggle's 2 vCPUs and the (unknown, to be
measured) per-move budget. See docs/benchmarking.md for the decision
framework each number feeds.
"""

from ptcg_ai.bench.harness import BenchResult, format_report, save_report

__all__ = ["BenchResult", "format_report", "save_report"]
