"""Observability layer (ADR-0009): pretty-printing and decision tracing.

The goal: "why did the agent do that?" must be answerable in seconds —
``ptcg show-obs`` for a snapshot, ``ptcg battle --trace`` for a full match.
"""

from ptcg_ai.debug.inspect import format_observation, format_option, format_select, summarize_logs
from ptcg_ai.debug.trace import DecisionTracer, TracedDecision, TracingPolicy

__all__ = [
    "DecisionTracer",
    "TracedDecision",
    "TracingPolicy",
    "format_observation",
    "format_option",
    "format_select",
    "summarize_logs",
]
