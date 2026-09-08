"""Analytics subsystem (Phase 1): answer game questions with data.

Pipeline: stored episodes (ADR-0011) → :mod:`extractors` (per-episode
records, streaming) → :mod:`aggregate` (accumulators + CIs) →
:mod:`report` (JSON / CSV / Markdown). Collection and end-to-end analysis
live in :mod:`collect` and :mod:`analyze` (ADR-0012).

Conventions (event counting, zone sampling, CI methodology) are normative in
docs/methodology.md — code comments here reference, not redefine, them.
"""

from ptcg_ai.analytics.aggregate import Aggregator
from ptcg_ai.analytics.analyze import analyze_experiment
from ptcg_ai.analytics.collect import run_collection
from ptcg_ai.analytics.extractors import EpisodeExtract, extract_episode

__all__ = [
    "Aggregator",
    "EpisodeExtract",
    "analyze_experiment",
    "extract_episode",
    "run_collection",
]
