"""Seeded randomness helpers.

The native engine uses its own non-reproducible RNG for real battles (see
docs/sdk_analysis.md); these helpers only control *our* Python-side choices
(policy sampling, determinization later), keeping runs as reproducible as
the engine allows.
"""

from __future__ import annotations

import random


def make_rng(seed: int | None = None) -> random.Random:
    """Return an isolated ``random.Random``.

    A fresh instance per component avoids hidden coupling through the global
    ``random`` module state.
    """
    return random.Random(seed)


def spawn(rng: random.Random) -> random.Random:
    """Derive an independent child RNG from ``rng`` (for per-battle streams)."""
    return random.Random(rng.getrandbits(64))
