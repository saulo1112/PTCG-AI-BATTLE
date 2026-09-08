"""M11 arena (gate G2): imitation-v1.2 (MLP MAIN scorer) vs imitation-v1 (linear).

Same greengreenpurple deck both sides — isolates the MLP MAIN scorer. Both are
plain BC (no search), so games run at v1 speed (~0.3 s/game). Reuses arena_m8.run.

Run:  uv run python scratchpad/arena_m11.py [n]
"""

from __future__ import annotations

import sys

from arena_m8 import run

G650 = "decks/greengreenpurple.csv"
W_V1 = "data/models/bc_650_v1.json"
W_V12 = "data/models/bc_650_v2.json"

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    run("imitation", G650, "imitation", G650, "G2 v1.2(MLP) vs v1(linear)", n=n, wa=W_V12, wb=W_V1)
