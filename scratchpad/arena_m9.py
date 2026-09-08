"""M9 arena: imitation-v3 (BC of the ~940-elo Bellibolt pilot kenN2439) vs v1.

Stage 1 of the two-stage promotion gate (the head-to-head arena — necessary but
NOT sufficient; the field gauntlet in field_gauntlet.py is the decisive gate).
v3 must beat our current champion imitation-v1. Swapped sides, Wilson 95% CI +
end-reason breakdown. Reuses arena_m8's run() harness verbatim.

Modes (n defaults in parens):
  smoke  v3 vs v1                                       (20)   sanity
  a      imitation-v3 vs imitation-v1     [PROMOTION]   (300)  ship gate: lo>0.5
  b      imitation-v3 vs greedy(940deck)  [PILOT-LIFT]  (200)  same-deck skill lift
  d      imitation-v3 mirror              [SELF-SANITY] (50)   traces
  all    a + b + d
"""

from __future__ import annotations

import sys

from arena_m8 import run

K940 = "decks/kenn2439.csv"
G650 = "decks/greengreenpurple.csv"
W_V1 = "data/models/bc_650_v1.json"
W_V3 = "data/models/bc_940_v1.json"


def _a(n=300):
    run("imitation", K940, "imitation", G650, "A PROMOTION v3(bellibolt940) vs v1(650)", n=n, wa=W_V3, wb=W_V1)

def _b(n=200):
    run("imitation", K940, "greedy", K940, "B PILOT-LIFT v3 vs greedy(kenn2439)", n=n, wa=W_V3)

def _d(n=50):
    run("imitation", K940, "imitation", K940, "D SELF v3 mirror", n=n, wa=W_V3, wb=W_V3)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if mode == "smoke":
        run("imitation", K940, "imitation", G650, "smoke v3 vs v1", n=n or 20, wa=W_V3, wb=W_V1)
    elif mode == "a":
        _a(n or 300)
    elif mode == "b":
        _b(n or 200)
    elif mode == "d":
        _d(n or 50)
    elif mode == "all":
        _a(n or 300); _b(200); _d(50)
    else:
        raise SystemExit(f"unknown mode {mode!r}")
