"""M41 — does 2x Enhanced Hammer -> 3x Night Stretcher help, BEFORE spending a submission?

Grounded in two measurements made this session, not guesswork:
  * M40: losing an Abra/Kadabra copy pre-assembly costs 1.71 turns (real mechanism), and
    the FROZEN policy already plays Night Stretcher/Sacred Ash about as well as Yushin
    does when it's in hand and legal (28.4% vs 21.8%) -- the bottleneck isn't play skill,
    it's DRAW probability (1 copy in 60 cards).
  * The hypergeometric math: 3 copies vs 1 roughly triples P(drawn by turn 5) (46% vs 18%).
  * Enhanced Hammer is empirically the LEAST-played card in the whole 60-card list (0.17
    plays per copy per game across 400 of Yushin's real replays) -- the cheapest card to
    give up two copies of.

This is NOT another offline fidelity proxy (eight of those have already failed to predict
ladder outcome this project). It plays REAL simulated games with the real engine and the
SAME frozen weights on both decks -- only deck.csv changes, nothing is retrained. That
makes it the same class of instrument as `field_gauntlet.py`/`arena_m8.py`, which this
project treats as the actual arbiter (a veto, not a promoter -- M25/M14: it can fail to
resolve SMALL true differences, so a null result here does not prove the idea is dead,
but a real signal here is real evidence, not a proxy that has already lied to us before).

Run:  PYTHONPATH=src python scratchpad/test_deck_variant.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import arena_m8

WEIGHTS = "data/models/bc_alakazam_fetch.json"
OLD_DECK = "decks/yushinito.csv"
NEW_DECK = "decks/yushinito_v3stretcher.csv"

OPPONENTS = {
    "mirror":     ("decks/yushinito.csv", WEIGHTS),
    "grimmsnarl": ("decks/luca.csv", "data/models/bc_luca_full.json"),
}

# n=200 showed the OLD deck's own score against the SAME opponent swinging up to 6.5pp
# between two independent runs (grimmsnarl 65.5%->67.0%, kangaskhan 32.0%->25.5%) --
# noise of that size is comparable to the deltas being measured. n=600 tightens the CI
# by ~sqrt(3)x. Kangaskhan is dropped: it's ~6% of the real field (vs ~31%/~25% for
# grimmsnarl/mirror) and its risk already shrank a lot going from a 2-copy to a 1-copy
# Enhanced Hammer cut, so spending time re-confirming it is lower priority than getting
# a trustworthy read on the two matchups that actually move the needle.
N = 600


def main() -> int:
    results = {}
    for name, (odeck, ow) in OPPONENTS.items():
        old = arena_m8.run("imitation", OLD_DECK, "imitation", odeck,
                           f"OLD vs {name}", n=N, wa=WEIGHTS, wb=ow)
        new = arena_m8.run("imitation", NEW_DECK, "imitation", odeck,
                           f"NEW vs {name}", n=N, wa=WEIGHTS, wb=ow)
        results[name] = (old.score_rate, new.score_rate)

    print("\n" + "=" * 60)
    print(f"{'oponente':<14}{'mazo viejo':>12}{'mazo nuevo':>12}{'delta':>10}")
    print("-" * 60)
    deltas = []
    for name, (o, n) in results.items():
        d = n - o
        deltas.append(d)
        print(f"{name:<14}{o:>12.3f}{n:>12.3f}{d:>+10.3f}")
    pooled = sum(deltas) / len(deltas)
    print("-" * 60)
    print(f"{'POOLED':<14}{'':<12}{'':<12}{pooled:>+10.3f}")
    print(f"\n(n={N} por oponente por mazo; esto es un veto, no una promesa -- una señal")
    print(" positiva real justifica subirlo, un resultado plano no descarta la idea del")
    print(" todo, solo dice que el gauntlet no la resuelve a este tamaño de muestra)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
