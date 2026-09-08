"""M42 G-A3-bis — candidate vs champion, clone-piloted, at real sample size.

`head_to_head_par.py` runs against a GREEDY-piloted field and saturates at 0.95-0.97
(M37: 67 of 72 decks at 100%), so it can veto but cannot resolve a small gain. This
is the complementary instrument: the two payloads play EACH OTHER on the identical
deck, so the only difference in the entire experiment is the weights file, and the
mirror is 0.500 by construction — which doubles as a free noise calibrator (M41: the
same deck against itself read 0.435 at n=200 and 0.503 at n=600, so anything below
n~600 in this engine is not measurable).

Also runs both payloads against the Grimmsnarl clone (luca), which is 30.8% of our
real ladder field and the matchup M32 identified as the wall.

Run:  PYTHONPATH=src python scratchpad/test_ctx_heads.py [n]
"""

from __future__ import annotations

import sys

from arena_m8 import run

CHAMPION = "data/models/bc_alakazam_final.json"
CANDIDATE = "data/models/bc_alakazam_k7_full.json"
DECK = "decks/yushinito.csv"
GRIMMSNARL = ("decks/luca.csv", "data/models/bc_luca_full.json")


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    rows = []

    # 1. The decisive one: candidate vs champion, identical deck, sides swapped.
    s = run("imitation", DECK, "imitation", DECK, f"CANDIDATE vs CHAMPION (n={n})",
            n=n, wa=CANDIDATE, wb=CHAMPION)
    rows.append(("candidate vs champion (mirror)", s.score_rate, 0.500))

    # 2. Noise calibrator: champion vs itself MUST read ~0.500.
    s = run("imitation", DECK, "imitation", DECK, f"CHAMPION vs CHAMPION control (n={n})",
            n=n, wa=CHAMPION, wb=CHAMPION)
    rows.append(("champion vs champion (control)", s.score_rate, 0.500))

    # 3. Both against the real wall.
    a = run("imitation", DECK, "imitation", GRIMMSNARL[0], f"CANDIDATE vs grimmsnarl (n={n})",
            n=n, wa=CANDIDATE, wb=GRIMMSNARL[1])
    b = run("imitation", DECK, "imitation", GRIMMSNARL[0], f"CHAMPION vs grimmsnarl (n={n})",
            n=n, wa=CHAMPION, wb=GRIMMSNARL[1])
    rows.append(("vs grimmsnarl: candidate", a.score_rate, b.score_rate))

    print("\n" + "=" * 68)
    print(f"{'matchup':<34}{'observed':>11}{'reference':>12}{'delta':>11}")
    print("-" * 68)
    for label, got, ref in rows:
        print(f"{label:<34}{got:>11.3f}{ref:>12.3f}{got-ref:>+11.3f}")
    print("-" * 68)
    print("Reference for rows 1-2 is 0.500 (mirror by construction). Row 2 is the")
    print("noise floor: if it is not ~0.500, row 1's delta is not trustworthy either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
