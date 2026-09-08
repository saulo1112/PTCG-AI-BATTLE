"""M47 — the gate that decides, generalised over checkpoints.

This is `test_ctx_heads.py` with the candidate as an argument and one row added. It is
the ONLY instrument in this project that has predicted the real ladder: it called M43's
`imitation-final` (+0.096 / +0.117 in two runs) which then delivered +30-40 implied elo
on the ladder (M44), and it correctly killed M46's k=7 (+0.007 once the control was
subtracted) and `bc_alakazam_v2` (+0.010, CI crossing 0.500).

Every row is n=600 by default because M41 measured this engine reading the SAME deck
against ITSELF as 0.435 at n=200 and 0.503 at n=600 -- below n~600 nothing here is
measurable. At 0.54 s/game that is ~5.4 min per row.

THE ROWS, and why each is there:

  1. candidate vs champion, identical deck   the mirror is 0.500 by construction, so
                                             the only difference in the experiment is
                                             the weights file
  2. champion vs champion                    the noise calibrator. If this is not
                                             ~0.500 the run is discarded, not
                                             interpreted
  3. candidate/champion vs Grimmsnarl        25% of the real field and the wall since
                                             M32; M43's candidate improved here and it
                                             showed up on the ladder
  4. candidate/champion vs Mega Lucario      HELD OUT -- 11.5% of the real field and NOT
                                             in the self-play opponent mix. An RL run
                                             that only beats what it trained against
                                             fails here, and nothing else would catch it

Run:
  PYTHONPATH=src python -u scratchpad/m47_arena.py data/models/rl_alakazam_final_iter12.json
  PYTHONPATH=src python -u scratchpad/m47_arena.py <ckpt> --n 600 --skip-control
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from arena_m8 import run  # noqa: E402

CHAMPION = "data/models/bc_alakazam_final.json"
DECK = "decks/yushinito.csv"
GRIMMSNARL = ("decks/luca.csv", "data/models/bc_luca_full.json")
LUCARIO = ("decks/lucario800.csv", "data/models/bc_800_v2.json")   # held out of training


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--champion", default=CHAMPION)
    ap.add_argument("--skip-control", action="store_true",
                    help="reuse a control measured in another run of the same session; "
                         "only for extra checkpoints, never for the first one")
    ap.add_argument("--skip-matchups", action="store_true",
                    help="mirror + control only (11 min instead of 27)")
    args = ap.parse_args()

    if not Path(args.candidate).is_file():
        raise SystemExit(f"candidate not found: {args.candidate}")
    n = args.n
    rows = []

    s = run("imitation", DECK, "imitation", DECK, f"CANDIDATE vs CHAMPION (n={n})",
            n=n, wa=args.candidate, wb=args.champion)
    verdict_mirror = s.wilson_interval()
    rows.append(("candidate vs champion (mirror)", s.score_rate, 0.500, verdict_mirror))

    if not args.skip_control:
        s = run("imitation", DECK, "imitation", DECK, f"CHAMPION vs CHAMPION control (n={n})",
                n=n, wa=args.champion, wb=args.champion)
        rows.append(("champion vs champion (control)", s.score_rate, 0.500, s.wilson_interval()))

    if not args.skip_matchups:
        for label, (odeck, ow) in (("grimmsnarl", GRIMMSNARL), ("lucario [HELD OUT]", LUCARIO)):
            a = run("imitation", DECK, "imitation", odeck, f"CANDIDATE vs {label} (n={n})",
                    n=n, wa=args.candidate, wb=ow)
            b = run("imitation", DECK, "imitation", odeck, f"CHAMPION vs {label} (n={n})",
                    n=n, wa=args.champion, wb=ow)
            rows.append((f"vs {label}: candidate", a.score_rate, b.score_rate,
                         a.wilson_interval()))

    print("\n" + "=" * 78)
    print(f"candidate: {args.candidate}")
    print(f"{'matchup':<34}{'observed':>10}{'reference':>11}{'delta':>10}{'95% CI':>21}")
    print("-" * 78)
    for label, got, ref, (lo, hi) in rows:
        print(f"{label:<34}{got:>10.3f}{ref:>11.3f}{got - ref:>+10.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>21}")
    print("-" * 78)
    lo, hi = verdict_mirror
    if lo > 0.500:
        print(f"MIRROR: CI [{lo:.3f}, {hi:.3f}] is ENTIRELY above 0.500 -> passes the M43 bar")
        print("        (~{:+.0f} elo equivalent)".format(400 * math.log10(
            max(rows[0][1], 1e-6) / max(1 - rows[0][1], 1e-6))))
    else:
        print(f"MIRROR: CI [{lo:.3f}, {hi:.3f}] touches or crosses 0.500 -> does NOT pass")
    print("Check the control row first: if it is not ~0.49-0.52 the mirror row means nothing.")
    print("Check the HELD-OUT lucario row: it is the only overfit-to-the-training-mix probe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
