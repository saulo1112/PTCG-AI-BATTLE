"""Is the fresh field even DISCRIMINATIVE? Run the baseline over all of it, cheaply.

If the shipped MLP already wins ~100% against every greedy-piloted deck in the field,
then no candidate can score better and head_to_head cannot veto anything -- the M11
"saturated gauntlet" failure, which cost that milestone its conclusion. Two games per
deck is enough to see saturation; it is not enough to rank anything, and is not used for
ranking here.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from field_gauntlet import _cards, extract_field
from head_to_head_par import _grimmsnarl_decks, _run_parallel

W = "data/models/bc_alakazam_fetch.json"


def main():
    sdk, cards = _cards()
    field = extract_field(cards)
    grim = _grimmsnarl_decks(field)
    t = time.perf_counter()
    res = _run_parallel("mlp-fetch", "imitation", "decks/yushinito.csv", W, field, 2, 4)
    print(f"\n{len(res)} decks in {time.perf_counter() - t:.0f}s")

    vals = sorted(res.values())
    n = len(vals)
    perfect = sum(1 for v in vals if v >= 0.999)
    losing = sum(1 for v in vals if v < 0.5)
    print(f"macro WR        : {sum(vals) / n:.3f}")
    print(f"decks at 100%   : {perfect}/{n}  ({100 * perfect / n:.0f}%)  <- headroom is 1-this")
    print(f"decks below 50% : {losing}/{n}")
    gv = [res[k] for k in res if k in grim]
    if gv:
        print(f"Grimmsnarl slice: {sum(gv) / len(gv):.3f} over {len(gv)} decks")
    print("\nhardest decks:")
    for k, v in sorted(res.items(), key=lambda kv: kv[1])[:8]:
        print(f"  {v:.2f}  {k}{'  [Grimmsnarl]' if k in grim else ''}")


if __name__ == "__main__":
    main()
