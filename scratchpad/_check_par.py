"""Throwaway: does splitting the field across processes change any number?

Windows uses spawn, so this MUST be a real importable module -- a heredoc piped to
stdin has no importable __main__ and the pool dies with BrokenProcessPool.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from field_gauntlet import _cards, extract_field, run_candidate
from head_to_head_par import _run_parallel

DECK = "decks/yushinito.csv"
W = "data/models/bc_alakazam_fetch.json"


def main():
    sdk, cards = _cards()
    field = extract_field(cards)[:8]

    t = time.perf_counter()
    ser = run_candidate("serial", "imitation", DECK, W, field, 2, sdk, cards)
    t_ser = time.perf_counter() - t

    t = time.perf_counter()
    par = _run_parallel("par", "imitation", DECK, W, field, 2, 4)
    t_par = time.perf_counter() - t

    print(f"\nserial {t_ser:.0f}s   paralelo {t_par:.0f}s   -> {t_ser / max(t_par, 1e-9):.1f}x")
    if ser == par:
        print("IDENTICO BIT A BIT")
    else:
        print("DIFIEREN")
        for k in ser:
            if ser[k] != par.get(k):
                print("  ", k, ser[k], "vs", par.get(k))


if __name__ == "__main__":
    main()
