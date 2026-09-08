"""Does splitting self-play collection across processes change the DISTRIBUTION?

Bit-identity is the wrong bar and an earlier version of this script used it. The engine
shuffles decks from entropy, so two identical serial runs already differ
(`_check_determinism.py` measures that). Collection is sampling; what must be preserved is:

  1. every opponent gets its exact game quota (a lost worker = skewed training mix);
  2. the candidate's win rate per opponent stays within sampling noise;
  3. no BC failures appear that were not there serially.

Must be a real importable module: Windows spawns workers by re-importing __main__.
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rl_selfplay as R

N = 60


def main():
    R.configure("alakazam")
    Path("data/rl").mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    s = R.collect(R.BASE_CKPT, "data/rl/_chk_serial.jsonl.gz", N, tau=1.0, seed=0, workers=1)
    ts = time.perf_counter() - t

    t = time.perf_counter()
    p = R.collect(R.BASE_CKPT, "data/rl/_chk_par.jsonl.gz", N, tau=1.0, seed=0, workers=4)
    tp = time.perf_counter() - t

    print(f"\nserie    {ts:6.1f}s  filas={s['n_main_rows']:5d}  juegos={s['n_games']}")
    print(f"paralelo {tp:6.1f}s  filas={p['n_main_rows']:5d}  juegos={p['n_games']}")
    print(f"speedup  {ts / max(tp, 1e-9):.2f}x\n")

    ok = True
    print(f"{'oponente':<14}{'juegos S':>9}{'juegos P':>9}{'WR S':>8}{'WR P':>8}{'|dif|':>8}{'2se':>8}")
    for name in R.OPPONENTS:
        a, b = s["by_opp"].get(name), p["by_opp"].get(name)
        if not a or not b:
            print(f"{name:<14}  FALTA en {'serie' if not a else 'paralelo'}"); ok = False; continue
        if a["games"] != b["games"]:
            print(f"{name:<14}  CUOTA DISTINTA {a['games']} vs {b['games']}"); ok = False; continue
        wa, wb = a["cand_winrate"], b["cand_winrate"]
        n = max(min(a["decided"], b["decided"]), 1)
        se2 = 2 * math.sqrt(2 * 0.25 / n)          # 2 s.e. de la diferencia de dos WR
        flag = "" if abs(wa - wb) <= se2 else "  <-- fuera de ruido"
        if flag:
            ok = False
        print(f"{name:<14}{a['games']:>9}{b['games']:>9}{wa:>8.3f}{wb:>8.3f}"
              f"{abs(wa - wb):>8.3f}{se2:>8.3f}{flag}")

    if s["bc_failures"] or p["bc_failures"]:
        print(f"\nBC FAILURES: serie={s['bc_failures']} paralelo={p['bc_failures']} -- deben ser 0")
        ok = False

    print("\nDISTRIBUCION EQUIVALENTE" if ok else "\nREVISAR: la distribucion no coincide")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
