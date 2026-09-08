"""Is the battle engine deterministic at all? Two identical serial runs, same process.

Motivated by a bad earlier check: I "verified" head_to_head_par as bit-identical using a
slice where the baseline won 1.000 on every deck. Identical saturated results prove
nothing. This uses self-play collection, whose win rates are nowhere near saturated.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rl_selfplay as R


def _rows(p):
    import gzip
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def main():
    R.configure("alakazam")
    Path("data/rl").mkdir(parents=True, exist_ok=True)
    a = R.collect(R.BASE_CKPT, "data/rl/_det_a.jsonl.gz", 24, tau=1.0, seed=0, workers=1)
    b = R.collect(R.BASE_CKPT, "data/rl/_det_b.jsonl.gz", 24, tau=1.0, seed=0, workers=1)
    ra, rb = _rows("data/rl/_det_a.jsonl.gz"), _rows("data/rl/_det_b.jsonl.gz")
    print(f"\nrun A: {len(ra):5d} filas  wr={ {k: v['cand_winrate'] for k, v in a['by_opp'].items()} }")
    print(f"run B: {len(rb):5d} filas  wr={ {k: v['cand_winrate'] for k, v in b['by_opp'].items()} }")
    same = len(ra) == len(rb) and all(
        {k: v for k, v in x.items() if k != "game_id"} == {k: v for k, v in y.items() if k != "game_id"}
        for x, y in zip(ra, rb))
    print("\nDETERMINISTA" if same else "\nESTOCASTICO — el motor no reproduce la misma partida")
    if not same:
        print("  => paralelo vs serie NO puede ser bit-identico, y no hace falta que lo sea:")
        print("     la recoleccion es un MUESTREO. Lo que hay que verificar es que la")
        print("     DISTRIBUCION coincide (mismo mix de oponentes, WR dentro del ruido).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
