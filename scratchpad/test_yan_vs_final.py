"""M45 — la puerta que decide: clon de Yan (Ogerpon) vs imitation-final (Alakazam).

Yan (55235071, implicito 997) paso las DOS puertas del screen por primera vez en el
proyecto: fuerza de mazo 0.722 (barra 0.40) Y clonabilidad MAIN 0.813 (barra 0.75),
esta ultima con un perfil GENERICO y un modelo LINEAL -- donde Yushin necesita perfil
hecho a mano + MLP ensemble para llegar a 0.780.

Esto mide lo unico que importa: JUGANDO, cual gana. Cada uno pilota SU propio mazo,
que es la comparacion real (no comparten lista). El control campeon-vs-campeon debe
leer ~0.500; si no, la medicion no vale (M41: a n=200 el espejo leyo 0.435).

OJO: el payload de Yan usa GENERIC_YAN, que NO esta en dp.PROFILES y por tanto NO
embarca. Esto es solo una MEDICION para decidir si vale la pena escribir un perfil
a mano. build_generic_profile es determinista, asi que reproduce exactamente el
perfil contra el que se entreno.

Run:  PYTHONPATH="src;scratchpad" python scratchpad/test_yan_vs_final.py [n]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from arena_m8 import run
from field_gauntlet import _cards

from ptcg_ai.imitation import deck_profiles as dp

FINAL = "data/models/bc_alakazam_final.json"
FINAL_DECK = "decks/yushinito.csv"
YAN = "data/models/yan_screen.json"
YAN_DECK = "decks/yan.csv"


def _ensure(deck_csv: str, weights: str, cards) -> None:
    name = json.loads(Path(weights).read_text(encoding="utf-8")).get("profile") or "TR_650"
    if name in dp.PROFILES:
        return
    deck = [int(x) for x in Path(deck_csv).read_text(encoding="utf-8").split()]
    prof = dp.build_generic_profile(name, tuple(deck), cards)
    dp.PROFILES[prof.name] = prof
    print(f"registrado {prof.name} (dim {prof.feature_dim})")


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    _sdk, cards = _cards()   # field_gauntlet._cards returns (sdk, CardDatabase)
    _ensure(YAN_DECK, YAN, cards)

    rows = []
    s = run("imitation", YAN_DECK, "imitation", FINAL_DECK,
            f"YAN(ogerpon) vs FINAL(alakazam) n={n}", n=n, wa=YAN, wb=FINAL)
    rows.append(("yan vs imitation-final", s.score_rate, 0.500))

    s = run("imitation", FINAL_DECK, "imitation", FINAL_DECK,
            f"CONTROL final vs final n={n}", n=n, wa=FINAL, wb=FINAL)
    rows.append(("control final vs final", s.score_rate, 0.500))

    print("\n" + "=" * 64)
    print(f"{'matchup':<32}{'observado':>11}{'ref':>8}{'delta':>10}")
    print("-" * 64)
    for lab, got, ref in rows:
        print(f"{lab:<32}{got:>11.3f}{ref:>8.3f}{got-ref:>+10.3f}")
    print("-" * 64)
    print("El control DEBE leer ~0.500; si no, la fila 1 no es de fiar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
