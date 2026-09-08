"""M46 — injerta el MAIN k=7 entrenado en Colab sobre el campeon actual.

build_fetch_weights.py rehusa deliberadamente tocar MAIN (ver M14: cambiar MAIN
persiguiendo fidelidad ya salio mal una vez). Aqui SI queremos cambiar MAIN, a
proposito, con G-1 (arena n=600 + control) como puerta antes de subir nada -
por eso este es un script dedicado y no una alteracion del guardian general.

Verifica que los otros 10 contextos (ACTIVATE, SWITCH, TO_HAND, conteo de
SETUP_BENCH, etc.) quedan sha-identicos al campeon: el archivo de Colab viene
de una base VIEJA (bc_alakazam_full.json) que no los tiene, asi que se
descartan y se preserva el campeon en todo menos MAIN.
"""
import hashlib, json
from pathlib import Path

def h(o): return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()[:16]

champion = json.loads(Path("data/models/bc_alakazam_final.json").read_text(encoding="utf-8"))
k7 = json.loads(Path("data/models/ctx_main_k7_only.json").read_text(encoding="utf-8"))

before = {c: h(s) for c, s in champion["contexts"].items()}
main_before = before["MAIN"]

champion["contexts"]["MAIN"] = k7["MAIN"]
champion["mlp_h"] = 48
champion["mlp_seeds"] = 7

after = {c: h(s) for c, s in champion["contexts"].items()}
drifted = [c for c in before if c != "MAIN" and before[c] != after[c]]
if drifted:
    raise SystemExit(f"contextos NO-MAIN cambiaron, algo esta mal: {drifted}")
if after["MAIN"] == main_before:
    raise SystemExit("MAIN no cambio - el injerto no hizo nada")

out = Path("data/models/bc_alakazam_k7_final.json")
out.write_text(json.dumps(champion), encoding="utf-8")
print(f"escrito {out}  ({out.stat().st_size/1e6:.1f} MB)")
print(f"  MAIN: {main_before} -> {after['MAIN']}  (k=3 -> k=7, h=48)")
print(f"  contextos sin cambio (sha verificado): {sorted(c for c in after if c != 'MAIN')}")
