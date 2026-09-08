"""M40 — measuring hypothesis A from M33/M34: does the clone lose Alakazam engine pieces
(Abra/Kadabra) to opponent disruption more than Yushin does, and recover worse when it happens?

M33 identified two candidate causes for the clone's slower combo assembly (Powerful-Hand-legal
turn 5.46 vs Yushin's 4.56): (B) bad play sequencing/priority, and (A) piece loss to disruption.
M34 tested (B) directly via a rebuilt TO_HAND scorer — the most rigorous test in the project: it
VERIFIED the intervention changed behavior (closed 57% of the choice gap) and STILL found no
assembly-speed or win-rate improvement, which falsified (B). This measures (A), the only
surviving hypothesis, with a pre-registered three-part gate decided BEFORE running, so a weak
result cannot be rationalized into "close enough" after the fact.

Read-only. No src/ changes. No games played, no submissions burned.

Definitions, chosen to reuse M33/M34's own calibrated instruments rather than invent new ones:
  * "assembly turn"  = first turn Powerful Hand (attack id 1072) is a legal MAIN option — the
    exact metric M33/M34 used and validated against real replays (M34 G-3 sanity check: simulated
    5.43 vs real 5.46).
  * "loss event"     = the first turn a copy of Abra (741) or Kadabra (742) newly appears in the
    discard pile, before the assembly turn. Alakazam (743) itself is deliberately excluded: it is
    the win condition switching zones (bench<->active) rather than typically discarded pre-KO in a
    way relevant to "did disruption cost us combo material".
  * "recovery item"  = Night Stretcher (1097) or Sacred Ash (1129) — the two recovery Items
    actually IN the shipped 60-card list (`decks/yushinito.csv`); without them the hypothesis
    would be structurally unfixable in this deck.

Run: PYTHONPATH=src python scratchpad/analyze_piece_loss.py
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

from ptcg_ai.observation.models import OptionKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.observation.resolve import resolve_option

MASTER_NAME = "Yushin Ito"
CLONE_NAME = "Saulo Quiñones Góngora"
MASTER_REPLAYS = "replays/54773249"
CLONE_REPLAYS = "replays/55145833"

ABRA, KADABRA, ALAKAZAM = 741, 742, 743
ENGINE_PRE = (ABRA, KADABRA)
POWERFUL_HAND = 1072
RECOVERY_IDS = {1097: "Night Stretcher", 1129: "Sacred Ash"}


def _our_seat(agents: list[dict], name: str) -> int | None:
    seats = [i for i, a in enumerate(agents) if a.get("Name", "") == name]
    return seats[0] if len(seats) == 1 else None


def analyze(folder: str, name: str) -> list[dict]:
    parser = ObservationParser()
    games = []
    for fp in sorted(glob.glob(f"{folder}/*.json")):
        if "metadata" in fp:
            continue
        try:
            d = json.loads(Path(fp).read_text(encoding="utf-8"))
        except Exception:
            continue
        agents = d.get("info", {}).get("Agents", [])
        seat = _our_seat(agents, name)
        if seat is None:
            continue

        loss_turn: int | None = None
        assembly_turn: int | None = None
        recovery_opportunities = 0   # MAIN decisions: item in hand AND legal to play, post-loss
        recovery_taken = 0           # of those, the chosen action WAS the recovery item
        prev_discard_ids: set[int] = set()

        for st in d["steps"]:
            cell = st[seat] if seat < len(st) else None
            if not cell or cell.get("status") != "ACTIVE":
                continue
            raw = cell.get("observation")
            if not raw or not raw.get("select"):
                continue
            try:
                obs = parser.parse(raw)
            except Exception:
                continue
            if obs.select is None or obs.current is None or obs.current.players is None:
                continue
            me = obs.current.players[obs.current.yourIndex]
            turn = obs.current.turn
            cur_discard_ids = {c.id for c in (me.discard or ())}

            if assembly_turn is None and obs.select.context.name == "MAIN":
                for opt in obs.select.option:
                    if opt.type is OptionKind.ATTACK and getattr(opt, "attackId", None) == POWERFUL_HAND:
                        assembly_turn = turn
                        break

            if assembly_turn is None and loss_turn is None:
                newly = (cur_discard_ids - prev_discard_ids) & set(ENGINE_PRE)
                if newly:
                    loss_turn = turn

            if (loss_turn is not None and assembly_turn is None
                    and obs.select.context.name == "MAIN"):
                hand_ids = {c.id for c in (me.hand or ())}
                if hand_ids & set(RECOVERY_IDS):
                    option_recovery_idx = set()
                    for i, opt in enumerate(obs.select.option):
                        if opt.type is not OptionKind.PLAY:
                            continue
                        try:
                            r = resolve_option(opt, obs)
                        except Exception:
                            continue
                        if r.card_id in RECOVERY_IDS:
                            option_recovery_idx.add(i)
                    if option_recovery_idx:
                        recovery_opportunities += 1
                        chosen = set(cell.get("action") or [])
                        if chosen & option_recovery_idx:
                            recovery_taken += 1

            prev_discard_ids = cur_discard_ids

        games.append({
            "loss": loss_turn is not None,
            "assembled": assembly_turn is not None,
            "assembly_turn": assembly_turn,
            "recovery_opportunities": recovery_opportunities,
            "recovery_taken": recovery_taken,
        })
    return games


def _wilson(w: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, z = w / n, 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def report(label: str, games: list[dict]) -> dict:
    n = len(games)
    n_loss = sum(1 for g in games if g["loss"])
    n_assembled = sum(1 for g in games if g["assembled"])
    with_loss_assembled = [g["assembly_turn"] for g in games if g["loss"] and g["assembled"]]
    without_loss_assembled = [g["assembly_turn"] for g in games if not g["loss"] and g["assembled"]]
    opp = sum(g["recovery_opportunities"] for g in games)
    taken = sum(g["recovery_taken"] for g in games)
    rec_rate = taken / opp if opp else float("nan")
    lo, hi = _wilson(taken, opp) if opp else (float("nan"), float("nan"))

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"\n=== {label} ===")
    print(f"  partidas analizadas: {n}")
    print(f"  con evento de pérdida (Abra/Kadabra al descarte pre-ensamblaje): "
          f"{n_loss}/{n} = {n_loss / n:.1%}" if n else "  sin partidas")
    print(f"  tasa de ensamblaje (llegó a tener Powerful Hand legal): "
          f"{n_assembled}/{n} = {n_assembled / n:.1%}" if n else "")
    print(f"  turno de ensamblaje CON pérdida: media {avg(with_loss_assembled):.2f}  "
          f"(n={len(with_loss_assembled)})")
    print(f"  turno de ensamblaje SIN pérdida: media {avg(without_loss_assembled):.2f}  "
          f"(n={len(without_loss_assembled)})")
    print(f"  oportunidades de recuperación (item en mano y legal, tras pérdida): {opp}")
    print(f"  recuperación REALMENTE jugada: {taken}/{opp} = {rec_rate:.1%}  "
          f"[IC95 {lo:.1%}-{hi:.1%}]" if opp else "  sin oportunidades de recuperación observadas")
    return {
        "n": n, "n_loss": n_loss,
        "assembly_with_loss": avg(with_loss_assembled),
        "assembly_without_loss": avg(without_loss_assembled),
        "recovery_opportunities": opp, "recovery_taken": taken, "recovery_rate": rec_rate,
    }


def main() -> int:
    master = analyze(MASTER_REPLAYS, MASTER_NAME)
    clone = analyze(CLONE_REPLAYS, CLONE_NAME)
    rm = report("MAESTRO (Yushin Ito)", master)
    rc = report("CLON (nuestro agente en el ladder)", clone)

    print("\n" + "=" * 70)
    print("PUERTA PRE-REGISTRADA (las 3 deben cumplirse para construir algo)")
    print("=" * 70)

    g1 = (not math.isnan(rc["recovery_rate"]) and not math.isnan(rm["recovery_rate"])
          and rc["recovery_rate"] < rm["recovery_rate"] - 0.30)
    print(f"[1] tasa de recuperación del clon MUCHO menor que la del maestro (gap > 0.30): "
          f"clon={rc['recovery_rate']:.1%} maestro={rm['recovery_rate']:.1%} -> "
          f"{'PASA' if g1 else 'FALLA'}")

    delay_master = (rm["assembly_with_loss"] - rm["assembly_without_loss"]
                    if not (math.isnan(rm["assembly_with_loss"]) or math.isnan(rm["assembly_without_loss"]))
                    else float("nan"))
    g2 = not math.isnan(delay_master) and delay_master >= 0.5
    print(f"[2] la pérdida SÍ retrasa el ensamblaje del maestro (>=0.5 turnos): "
          f"delta={delay_master:.2f} -> {'PASA' if g2 else 'FALLA'}")

    g3 = rc["recovery_opportunities"] >= 20 and rm["recovery_opportunities"] >= 20
    print(f"[3] suficientes casos para no ser anecdótico (>=20 oportunidades cada uno): "
          f"clon={rc['recovery_opportunities']} maestro={rm['recovery_opportunities']} -> "
          f"{'PASA' if g3 else 'FALLA'}")

    veredicto = g1 and g2 and g3
    print(f"\nVEREDICTO: {'CONSTRUIR el arreglo (las 3 condiciones se cumplen)' if veredicto else 'NO construir — hipótesis A también descartada o sin evidencia suficiente'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
