"""Phase-0 OOD diagnostic: did v1.2's MLP behave as built on the real ladder, and
does it diverge from the linear model MORE on ladder states than on training states?

Two questions M11 never answered:
  1. PARITY: replay v1.2's real ladder MAIN decisions through the local
     ImitationPolicy('bc_650_v2.json'). ~100% agreement with the logged action ⇒
     the MLP engaged and was deterministic (rules out an infra/serving bug).
  2. OOD DIVERGENCE: on those same ladder MAIN states, how often would the LINEAR
     scorer (v1) have chosen differently than the MLP (v1.2)? Compare that to the
     same disagreement rate on held-out TEACHER states (in-distribution). A big jump
     ⇒ the two models diverge exactly where the training distribution ran out — i.e.
     the MLP is extrapolating on ladder-only opponents, the suspected v1.2 failure.

Run:  uv run --group dev python scratchpad/diagnose_v12_parity.py
"""

from __future__ import annotations

from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.imitation.kaggle_replay import iter_player_decisions
from ptcg_ai.imitation.policy import ImitationPolicy
from ptcg_ai.observation.models import SelectContextKind
from ptcg_ai.observation.parser import ObservationParser

V12_LOGS = Path("Logs/Submission imitation v1.2")
OUR = "Saulo Quiñones Góngora"
DECK = [int(x) for x in Path("decks/greengreenpurple.csv").read_text().split()]


def _main_decisions_from_replays(folder: Path):
    for f in sorted(folder.glob("*.json")):
        if f.name == "metadata.json":
            continue
        for dec in iter_player_decisions(f, OUR):
            if SelectContextKind(dec.context) is SelectContextKind.MAIN:
                yield dec


def main() -> int:
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()
    mlp = ImitationPolicy("data/models/bc_650_v2.json", deck=DECK)   # v1.2
    lin = ImitationPolicy("data/models/bc_650_v1.json", deck=DECK)   # v1

    def picks(dec):
        obs = parser.parse(dec.raw_observation)
        if obs.select is None or obs.current is None or len(obs.select.option) < 2:
            return None
        ctx = DecisionContext(raw=dec.raw_observation, observation=obs, cards=cards)
        try:
            return set(mlp.choose(ctx)), set(lin.choose(ctx)), set(dec.action)
        except Exception:
            return None

    # --- ladder states (out-of-distribution opponents) ---
    parity_ok = parity_tot = disagree = considered = 0
    for dec in _main_decisions_from_replays(V12_LOGS):
        p = picks(dec)
        if p is None:
            continue
        mlp_pick, lin_pick, logged = p
        considered += 1
        parity_tot += 1
        if mlp_pick == logged:
            parity_ok += 1
        if mlp_pick != lin_pick:
            disagree += 1
    print(f"=== v1.2 ladder MAIN decisions (n={considered}) ===")
    print(f"  PARITY  (MLP replay == logged ladder action): {parity_ok}/{parity_tot} = "
          f"{parity_ok/max(parity_tot,1):.1%}")
    print(f"  MLP vs LINEAR disagreement on ladder states:  {disagree}/{considered} = "
          f"{disagree/max(considered,1):.1%}")
    ladder_disagree = disagree / max(considered, 1)

    # --- teacher held-out states (in-distribution) ---
    rows = list(read_decision_dataset(Path("data/imitation/greengreenpurple.jsonl.gz")))
    _, val = split_by_game(rows, val_fraction=0.2, seed=0)
    d2 = c2 = 0
    for r in val:
        if SelectContextKind(r.context) is not SelectContextKind.MAIN:
            continue
        p = picks(r)
        if p is None:
            continue
        mlp_pick, lin_pick, _ = p
        c2 += 1
        if mlp_pick != lin_pick:
            d2 += 1
    iid_disagree = d2 / max(c2, 1)
    print(f"\n=== teacher held-out MAIN states (n={c2}, in-distribution) ===")
    print(f"  MLP vs LINEAR disagreement (IID): {d2}/{c2} = {iid_disagree:.1%}")

    print(f"\n=== OOD verdict ===")
    print(f"  ladder disagreement {ladder_disagree:.1%} vs IID {iid_disagree:.1%}  "
          f"(ratio {ladder_disagree/max(iid_disagree,1e-9):.2f}x)")
    print("  >1.3x => MLP diverges from linear notably more on ladder => OOD extrapolation risk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
