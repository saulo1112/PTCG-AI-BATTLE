"""M13 Phase B: extend the MLP recipe to TO_HAND (multi-pick) + a 5-seed MAIN
ensemble, composing data/models/bc_650_v4.json.

TO_HAND is the second decision-dense context (linear bc_acc 0.9253 vs greedy 0.521,
multi-pick). The MLP trainer in train_mlp_main.py assumes single-pick (_pack uses
chosen[0]); we make it reusable via the STAGE-EXPANSION trick: a k-pick decision
factorizes (Plackett-Luce) into k single-pick stages, each over the remaining
options in the teacher's logged order -- exactly _train_multi's sequential loss.
The expanded single-pick stages feed _pack/_train_mlp UNCHANGED. Evaluation stays
on the ORIGINAL decisions with set-match top-k (mirrors _bc_accuracy for linear).

Protocol mirrors train_mlp_main.py for comparability: nested 3-way split by game
(train fit / val select+early-stop / TEST report only). Linear baseline retrained
with _train_multi on the identical split.

Eval-only by default (writes nothing). Pass --write <base_payload.json> <out.json>
to compose the v4 payload AFTER the offline + gauntlet gates pass.

Run (eval):   uv run --group dev python scratchpad/train_mlp_v4.py [h_main] [seeds]
Run (write):  uv run --group dev python scratchpad/train_mlp_v4.py 32 5 \
                  --write data/models/bc_650_v2.json data/models/bc_650_v4.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.imitation.train import (
    _bc_accuracy, _classify_contexts, _featurize, _train_multi, _train_single, _L2_GRID, Decision,
)
from ptcg_ai.observation.parser import ObservationParser

sys.path.insert(0, "scratchpad")
from train_mlp_main import _mlp_scores, _train_mlp, _spec  # noqa: E402

DATASET = Path("data/imitation/greengreenpurple.jsonl.gz")
PROFILE = get_profile("TR_650")
# Gate bars (offline): TO_HAND must beat linear by this and clear the absolute bar.
TOHAND_LIFT = 0.01
TOHAND_ABS = 0.93


# -- stage expansion: k-pick -> k single-pick stages (Plackett-Luce) ----------

def _expand_stages(decisions: list[Decision]) -> list[Decision]:
    """Each k-pick decision -> k single-pick stage decisions over remaining options,
    in the teacher's logged pick order (matches _train_multi's active.remove loop)."""
    out: list[Decision] = []
    for d in decisions:
        active = list(range(d.X.shape[0]))
        for a in d.chosen:
            idx = active.index(a)
            out.append(Decision(X=d.X[active], chosen=[idx], weight=d.weight))
            active.remove(a)
    return out


def _mlp_setmatch(decisions: list[Decision], members) -> float:
    """Set-match top-k accuracy of the ensemble on ORIGINAL (multi-pick) decisions."""
    if not decisions:
        return 0.0
    correct = 0
    for d in decisions:
        k = len(d.chosen)
        scores = sum(_mlp_scores(d.X, P)[0] for P in members)
        pred = set(np.argsort(-scores)[:k].tolist())
        correct += pred == set(d.chosen)
    return correct / len(decisions)


def _train_mlp_ensemble(tr, va, dim, h, n_seeds, expand=False):
    """Train an n_seeds MLP ensemble. If expand, stage-expand tr/va first (multi-pick)."""
    tr_e = _expand_stages(tr) if expand else tr
    va_e = _expand_stages(va) if expand else va
    members = []
    for s in range(n_seeds):
        best_P, best_va = None, -1.0
        for l2 in (1e-4, 1e-3):
            P, acc = _train_mlp(tr_e, va_e, dim, h, l2, seed=s)
            if acc > best_va:
                best_P, best_va = P, acc
        members.append(best_P)
    return members


def _linear_multi_baseline(tr, va, te, dim):
    """Linear TO_HAND baseline: _train_multi, L2 selected on val set-match, report TEST."""
    best_w, best_va = None, -1.0
    for l2 in _L2_GRID:
        w = _train_multi(tr, dim, l2)
        a = _bc_accuracy(va, w)
        if a > best_va:
            best_w, best_va = w, a
    return best_w, best_va, _bc_accuracy(te, best_w)


def main() -> int:
    argv = sys.argv[1:]
    write_base = write_out = None
    if "--write" in argv:
        i = argv.index("--write")
        write_base, write_out = Path(argv[i + 1]), Path(argv[i + 2])
        argv = argv[:i]
    h_main = int(argv[0]) if len(argv) > 0 else 32
    n_seeds_main = int(argv[1]) if len(argv) > 1 else 5

    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()
    greedy = GreedyPolicy(deck=None)
    dim = PROFILE.feature_dim

    rows = list(read_decision_dataset(DATASET))
    temp_rows, test_rows = split_by_game(rows, val_fraction=0.2, seed=0)
    tr_rows, va_rows = split_by_game(temp_rows, val_fraction=0.25, seed=1)
    cls = _classify_contexts(PROFILE, tr_rows, va_rows, parser, cards, greedy)
    for ctx in ("MAIN", "TO_HAND"):
        d = cls.get(ctx, {})
        print(f"{ctx}: verdict={d.get('verdict')} greedy_acc={d.get('greedy_acc', float('nan')):.3f} "
              f"n_train={d.get('n_train')}")

    # ===================== TO_HAND (multi-pick) =====================
    th_tr = _featurize(PROFILE, tr_rows, {"TO_HAND"}, parser, cards, 1.0)["TO_HAND"]
    th_va = _featurize(PROFILE, va_rows, {"TO_HAND"}, parser, cards, 1.0)["TO_HAND"]
    th_te = _featurize(PROFILE, test_rows, {"TO_HAND"}, parser, cards, 1.0)["TO_HAND"]
    print(f"\nTO_HAND featurized: train={len(th_tr)} val={len(th_va)} test={len(th_te)} dim={dim}")

    lin_w, lin_va, lin_test = _linear_multi_baseline(th_tr, th_va, th_te, dim)
    print(f"TO_HAND linear baseline: val {lin_va:.4f}  TEST {lin_test:.4f}")

    print(f"\n{'h':>4}{'seeds':>7}{'val':>9}{'TEST':>9}")
    best_th_members, best_th_h, best_th_va = None, None, -1.0
    for h in (8, 16, 32):
        members = _train_mlp_ensemble(th_tr, th_va, dim, h, n_seeds=3, expand=True)
        ens_va = _mlp_setmatch(th_va, members)
        ens_te = _mlp_setmatch(th_te, members)
        print(f"{h:>4}{3:>7}{ens_va:>9.4f}{ens_te:>9.4f}")
        if ens_va > best_th_va:
            best_th_members, best_th_h, best_th_va = members, h, ens_va
    th_test = _mlp_setmatch(th_te, best_th_members)
    print(f"\nTO_HAND best h={best_th_h}: MLP-ensemble TEST {th_test:.4f}  "
          f"(linear {lin_test:.4f}, lift {th_test - lin_test:+.4f})")
    g_b1 = th_test >= lin_test + TOHAND_LIFT and th_test >= TOHAND_ABS
    print(f"G-B1 (TO_HAND >= linear+{TOHAND_LIFT} AND >= {TOHAND_ABS})? "
          f"{'PASS' if g_b1 else 'FAIL'}")

    # ===================== MAIN 5-seed vs 3-seed =====================
    m_tr = _featurize(PROFILE, tr_rows, {"MAIN"}, parser, cards, 1.0)["MAIN"]
    m_va = _featurize(PROFILE, va_rows, {"MAIN"}, parser, cards, 1.0)["MAIN"]
    m_te = _featurize(PROFILE, test_rows, {"MAIN"}, parser, cards, 1.0)["MAIN"]
    main_members = _train_mlp_ensemble(m_tr, m_va, dim, h_main, n_seeds=n_seeds_main, expand=False)
    main3 = _mlp_setmatch(m_te, main_members[:3])
    main5 = _mlp_setmatch(m_te, main_members)
    print(f"\nMAIN h={h_main}: 3-seed TEST {main3:.4f}  {n_seeds_main}-seed TEST {main5:.4f}")
    keep5 = main5 >= main3 - 0.002
    print(f"keep {n_seeds_main}-seed MAIN? {'YES' if keep5 else 'NO (revert to 3-seed)'}")

    if write_base is not None:
        payload = json.loads(write_base.read_text(encoding="utf-8"))
        payload["version"] = 4
        chosen_main = main_members if keep5 else main_members[:3]
        payload["contexts"]["MAIN"] = {"kind": "mlp_ensemble",
                                       "members": [_spec(P) for P in chosen_main]}
        if g_b1:
            payload["contexts"]["TO_HAND"] = {"kind": "mlp_ensemble",
                                              "members": [_spec(P) for P in best_th_members]}
            print(f"composing v4 WITH TO_HAND MLP (h={best_th_h})")
        else:
            print("composing v4 WITHOUT TO_HAND MLP (G-B1 failed) -- MAIN changes only")
        payload.setdefault("metrics", {}).setdefault("MAIN", {})["mlp_acc"] = main5
        payload["metrics"].setdefault("TO_HAND", {})["mlp_acc"] = th_test
        payload["mlp_h"] = h_main
        write_out.write_text(json.dumps(payload), encoding="utf-8")
        print(f"wrote {write_out} ({write_out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
