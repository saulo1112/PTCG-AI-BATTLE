"""M17 Fase 1 — offline gate for the v2 critic features (no games, ~222k cached states).

Fits the P(win|state) logistic critic on self-play trajectories using
`value_features_v2`, and ablates the three domain terms (T1 prize-weighted threat,
T2 board prize pool, T3 prose-aware threat) against the base-7 baseline recomputed
on the SAME temporal split. Cheap, high-resolution (AUC on 10k+ held-out states),
and the M16 lesson-honoring way to attribute each term's value before spending a
single gauntlet game.

Primary split matches the v_rl critic exactly (train iter1-3, val iter4) for
apples-to-apples; a larger split (train 1-6, val 7-8) is reported for robustness.

GATE: adopt v2 iff the best combo's held-out AUC >= base-7 AUC + 0.010 on BOTH splits.

Run:  uv run --group dev python scratchpad/critic_v2_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scratchpad")
from train_value import _auc, _fit_logreg  # noqa: E402
from value_features_v2 import BASE_COLS, FEATURE_NAMES, GROUP_COLS, features  # noqa: E402

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset
from ptcg_ai.observation.models import SelectContextKind
from ptcg_ai.observation.parser import ObservationParser

MAIN_CTX = int(SelectContextKind.MAIN)
GATE_LIFT = 0.010

SPLITS = {
    "v_rl-match (tr 1-3 / va 4)": ([1, 2, 3], [4]),
    "large (tr 1-6 / va 7-8)": ([1, 2, 3, 4, 5, 6], [7, 8]),
}


def _traj(i: int) -> str:
    return f"data/rl/iter{i}_traj.jsonl.gz"


def _build(iters, cards, parser):
    X, y = [], []
    for i in iters:
        for r in read_decision_dataset(Path(_traj(i))):
            if int(r.context) != MAIN_CTX:
                continue
            obs = parser.parse(r.raw_observation)
            f = features(obs, cards)
            if f is None:
                continue
            X.append(f)
            y.append(1.0 if r.won else 0.0)
    return np.asarray(X), np.asarray(y)


def _fit_eval(Xtr, ytr, Xva, yva, cols):
    """Standardize on train, fit logreg on the selected columns, return (val_auc, w)."""
    cols = list(cols)
    Xt, Xv = Xtr[:, cols], Xva[:, cols]
    mu, sd = Xt.mean(axis=0), Xt.std(axis=0) + 1e-9
    w, b = _fit_logreg((Xt - mu) / sd, ytr)
    auc = _auc(((Xv - mu) / sd) @ w + b, yva)
    return auc, w


def main() -> int:
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()

    combos = {
        "base-7": BASE_COLS,
        "base+T1(pw_threat)": BASE_COLS + GROUP_COLS["t1"],
        "base+T2(board_prize)": BASE_COLS + GROUP_COLS["t2"],
        "base+T3(prose_threat)": BASE_COLS + GROUP_COLS["t3"],
        "base+ALL": BASE_COLS + GROUP_COLS["t1"] + GROUP_COLS["t2"] + GROUP_COLS["t3"],
    }

    verdicts = []
    for split_name, (tr_iters, va_iters) in SPLITS.items():
        Xtr, ytr = _build(tr_iters, cards, parser)
        Xva, yva = _build(va_iters, cards, parser)
        print(f"\n{'='*72}\nSPLIT {split_name}: "
              f"train={len(ytr)} (win {ytr.mean():.3f})  val={len(yva)} (win {yva.mean():.3f})")
        print(f"{'combo':<26}{'val AUC':>10}{'vs base':>10}")
        print("-" * 46)
        base_auc = None
        best_lift = -1.0
        for name, cols in combos.items():
            auc, _ = _fit_eval(Xtr, ytr, Xva, yva, cols)
            if name == "base-7":
                base_auc = auc
                print(f"{name:<26}{auc:>10.4f}{'—':>10}")
            else:
                lift = auc - base_auc
                best_lift = max(best_lift, lift)
                print(f"{name:<26}{auc:>10.4f}{lift:>+10.4f}")
        # learned weight signs for the domain terms (sanity), full-combo fit
        _, w_all = _fit_eval(Xtr, ytr, Xva, yva, combos["base+ALL"])
        print("  learned weights (standardized, base+ALL):")
        for idx, col in enumerate(combos["base+ALL"]):
            print(f"    {FEATURE_NAMES[col]:<14}{w_all[idx]:+.3f}")
        verdicts.append((split_name, best_lift))

    print(f"\n{'='*72}\nGATE (lift >= +{GATE_LIFT:.3f} on BOTH splits):")
    passed = all(lift >= GATE_LIFT for _, lift in verdicts)
    for split_name, lift in verdicts:
        print(f"  {split_name:<28} best lift {lift:+.4f}  "
              f"{'PASS' if lift >= GATE_LIFT else 'FAIL'}")
    print(f"VERDICT: {'PROCEED to Fase 2 (rerun RL with v2 critic)' if passed else 'STOP — domain terms do not lift the critic; document as a kill'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
