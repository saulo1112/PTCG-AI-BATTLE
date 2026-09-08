"""Phase-2 step 1: learn a value function V(state)->P(win) for the search leaf.

The M6 search lost (0.44) with a HAND-TUNED linear V (decision/evaluator.py). Before
building the whole BC-guided search, this is the cheap de-risking gate: fit logistic
regression on the SAME 7 antisymmetric features the hand-tuned V uses, to the teacher
dataset's win/loss label, and check whether learned weights rank won-vs-lost states
BETTER than the hand-tuned weights (held-out AUC, split by game).

- If learned AUC >> hand-tuned AUC: the weights were suboptimal; search-with-learned-V
  has a real shot. Proceed to build search_bc.py.
- If they're similar: these 7 features are the ceiling — a linear V (any weights) can't
  rank states much better, so Phase 2 would need richer features, not just search.

Writes data/models/v_650.json (weights + feature standardization) for the leaf eval.

Run:  uv run --group dev python scratchpad/train_value.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.evaluator import EvalWeights, _board_energy, _deckout_risk
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

#: Defaults reproduce the original M-phase-2 run exactly. M38 overrides them from the CLI
#: to fit the SAME 7 antisymmetric features on Yushin's corpus instead: `v_650` was fit on
#: TR-650 self-play and M18 already measured that it is mis-calibrated out of distribution,
#: so reusing it as the RL advantage baseline would corrupt every advantage we compute.
DATASET = Path("data/imitation/greengreenpurple.jsonl.gz")
OUT = Path("data/models/v_650.json")
FEATURE_NAMES = ("prize", "threat", "reserve", "survival", "energy", "hand", "deck_out")


def _clip1(x: float) -> float:
    return max(-1.0, min(1.0, x))


def features(obs, cards) -> list[float] | None:
    """The 7 evaluator.py feature differences, from the teacher's perspective."""
    state = obs.current
    if state is None:
        return None
    gs = GameState.build(obs, cards)
    me, opp = gs.me, gs.opponent
    if me is None or opp is None:
        return None
    prize = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    reserve = _clip1((gs.reserve_attackers(me) - gs.reserve_attackers(opp)) / 5.0)
    survival = _clip1((gs.my_bench_count - len(opp.bench)) / 5.0)
    on_them = on_me = 0.0
    if gs.opp_active is not None and gs.opp_active.hp > 0:
        on_them = min(1.0, gs.max_threat(gs.my_active, gs.opp_active, extra_energy=1) / gs.opp_active.hp)
    if gs.my_active is not None and gs.my_active.hp > 0:
        on_me = min(1.0, gs.max_threat(gs.opp_active, gs.my_active, extra_energy=1) / gs.my_active.hp)
    threat = on_them - on_me
    energy = _clip1((_board_energy(me) - _board_energy(opp)) / 10.0)
    hand = _clip1((me.handCount - opp.handCount) / 10.0)
    deck_out = _deckout_risk(opp.deckCount) - _deckout_risk(me.deckCount)
    return [prize, threat, reserve, survival, energy, hand, deck_out]


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score(win) > score(loss))."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _fit_logreg(X, y, l2=1e-3, epochs=300, lr=0.1):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    m_w = np.zeros(d); v_w = np.zeros(d); m_b = v_b = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        gw = X.T @ (p - y) / n + l2 * w
        gb = (p - y).mean()
        m_w = b1 * m_w + (1 - b1) * gw; v_w = b2 * v_w + (1 - b2) * gw * gw
        m_b = b1 * m_b + (1 - b1) * gb; v_b = b2 * v_b + (1 - b2) * gb * gb
        w -= lr * (m_w / (1 - b1**t)) / (np.sqrt(v_w / (1 - b2**t)) + eps)
        b -= lr * (m_b / (1 - b1**t)) / (np.sqrt(v_b / (1 - b2**t)) + eps)
    return w, b


def main() -> int:
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()

    rows = list(read_decision_dataset(DATASET))
    train_rows, val_rows = split_by_game(rows, val_fraction=0.2, seed=0)

    def build(rows):
        X, y = [], []
        for r in rows:
            obs = parser.parse(r.raw_observation)
            f = features(obs, cards)
            if f is None:
                continue
            X.append(f)
            y.append(1.0 if r.won else 0.0)
        return np.asarray(X), np.asarray(y)

    Xtr, ytr = build(train_rows)
    Xva, yva = build(val_rows)
    print(f"states: train={len(ytr)} (win {ytr.mean():.2f})  val={len(yva)} (win {yva.mean():.2f})")

    # standardize (store for inference)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-9
    Xtr_s = (Xtr - mu) / sd
    Xva_s = (Xva - mu) / sd

    w, b = _fit_logreg(Xtr_s, ytr)
    learned_tr = _auc(Xtr_s @ w + b, ytr)
    learned_va = _auc(Xva_s @ w + b, yva)

    # hand-tuned linear V as a scorer (its weighted sum of the same features)
    hw = EvalWeights()
    hand_vec = np.array([hw.prize, hw.threat, hw.reserve, hw.survival, hw.energy, hw.hand, hw.deck_out])
    hand_tr = _auc(Xtr @ hand_vec, ytr)
    hand_va = _auc(Xva @ hand_vec, yva)

    print(f"\nAUC (rank won-vs-lost states):")
    print(f"  hand-tuned linear V:  train {hand_tr:.3f}  val {hand_va:.3f}")
    print(f"  learned logistic V:   train {learned_tr:.3f}  val {learned_va:.3f}")
    print(f"  val AUC lift: {learned_va - hand_va:+.3f}")

    print(f"\nlearned weights (standardized) vs hand-tuned:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:<10} learned {w[i]:+.3f}   hand {hand_vec[i]:+.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "version": 1, "features": list(FEATURE_NAMES),
        "mean": mu.tolist(), "std": sd.tolist(),
        "weights": w.tolist(), "bias": float(b),
        "val_auc": learned_va, "hand_val_auc": hand_va,
    }), encoding="utf-8")
    print(f"\nwrote {OUT}")
    verdict = "PROCEED (learned V ranks better)" if learned_va - hand_va > 0.02 else \
              "WEAK (features are the ceiling — Phase 2 needs richer features, not just search)"
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--dataset", type=Path, default=DATASET)
    _ap.add_argument("--out", type=Path, default=OUT)
    _a = _ap.parse_args()
    DATASET, OUT = _a.dataset, _a.out
    if not DATASET.is_file():
        raise SystemExit(f"dataset not found: {DATASET}")
    print(f"fitting V on {DATASET} -> {OUT}")
    raise SystemExit(main())
