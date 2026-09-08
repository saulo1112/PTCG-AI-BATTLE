"""M11 Step 1: a 1-hidden-layer MLP scorer for the MAIN context (offline gate G1).

v1's MAIN scorer is linear (w·x, held-out acc 0.862). This trains a small MLP
score(x)=w2·relu(W1 x+b1)+b2 through the SAME featurizer and the SAME conditional-
logit objective (segment-softmax CE over each decision's options), reusing train.py's
pipeline (_classify_contexts/_featurize/_pack/_seg_softmax). Only MAIN changes; every
other context keeps v1's linear vectors. Gate G1: ensemble held-out MAIN top-1 acc
>= 0.875 (vs the linear baseline retrained on the identical split for a fair compare).

If it passes, writes data/models/bc_650_v2.json (payload v2: MAIN = mlp-ensemble spec,
other contexts copied verbatim from bc_650_v1.json).

Run:  uv run --group dev python scratchpad/train_mlp_main.py [h] [seeds]
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
    _bc_accuracy, _classify_contexts, _featurize, _pack, _seg_softmax, _train_single, _L2_GRID,
)
from ptcg_ai.observation.parser import ObservationParser

DATASET = Path("data/imitation/greengreenpurple.jsonl.gz")
V1 = Path("data/models/bc_650_v1.json")
OUT = Path("data/models/bc_650_v2.json")
PROFILE = get_profile("TR_650")
G1_BAR = 0.875


class _Adam:
    def __init__(self, shape, lr):
        self.lr = lr; self.m = np.zeros(shape); self.v = np.zeros(shape); self.t = 0
    def step(self, param, grad):
        self.t += 1; b1, b2, eps = 0.9, 0.999, 1e-8
        self.m = b1 * self.m + (1 - b1) * grad
        self.v = b2 * self.v + (1 - b2) * grad * grad
        param -= self.lr * (self.m / (1 - b1**self.t)) / (np.sqrt(self.v / (1 - b2**self.t)) + eps)
        return param


def _mlp_scores(X, P):
    Z1 = X @ P["W1"] + P["b1"]
    A1 = np.maximum(Z1, 0.0)
    return A1 @ P["w2"] + P["b2"], Z1, A1


def _mlp_accuracy(decisions, params_list) -> float:
    """Top-1 accuracy of the (ensemble-averaged) MLP; MAIN is single-pick k=1."""
    correct = 0
    for d in decisions:
        scores = sum(_mlp_scores(d.X, P)[0] for P in params_list)
        if int(np.argmax(scores)) == d.chosen[0]:
            correct += 1
    return correct / max(len(decisions), 1)


def _train_mlp(tr, va, dim, h, l2, seed, epochs=400, lr=0.01, patience_checks=6):
    """Fit; early-stop on `va` accuracy (checked every 10 epochs, patience in
    checks). Returns (best_params_by_val, best_val_acc). `va` is ONLY for
    early-stop/selection — never the final reported number (that's the test set)."""
    rng = np.random.default_rng(seed)
    bigX, starts, lens, chosen_global, weights = _pack(tr)
    per_opt_w = np.repeat(weights, lens)
    onehot = np.zeros(bigX.shape[0]); onehot[chosen_global] = 1.0
    wsum = weights.sum()
    P = {
        "W1": rng.standard_normal((dim, h)) * np.sqrt(2.0 / dim),
        "b1": np.zeros(h),
        "w2": rng.standard_normal(h) * np.sqrt(1.0 / h),
        "b2": 0.0,
    }
    opt = {k: _Adam(np.shape(P[k]), lr) for k in P}
    best_acc, best_P, wait = -1.0, None, 0
    for t in range(1, epochs + 1):
        scores, Z1, A1 = _mlp_scores(bigX, P)
        probs = _seg_softmax(scores, starts, lens)
        g = (probs - onehot) * per_opt_w / wsum          # dL/dscore
        dw2 = A1.T @ g + l2 * P["w2"]
        db2 = g.sum()
        dZ1 = np.outer(g, P["w2"]) * (Z1 > 0)
        dW1 = bigX.T @ dZ1 + l2 * P["W1"]
        db1 = dZ1.sum(axis=0)
        P["W1"] = opt["W1"].step(P["W1"], dW1)
        P["b1"] = opt["b1"].step(P["b1"], db1)
        P["w2"] = opt["w2"].step(P["w2"], dw2)
        P["b2"] = float(opt["b2"].step(np.array(P["b2"]), db2))
        if t % 10 == 0:
            acc = _mlp_accuracy(va, [P])
            if acc > best_acc:
                best_acc, best_P, wait = acc, {k: np.copy(v) for k, v in P.items()}, 0
            else:
                wait += 1
                if wait >= patience_checks and t > 100:
                    break
    return best_P, best_acc


def _spec(P) -> dict:
    return {"kind": "mlp", "h": int(P["b1"].shape[0]),
            "W1": P["W1"].tolist(), "b1": P["b1"].tolist(),
            "w2": P["w2"].tolist(), "b2": float(P["b2"])}


def main() -> int:
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()
    greedy = GreedyPolicy(deck=None)

    rows = list(read_decision_dataset(DATASET))
    # THREE-way split by game: train (fit) / val (early-stop + model selection) /
    # test (untouched — the only unbiased number). Nested split_by_game calls.
    temp_rows, test_rows = split_by_game(rows, val_fraction=0.2, seed=0)
    tr_rows, va_rows = split_by_game(temp_rows, val_fraction=0.25, seed=1)
    cls = _classify_contexts(PROFILE, tr_rows, va_rows, parser, cards, greedy)
    print(f"MAIN verdict={cls.get('MAIN', {}).get('verdict')}  "
          f"greedy_acc={cls.get('MAIN', {}).get('greedy_acc'):.3f}")

    tr = _featurize(PROFILE, tr_rows, {"MAIN"}, parser, cards, 1.0)["MAIN"]
    va = _featurize(PROFILE, va_rows, {"MAIN"}, parser, cards, 1.0)["MAIN"]
    te = _featurize(PROFILE, test_rows, {"MAIN"}, parser, cards, 1.0)["MAIN"]
    dim = PROFILE.feature_dim
    print(f"MAIN featurized: train={len(tr)} val={len(va)} test={len(te)} dim={dim} h={h} seeds={n_seeds}")

    # linear baseline: pick L2 on val, report on TEST (same protocol as the MLP)
    best_lin, best_lin_va = None, -1.0
    for l2 in _L2_GRID:
        w = _train_single(tr, dim, l2)
        a = _bc_accuracy(va, w)
        if a > best_lin_va:
            best_lin, best_lin_va = w, a
    lin_test = _bc_accuracy(te, best_lin)
    print(f"\nlinear baseline MAIN acc: val {best_lin_va:.4f}  TEST {lin_test:.4f}")

    # MLP: select seed+L2+early-stop on val, report ensemble on TEST
    members = []
    for s in range(n_seeds):
        best_over_l2, va_over_l2 = None, -1.0
        for l2 in (1e-4, 1e-3):
            P, acc = _train_mlp(tr, va, dim, h, l2, seed=s)
            if acc > va_over_l2:
                best_over_l2, va_over_l2 = P, acc
        members.append(best_over_l2)
        print(f"  MLP seed {s}: val {va_over_l2:.4f}  test {_mlp_accuracy(te, [best_over_l2]):.4f}")
    ens_va = _mlp_accuracy(va, members)
    ens_test = _mlp_accuracy(te, members)
    print(f"\nMLP ensemble ({n_seeds} seeds) MAIN acc: val {ens_va:.4f}  TEST {ens_test:.4f}")
    print(f"linear {lin_test:.4f}  ->  MLP-ensemble {ens_test:.4f}  (TEST lift {ens_test-lin_test:+.4f})")
    print(f"G1 (TEST >= {G1_BAR})? {'PASS' if ens_test >= G1_BAR else 'FAIL'}")

    if ens_test >= G1_BAR:
        payload = json.loads(V1.read_text(encoding="utf-8"))
        payload["version"] = 2
        payload["contexts"]["MAIN"] = {"kind": "mlp_ensemble", "members": [_spec(P) for P in members]}
        payload.setdefault("metrics", {}).setdefault("MAIN", {})["mlp_acc"] = ens_test
        payload["mlp_h"] = h
        OUT.write_text(json.dumps(payload), encoding="utf-8")
        print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    else:
        print("G1 failed — not writing v2; consider Step 1b (richer features) or STOP.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
