"""M28 Phase 2 — the capacity experiment for the Yushin (ALAKAZAM) clone.

Reuses train_mlp_main.py's MLP machinery + train.py's pipeline, but parameterized by
dataset + profile, and reports TRAIN/VAL/TEST for BOTH the linear baseline and the MLP
ensemble on the identical 3-way game-split. The point (per m28 plan): in the top-1 +
big-data regime, does higher capacity raise HELD-OUT (test) MAIN fidelity WITHOUT a large
train/test overfit gap — the tripwire the small-data M11/M14 MLP tripped?

Run:  uv run --group dev python scratchpad/train_mlp_alakazam.py <dataset.jsonl.gz> <PROFILE> [h] [seeds]
"""

from __future__ import annotations

import sys
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.imitation.train import (
    _L2_GRID, _bc_accuracy, _classify_contexts, _featurize, _train_single,
)
from ptcg_ai.observation.parser import ObservationParser

from train_mlp_main import _mlp_accuracy, _train_mlp


def main() -> int:
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/imitation/yushinito_full.jsonl.gz")
    profile_name = sys.argv[2] if len(sys.argv) > 2 else "ALAKAZAM"
    h = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    n_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    PROFILE = get_profile(profile_name)
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()
    greedy = GreedyPolicy(deck=None)

    rows = list(read_decision_dataset(dataset))
    temp_rows, test_rows = split_by_game(rows, val_fraction=0.2, seed=0)
    tr_rows, va_rows = split_by_game(temp_rows, val_fraction=0.25, seed=1)
    cls = _classify_contexts(PROFILE, tr_rows, va_rows, parser, cards, greedy)
    print(f"profile={PROFILE.name} dim={PROFILE.feature_dim} h={h} seeds={n_seeds}")
    print(f"MAIN greedy_acc={cls.get('MAIN', {}).get('greedy_acc'):.3f}")

    tr = _featurize(PROFILE, tr_rows, {"MAIN"}, parser, cards, 1.0)["MAIN"]
    va = _featurize(PROFILE, va_rows, {"MAIN"}, parser, cards, 1.0)["MAIN"]
    te = _featurize(PROFILE, test_rows, {"MAIN"}, parser, cards, 1.0)["MAIN"]
    dim = PROFILE.feature_dim
    print(f"MAIN featurized: train={len(tr)} val={len(va)} test={len(te)}\n")

    # linear baseline: L2 by val, report TRAIN/VAL/TEST
    best_lin, best_lin_va = None, -1.0
    for l2 in _L2_GRID:
        w = _train_single(tr, dim, l2)
        a = _bc_accuracy(va, w)
        if a > best_lin_va:
            best_lin, best_lin_va = w, a
    lin_tr, lin_te = _bc_accuracy(tr, best_lin), _bc_accuracy(te, best_lin)
    print(f"LINEAR   train {lin_tr:.4f}  val {best_lin_va:.4f}  TEST {lin_te:.4f}"
          f"  (train-test gap {lin_tr-lin_te:+.4f})")

    # MLP ensemble: seed+L2 by val, report TRAIN/VAL/TEST
    members = []
    for s in range(n_seeds):
        best_P, va_best = None, -1.0
        for l2 in (1e-4, 1e-3):
            P, acc = _train_mlp(tr, va, dim, h, l2, seed=s)
            if acc > va_best:
                best_P, va_best = P, acc
        members.append(best_P)
        print(f"  MLP seed {s}: val {va_best:.4f}  test {_mlp_accuracy(te, [best_P]):.4f}")
    mlp_tr = _mlp_accuracy(tr, members)
    mlp_va = _mlp_accuracy(va, members)
    mlp_te = _mlp_accuracy(te, members)
    print(f"MLP-ens  train {mlp_tr:.4f}  val {mlp_va:.4f}  TEST {mlp_te:.4f}"
          f"  (train-test gap {mlp_tr-mlp_te:+.4f})")
    print(f"\n>> TEST lift MLP - linear: {mlp_te-lin_te:+.4f}  "
          f"(overfit gap: linear {lin_tr-lin_te:+.4f} vs MLP {mlp_tr-mlp_te:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
