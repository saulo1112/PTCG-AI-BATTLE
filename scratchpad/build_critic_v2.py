"""M17 Fase 2 prep — fit and save the v2 critic (base-7 + T3 prose_threat).

The gate (critic_v2_gate.py) selected base-7 + T3 as the principled critic: T3 alone
lifts held-out AUC +0.014/+0.017, while T1/T2 add ~0 or fit a confound. This fits that
8-feature logistic critic on the self-play states and writes it in the format
`rl_selfplay._load_v650` expects, tagged with a "features" list so the loader routes the
right columns.

Fit on iter1-6, validate on iter7-8 (matches the gate's "large" split), then save the
iter1-6 fit as the deployed critic.

Run:  uv run --group dev python scratchpad/build_critic_v2.py
"""

from __future__ import annotations

import json
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
COLS = list(BASE_COLS) + list(GROUP_COLS["t3"])  # base-7 + prose_threat
OUT = Path("data/models/v_rl_v2.json")


def _build(iters, cards, parser):
    X, y = [], []
    for i in iters:
        for r in read_decision_dataset(Path(f"data/rl/iter{i}_traj.jsonl.gz")):
            if int(r.context) != MAIN_CTX:
                continue
            obs = parser.parse(r.raw_observation)
            f = features(obs, cards)
            if f is None:
                continue
            X.append([f[c] for c in COLS])
            y.append(1.0 if r.won else 0.0)
    return np.asarray(X), np.asarray(y)


def main() -> int:
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()

    Xtr, ytr = _build([1, 2, 3, 4, 5, 6], cards, parser)
    Xva, yva = _build([7, 8], cards, parser)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    w, b = _fit_logreg((Xtr - mu) / sd, ytr)
    val_auc = _auc(((Xva - mu) / sd) @ w + b, yva)

    feat_names = [FEATURE_NAMES[c] for c in COLS]
    OUT.write_text(json.dumps({
        "version": 2, "features": feat_names,
        "mean": mu.tolist(), "std": sd.tolist(),
        "weights": w.tolist(), "bias": float(b),
        "val_auc": val_auc, "trained_on": "selfplay iter1-6, val iter7-8",
    }), encoding="utf-8")
    print(f"features: {feat_names}")
    print(f"train={len(ytr)} val={len(yva)}  held-out AUC={val_auc:.4f}")
    print(f"weights: " + "  ".join(f"{n}={wi:+.3f}" for n, wi in zip(feat_names, w)))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
