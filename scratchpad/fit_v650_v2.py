"""M17 search pilot prep — fit the TEACHER-trained base-7 + T3 leaf V (v_650_v2.json).

The M10 search leaf used `v_650.json` (7 features, fit on the teacher decision dataset).
To isolate whether the ONE corrected feature (T3 prose-aware threat) helps the SEARCH
mechanism, we need a leaf V that differs from v_650 by exactly that feature and nothing
else — same dataset, same split, same fitter. (Using v_rl_v2.json instead would also
change the training distribution to self-play, confounding the pilot.)

Mirrors scratchpad/train_value.py (same split_by_game, same _fit_logreg) but featurizes
with value_features_v2 and keeps columns [base-7, prose_threat].

Run:  uv run --group dev python scratchpad/fit_v650_v2.py
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
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.observation.parser import ObservationParser

DATASET = Path("data/imitation/greengreenpurple.jsonl.gz")
OUT = Path("data/models/v_650_v2.json")
COLS = list(BASE_COLS) + list(GROUP_COLS["t3"])  # base-7 + prose_threat


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
            X.append([f[c] for c in COLS])
            y.append(1.0 if r.won else 0.0)
        return np.asarray(X), np.asarray(y)

    Xtr, ytr = build(train_rows)
    Xva, yva = build(val_rows)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    w, b = _fit_logreg((Xtr - mu) / sd, ytr)
    val_auc = _auc(((Xva - mu) / sd) @ w + b, yva)

    # base-7-only baseline on the SAME split (what v_650 effectively is)
    b7 = list(range(7))
    mu7, sd7 = Xtr[:, b7].mean(axis=0), Xtr[:, b7].std(axis=0) + 1e-9
    w7, b_7 = _fit_logreg((Xtr[:, b7] - mu7) / sd7, ytr)
    val_auc7 = _auc(((Xva[:, b7] - mu7) / sd7) @ w7 + b_7, yva)

    feat_names = [FEATURE_NAMES[c] for c in COLS]
    OUT.write_text(json.dumps({
        "version": 2, "features": feat_names,
        "mean": mu.tolist(), "std": sd.tolist(),
        "weights": w.tolist(), "bias": float(b),
        "val_auc": val_auc, "trained_on": "teacher greengreenpurple, base-7+T3",
    }), encoding="utf-8")
    print(f"states: train={len(ytr)} val={len(yva)}")
    print(f"teacher-fit AUC:  base-7 {val_auc7:.4f}   base-7+T3 {val_auc:.4f}   ({val_auc-val_auc7:+.4f})")
    print(f"weights: " + "  ".join(f"{n}={wi:+.3f}" for n, wi in zip(feat_names, w)))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
