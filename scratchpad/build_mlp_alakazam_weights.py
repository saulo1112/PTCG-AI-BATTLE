"""M28 Phase 3 — produce SHIPPABLE MLP-ensemble weights for the Yushin (ALAKAZAM) clone.

The Phase-2 experiment (train_mlp_alakazam.py) proved the MLP lifts held-out MAIN
fidelity 0.556 -> 0.780 with a SMALLER overfit gap than linear. This trains the final
ensemble on the FULL 2-way train split (max data, early-stop on a val slice) and writes
a shippable payload: MAIN = mlp_ensemble, every other context copied verbatim from the
linear bc_alakazam_full.json (the M11 bc_650_v2 flow). Inference already supports it
(policy._score -> mlp_ensemble).

Run:  uv run --group dev python scratchpad/build_mlp_alakazam_weights.py [h] [seeds]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.imitation.train import _classify_contexts, _featurize
from ptcg_ai.observation.parser import ObservationParser

from train_mlp_main import _mlp_accuracy, _spec, _train_mlp

DATASET = Path("data/imitation/yushinito_full.jsonl.gz")
LINEAR = Path("data/models/bc_alakazam_full.json")
OUT = Path("data/models/bc_alakazam_mlp.json")
PROFILE = get_profile("ALAKAZAM")


def main() -> int:
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()
    greedy = GreedyPolicy(deck=None)

    rows = list(read_decision_dataset(DATASET))
    # 2-way split: max training data, hold out 20% ONLY for MLP early-stop/selection.
    tr_rows, va_rows = split_by_game(rows, val_fraction=0.2, seed=0)
    _classify_contexts(PROFILE, tr_rows, va_rows, parser, cards, greedy)  # warms nothing; parity w/ train
    tr = _featurize(PROFILE, tr_rows, {"MAIN"}, parser, cards, 1.0)["MAIN"]
    va = _featurize(PROFILE, va_rows, {"MAIN"}, parser, cards, 1.0)["MAIN"]
    dim = PROFILE.feature_dim
    print(f"MAIN featurized: train={len(tr)} val={len(va)} dim={dim} h={h} seeds={n_seeds}", flush=True)

    members = []
    for s in range(n_seeds):
        best_P, va_best = None, -1.0
        for l2 in (1e-4, 1e-3):
            P, acc = _train_mlp(tr, va, dim, h, l2, seed=s)
            if acc > va_best:
                best_P, va_best = P, acc
        members.append(best_P)
        print(f"  MLP seed {s}: val {va_best:.4f}", flush=True)
    print(f"ensemble val MAIN acc: {_mlp_accuracy(va, members):.4f}", flush=True)

    payload = json.loads(LINEAR.read_text(encoding="utf-8"))
    payload["version"] = 2
    payload["contexts"]["MAIN"] = {"kind": "mlp_ensemble", "members": [_spec(P) for P in members]}
    payload["mlp_h"] = h
    payload["mlp_seeds"] = n_seeds
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
