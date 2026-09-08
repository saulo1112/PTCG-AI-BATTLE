"""M24 — score several BC weight files on ONE fixed held-out set (read-only).

Each ``train_bc`` run does its own ``split_by_game`` over its own rows, so the
``weighted_bc_acc`` stored in each JSON is measured on a DIFFERENT val set and is
not comparable across variants. This tool decouples "which weights" from "which
rows": it fixes a single evaluation set E and scores every model on the SAME E,
so the pooled-vs-individual comparison is apples-to-apples.

Primary E (``--eval third``, default): the MAIN-context decisions in the *val*
partition of THIRD's own dataset (seed 0). Because ``split_by_game`` hashes the
globally-unique episode-id game_id, those game_ids land on the val side in every
source, so NO variant trained on E (solo-THIRD held it out; solo-Dries/team_name
never saw THIRD games at all; pooled excluded them via the same hash). It is thus
a clean held-out measure of fidelity to the SHIPPED teacher (THIRD PTCG Club).

Reuses ``train.py`` internals verbatim (``_featurize``, ``split_by_game``,
``read_decision_dataset``, ``get_profile``) so the featurization path is identical
to training — no train/serve skew in the evaluation itself.

Run:  uv run --group dev python scratchpad/eval_fidelity_pooled.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.imitation.train import _featurize
from ptcg_ai.observation.parser import ObservationParser

DECK = "decks/thirdptcgclub.csv"
PROFILE = "THIRD_PTCG"
SEED = 0

THIRD_DS = "data/imitation/third_full.jsonl.gz"
POOLED_DS = "data/imitation/third_pooled3.jsonl.gz"

# label -> weights path. Order matters only for display.
MODELS = [
    ("solo-THIRD", "data/models/bc_third_ptcg.json"),
    ("solo-Dries", "data/models/bc_third_dries.json"),
    ("solo-team_name", "data/models/bc_third_teamname.json"),
    ("pooled-3", "data/models/bc_third_pooled3.json"),
]
INDIVIDUALS = {"solo-THIRD", "solo-Dries", "solo-team_name"}


def _correct_vector(decisions, w: np.ndarray) -> np.ndarray:
    """Per-decision boolean correctness under top-k (same rule as train._bc_accuracy)."""
    out = np.empty(len(decisions), dtype=bool)
    for i, d in enumerate(decisions):
        k = len(d.chosen)
        pred = set(np.argsort(-(d.X @ w))[:k].tolist())
        out[i] = pred == set(d.chosen)
    return out


def _load_main_weight(path: str, dim: int) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("profile") != PROFILE:
        raise SystemExit(f"{path}: profile {payload.get('profile')} != {PROFILE}")
    vec = payload["contexts"].get("MAIN")
    if vec is None:
        raise SystemExit(f"{path}: no MAIN context learned")
    w = np.asarray(vec, dtype=np.float64)
    if w.shape[0] != dim:
        raise SystemExit(f"{path}: MAIN dim {w.shape[0]} != profile dim {dim}")
    return w


def _build_eval_set(dataset: str, profile, parser, cards):
    """MAIN decisions in the seed-0 val partition of `dataset`, with the val game_ids."""
    rows = list(read_decision_dataset(Path(dataset)))
    train_rows, val_rows = split_by_game(rows, val_fraction=0.2, seed=SEED)
    train_games = {r.game_id for r in train_rows}
    val_games = {r.game_id for r in val_rows}
    E = _featurize(profile, val_rows, {"MAIN"}, parser, cards, alpha=1.0)["MAIN"]
    return E, train_games, val_games


def _paired_bootstrap(a: np.ndarray, b: np.ndarray, iters: int = 5000, seed: int = 0):
    """90% CI of mean(a) - mean(b) resampling decision indices (paired)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    deltas = np.empty(iters)
    for t in range(iters):
        idx = rng.integers(0, n, n)
        deltas[t] = a[idx].mean() - b[idx].mean()
    return float(a.mean() - b.mean()), float(np.percentile(deltas, 5)), float(np.percentile(deltas, 95))


def _score_on(E, dim):
    accs, vecs = {}, {}
    for label, path in MODELS:
        if not Path(path).exists():
            print(f"  (skip {label}: {path} missing)")
            continue
        w = _load_main_weight(path, dim)
        cv = _correct_vector(E, w)
        accs[label] = cv.mean()
        vecs[label] = cv
    return accs, vecs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()
    profile = get_profile(PROFILE)
    dim = profile.feature_dim

    # Primary held-out: THIRD's own val MAIN decisions.
    E_third, third_train_games, third_val_games = _build_eval_set(THIRD_DS, profile, parser, cards)
    # Leak assert: THIRD's val games must be on the VAL side of the pooled split too.
    pooled_rows = list(read_decision_dataset(Path(POOLED_DS)))
    p_train, p_val = split_by_game(pooled_rows, val_fraction=0.2, seed=SEED)
    p_train_games = {r.game_id for r in p_train}
    leaked = third_val_games & p_train_games
    print(f"profile={PROFILE} dim={dim} seed={SEED}")
    print(f"E (THIRD val MAIN decisions): {len(E_third)}  | THIRD val games={len(third_val_games)}")
    print(f"LEAK CHECK: THIRD-val games that landed in pooled TRAIN = {len(leaked)} "
          f"(must be 0){' OK' if not leaked else ' !!!'}")

    print("\n=== PRIMARY: MAIN accuracy on E = THIRD held-out (same rows for all) ===")
    accs, vecs = _score_on(E_third, dim)
    for label, _ in MODELS:
        if label in accs:
            print(f"  {label:<16} {accs[label]:.4f}")

    best_ind = max((l for l in INDIVIDUALS if l in accs), key=lambda l: accs[l], default=None)
    if best_ind and "pooled-3" in accs:
        pt, lo, hi = _paired_bootstrap(vecs["pooled-3"], vecs[best_ind], seed=args.boot_seed)
        print(f"\n  best individual = {best_ind} ({accs[best_ind]:.4f})")
        print(f"  pooled-3 - {best_ind}: {pt:+.4f}  90%CI [{lo:+.4f}, {hi:+.4f}]")
        gate = (pt >= 0.02 and lo > 0.0 and accs["pooled-3"] >= accs["solo-THIRD"])
        print(f"\n  PRE-REGISTERED GATE (pooled-best>=+0.02 AND CI-lo>0 AND pooled>=solo-THIRD):"
              f" {'PASS -> escalate (with approval)' if gate else 'FAIL -> NEGATIVE, touch nothing'}")

    # Secondary diagnostic: union held-out (generic-pilot fidelity).
    print("\n=== SECONDARY (diagnostic): MAIN accuracy on E_union = pooled held-out ===")
    E_union = _featurize(profile, p_val, {"MAIN"}, parser, cards, alpha=1.0)["MAIN"]
    print(f"E_union: {len(E_union)} decisions")
    accs_u, _ = _score_on(E_union, dim)
    for label, _ in MODELS:
        if label in accs_u:
            print(f"  {label:<16} {accs_u[label]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
