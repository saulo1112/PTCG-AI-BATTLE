"""M34 Track C0/C1 — cheap LINEAR A/B on TO_HAND: does ALAKAZAM_FETCH help?

Track B measured the defect (scratchpad/analyze_tohand_divergence.py): on the
teacher's own early-turn tutor decisions the clone fetches a combo piece 21.0%
of the time where the teacher fetches 49.4%, and the governing variable ("do I
already hold an Abra/Kadabra") is absent from `reduced` -- the ONLY block that
conditions a fetch on the situation, because every other block is constant
across the options of a TO_HAND decision and cancels in the softmax.

ALAKAZAM_FETCH adds exactly those 4 scalars to `reduced` (9 -> 13, dim 750).
This is the pre-registered, cheap linear check before spending on an MLP:
if the feature is right, a LINEAR model should already move, because the model
previously could not express the rule at all.

Two arms on one leak-free 3-way split by game (same seeds as
train_mlp_alakazam.py:44-45 and linear_triage_matchup.py:77-78, so every
ALAKAZAM experiment stays comparable):
  ALAKAZAM        dim 658  -- also the honest C0 baseline, retrained on the
                              CURRENT dataset (the shipped TO_HAND vector was
                              fit on ~8,996 decisions, the dataset now has more)
  ALAKAZAM_FETCH  dim 750

Reported twice, always: over all decisions, and over the NON-TRIVIAL subset
(>= 2 distinct card ids among the options). ~17% of TO_HAND decisions have
every option pointing at the same card, so the headline number is inflated by
free correctness and hides the effect.

Tie-breaks use the live convention (lowest index wins) rather than np.argsort,
which is unstable -- 55% of these decisions contain duplicate options with
identical features, so the two disagree materially here.

READ-ONLY (trains in memory, writes nothing). Run:
  uv run --group dev python scratchpad/triage_tohand_profile.py
  uv run --group dev python scratchpad/triage_tohand_profile.py --exact-multi
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.imitation.train import (
    _L2_GRID, Decision, _featurize, _train_multi, _train_single,
)
from ptcg_ai.observation.parser import ObservationParser

DATASET = Path("data/imitation/yushinito_full.jsonl.gz")
ARMS = ("ALAKAZAM", "ALAKAZAM_FETCH")


def _stable_topk(scores: np.ndarray, k: int) -> set[int]:
    """Top-k with the LIVE tie-break (policy._rank_order: lowest index wins)."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return set(order[:k])


def _setmatch(decisions: list[Decision], w: np.ndarray) -> float:
    if not decisions:
        return float("nan")
    hit = sum(_stable_topk(d.X @ w, len(d.chosen)) == set(d.chosen) for d in decisions)
    return hit / len(decisions)


def _expand_stages(decisions: list[Decision]) -> list[Decision]:
    """Plackett-Luce as a sequence of single-pick stages.

    Maximising the PL likelihood of a k-pick decision is identical to summing
    the single-pick softmax NLL over the k stages with the already-taken
    options removed, so the vectorized `_train_single` optimises the same
    objective as `_train_multi` (up to the gradient normaliser, which Adam
    absorbs into an effective L2 rescale).
    """
    out: list[Decision] = []
    for d in decisions:
        if len(d.chosen) == 1:
            out.append(d)
            continue
        active = list(range(d.X.shape[0]))
        for a in d.chosen:
            out.append(Decision(X=d.X[active], chosen=[active.index(a)], weight=d.weight))
            active.remove(a)
    return out


def _nontrivial_mask(decisions: list[Decision]) -> list[bool]:
    """A decision is trivial when every option has an identical feature row.

    Identical rows == the same card id in the same zone, which is exactly the
    "all options are the same card" case: any legal pick is correct and the
    take-k rule alone scores it.
    """
    out = []
    for d in decisions:
        X = d.X
        out.append(bool(X.shape[0] > 1 and not np.allclose(X, X[0], atol=0.0)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact-multi", action="store_true",
                    help="train with the shipped _train_multi loop instead of the "
                         "equivalent-but-vectorized stage expansion (slow, exact)")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--context", default="TO_HAND",
                    help="which SelectContextKind to A/B (TO_HAND, MAIN, SWITCH, ...). "
                         "MAIN matters because the payload carries ONE profile for every "
                         "context, so shipping ALAKAZAM_FETCH retrains MAIN at dim 750 too.")
    args = ap.parse_args()
    CONTEXT = args.context

    t0 = time.time()
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()
    print(f"[{time.time()-t0:6.1f}s] card DB loaded", flush=True)

    rows = list(read_decision_dataset(DATASET))
    temp_rows, test_rows = split_by_game(rows, val_fraction=0.2, seed=0)
    tr_rows, va_rows = split_by_game(temp_rows, val_fraction=0.25, seed=1)
    print(f"[{time.time()-t0:6.1f}s] rows: train={len(tr_rows)} val={len(va_rows)} "
          f"test={len(test_rows)}", flush=True)

    results = {}
    for name in ARMS:
        profile = get_profile(name)
        dim = profile.feature_dim
        tr = _featurize(profile, tr_rows, {CONTEXT}, parser, cards, 1.0)[CONTEXT]
        va = _featurize(profile, va_rows, {CONTEXT}, parser, cards, 1.0)[CONTEXT]
        te = _featurize(profile, test_rows, {CONTEXT}, parser, cards, 1.0)[CONTEXT]
        nt = _nontrivial_mask(te)
        te_nt = [d for d, keep in zip(te, nt) if keep]
        print(f"[{time.time()-t0:6.1f}s] {name}: featurized "
              f"train={len(tr)} val={len(va)} test={len(te)} (non-trivial {len(te_nt)})",
              flush=True)

        fit = (lambda d, l2: _train_multi(d, dim, l2, epochs=args.epochs)) if args.exact_multi \
            else (lambda d, l2, e=_expand_stages(tr): _train_single(e, dim, l2, epochs=args.epochs))

        best_w, best_va, best_l2 = None, -1.0, None
        for l2 in _L2_GRID:
            w = fit(tr, l2)
            a = _setmatch(va, w)
            print(f"[{time.time()-t0:6.1f}s]   l2={l2:<8g} val set-match {a:.4f}", flush=True)
            if a > best_va:
                best_w, best_va, best_l2 = w, a, l2

        results[name] = {
            "dim": dim, "l2": best_l2, "val": best_va,
            "test": _setmatch(te, best_w),
            "test_nt": _setmatch(te_nt, best_w),
            "train": _setmatch(tr, best_w),
            "n_test": len(te), "n_test_nt": len(te_nt),
        }

    print("\n" + "=" * 78)
    print(f"M34 — LINEAR A/B on {CONTEXT} "
          f"({'exact multi-pick' if args.exact_multi else 'stage-expanded'})")
    print("=" * 78)
    print(f"{'arm':<18}{'dim':>6}{'l2':>9}{'val':>9}{'TEST':>9}{'TEST-nt':>10}{'train-test':>12}")
    for name in ARMS:
        r = results[name]
        print(f"{name:<18}{r['dim']:>6}{r['l2']:>9g}{r['val']:>9.4f}"
              f"{r['test']:>9.4f}{r['test_nt']:>10.4f}{r['train'] - r['test']:>+12.4f}")

    base, cand = results[ARMS[0]], results[ARMS[1]]
    d_all = cand["test"] - base["test"]
    d_nt = cand["test_nt"] - base["test_nt"]
    print(f"\ndelta (FETCH - ALAKAZAM):  TEST {d_all:+.4f}   TEST-non-trivial {d_nt:+.4f}")
    print(f"n_test = {base['n_test']}   n_test_non_trivial = {base['n_test_nt']}")
    print(f"\nG-1 (pre-registered): non-trivial TEST delta >= +0.02 AND the overfit gap")
    print(f"    (train-test) must not grow by more than +0.02 vs the baseline.")
    gap_growth = (cand["train"] - cand["test"]) - (base["train"] - base["test"])
    ok = d_nt >= 0.02 and gap_growth <= 0.02
    print(f"    non-trivial delta {d_nt:+.4f}   overfit-gap growth {gap_growth:+.4f}"
          f"   -> {'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
