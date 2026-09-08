"""M42 Arm B — learn HOW MANY options to take, not just which ones.

THE GAP. `ImitationPolicy._top_k` implements "take min(maxCount, n), floored at
minCount" — the teacher's rule, verified for every context except one. Measured on
Yushin's 2,330-game corpus with the vetted extractor:

    SETUP_BENCH_POKEMON   teacher declines (k=0) on 37.4% of decisions
                          our clone declined 0 of 45 on its own ladder replays

No ranking model can close that at any capacity or depth: the count is not a
property of the ranking. Hence a separate head.

FEATURES. The mean of the option vectors (every option shares the state blocks, so
pooling preserves the state exactly and averages the card identities on offer) plus
two set-level scalars the per-option featurizer cannot see — n_options and maxCount.
Mirrors `ImitationPolicy._count_from_head` exactly; if you change one, change both.

THE TRAP THIS SCRIPT AVOIDS. `train._featurize` drops decisions whose `chosen` is
empty ("no legal chosen index"), which is PRECISELY the k=0 rows — the entire signal
being learned here. So this reads the dataset directly instead of reusing it.

REPORTED NUMBER is end-to-end: rank with the ranking head, take count-head-many,
set-match against the teacher. Scoring the count head in isolation would flatter it,
because a perfect count over a bad ranking still picks the wrong cards.

Run:
  PYTHONPATH=src python scratchpad/train_count_head.py --out data/models/ctx_count.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation import features as F
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.imitation.train import Decision, _train_multi
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

DATASET = Path("data/imitation/yushinito_full.jsonl.gz")
DECK = Path("decks/yushinito.csv")
CONTEXT = "SETUP_BENCH_POKEMON"
GATE = 0.05


def _rank_order(scores):
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def _collect(rows, profile, parser, cards):
    """(pooled_x, k, X, chosen, lo, hi) per decision — k=0 rows KEPT."""
    out = []
    for r in rows:
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None:
            continue
        if obs.select.context.name != CONTEXT or F.is_prize_pick(obs.select):
            continue
        n = len(obs.select.option)
        if n == 0:
            continue
        chosen = [a for a in r.action if 0 <= a < n]
        if len(set(chosen)) != len(chosen):
            continue
        gs = GameState.build(obs, cards)
        X = np.asarray(F.featurize_decision(profile, obs.select, obs, gs, cards), dtype=np.float64)
        pooled = np.concatenate([X.mean(axis=0), [n / 4.0, obs.select.maxCount / 4.0]])
        lo = min(max(obs.select.minCount, 0), n)
        hi = min(obs.select.maxCount, n)
        out.append((pooled, len(chosen), X, chosen, lo, hi, r))
    return out


def _fit_multinomial(Xf, yf, n_cls, l2, steps=1500, lr=0.05):
    d = Xf.shape[1]
    W = np.zeros((n_cls, d)); b = np.zeros(n_cls)
    mW = np.zeros_like(W); vW = np.zeros_like(W); mb = np.zeros(n_cls); vb = np.zeros(n_cls)
    Y = np.zeros((len(yf), n_cls)); Y[np.arange(len(yf)), yf] = 1.0
    for t in range(1, steps + 1):
        z = Xf @ W.T + b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        g = (p - Y) / len(yf)
        gW = g.T @ Xf + l2 * W; gb = g.sum(axis=0)
        mW = 0.9 * mW + 0.1 * gW; vW = 0.999 * vW + 0.001 * gW ** 2
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb ** 2
        W -= lr * (mW / (1 - 0.9 ** t)) / (np.sqrt(vW / (1 - 0.999 ** t)) + 1e-8)
        b -= lr * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
    return W, b


def _predict_k(W, b, classes, pooled, lo, hi):
    z = W @ pooled + b
    return max(lo, min(hi, int(classes[int(np.argmax(z))])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--profile", default="ALAKAZAM")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--bar", type=float, default=GATE)
    args = ap.parse_args()

    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()
    profile = get_profile(args.profile)
    deck = [int(x) for x in DECK.read_text(encoding="utf-8").split()]
    greedy = GreedyPolicy(deck=deck)

    rows = list(read_decision_dataset(args.dataset))
    temp_rows, test_rows = split_by_game(rows, val_fraction=0.2, seed=0)
    tr_rows, va_rows = split_by_game(temp_rows, val_fraction=0.25, seed=1)
    tr = _collect(tr_rows, profile, parser, cards)
    va = _collect(va_rows, profile, parser, cards)
    te = _collect(te_rows := test_rows, profile, parser, cards)
    print(f"{CONTEXT}: train={len(tr)} val={len(va)} test={len(te)}  profile={profile.name}/{profile.feature_dim}")
    dist = collections.Counter(k for _, k, *_ in tr)
    print(f"  teacher k distribution (train): {sorted(dist.items())}")
    print(f"  declines (k=0): {dist[0]}/{len(tr)} = {dist[0]/max(len(tr),1):.3f}")

    classes = sorted({k for _, k, *_ in tr})
    cls_idx = {c: i for i, c in enumerate(classes)}
    Xtr = np.stack([p for p, *_ in tr]); ytr = np.array([cls_idx[k] for _, k, *_ in tr])
    Xva = np.stack([p for p, *_ in va])

    # --- ranking head (multi-pick) on the k>=1 rows ---------------------------
    rank_tr = [Decision(X=X, chosen=ch, weight=1.0) for _, k, X, ch, *_ in tr if k >= 1]
    best_w, best_va_rank = None, -1.0
    for l2 in (1e-4, 1e-3, 1e-2):
        w = _train_multi(rank_tr, profile.feature_dim, l2)
        hit = tot = 0
        for _, k, X, ch, *_ in va:
            if k < 1:
                continue
            tot += 1
            hit += set(_rank_order((X @ w).tolist())[:k]) == set(ch)
        acc = hit / max(tot, 1)
        print(f"    ranking l2={l2:<7} val(oracle-k)={acc:.4f}")
        if acc > best_va_rank:
            best_w, best_va_rank = w, acc

    # --- count head -----------------------------------------------------------
    best_cnt, best_va_cnt = None, -1.0
    for l2 in (1e-4, 1e-3, 1e-2, 1e-1):
        W, b = _fit_multinomial(Xtr, ytr, len(classes), l2)
        hit = sum(_predict_k(W, b, classes, p, lo, hi) == k for p, k, _, _, lo, hi, _ in va)
        acc = hit / max(len(va), 1)
        print(f"    count   l2={l2:<7} val(k-exact)={acc:.4f}")
        if acc > best_va_cnt:
            best_cnt, best_va_cnt = (W, b), acc
    W, b = best_cnt

    # --- end-to-end TEST: greedy vs ranking@cap vs ranking@count -------------
    def _greedy_pred(r, obs_row):
        try:
            return greedy.choose(DecisionContext(raw=r.raw_observation,
                                                 observation=parser.parse(r.raw_observation),
                                                 cards=cards))
        except Exception:  # noqa: BLE001
            return []

    g_ok = cap_ok = cnt_ok = k_ok = 0
    cnt_declines = teacher_declines = 0
    for p, k, X, ch, lo, hi, r in te:
        scores = (X @ best_w).tolist()
        order = _rank_order(scores)
        kc = _predict_k(W, b, classes, p, lo, hi)
        g_ok += set(_greedy_pred(r, None)) == set(ch)
        cap_ok += set(order[:hi if hi >= lo else lo]) == set(ch)
        cnt_ok += set(order[:kc]) == set(ch)
        k_ok += kc == k
        cnt_declines += kc == 0
        teacher_declines += k == 0
    n = max(len(te), 1)
    print(f"\n  TEST n={n}")
    print(f"    greedy (SHIPPED)            {g_ok/n:.4f}")
    print(f"    ranking head @ cap          {cap_ok/n:.4f}")
    print(f"    ranking head @ COUNT head   {cnt_ok/n:.4f}   <- candidate")
    print(f"    count exact-match           {k_ok/n:.4f}")
    print(f"    declines: teacher {teacher_declines}/{n} = {teacher_declines/n:.3f}   "
          f"candidate {cnt_declines}/{n} = {cnt_declines/n:.3f}")
    lift = cnt_ok / n - g_ok / n
    verdict = "PASS" if lift >= args.bar else "FAIL"
    print(f"\n  G-B1 (lift over shipped >= +{args.bar:.2f}): {verdict}  ({lift:+.4f})")

    if args.out and verdict == "PASS":
        payload = {
            "contexts": {CONTEXT: {"kind": "linear", "w": [float(x) for x in best_w]}},
            "count_heads": {CONTEXT: {
                "classes": [int(c) for c in classes],
                "W": [[float(x) for x in row] for row in W],
                "b": [float(x) for x in b],
            }},
            "_metrics": {CONTEXT: {
                "test_acc": cnt_ok / n, "baseline_acc": g_ok / n, "lift": lift,
                "count_exact": k_ok / n, "n_test": n,
            }},
        }
        args.out.write_text(json.dumps(payload), encoding="utf-8")
        print(f"  wrote {args.out}")
    elif args.out:
        print("  gate failed — not writing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
