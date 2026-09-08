"""Behavior-cloning trainer (dev-only; requires numpy).

Fits one linear conditional-logit weight vector per learned context: the score
of an option is ``w · φ(option)`` and the policy is ``softmax`` over a decision's
options, trained by cross-entropy against the logged player's choice. Single-pick
contexts train with a fully vectorized segment-softmax; multi-pick contexts (the
take-max ones) use a sequential Plackett-Luce loss (the teacher's pick-count is
deterministic, so only the ranking is learned).

Context selection is **data-driven** (no hard-coded list): a context is learned
iff it has ≥250 train rows AND greedy-as-predictor val accuracy <95% AND is not
"degenerate" (nearly every decision's options resolve to a single card id — e.g.
discarding identical energies). Degenerate contexts get a zeros weight vector so
the live policy's take-k rule still fires (fixing greedy's return-``[]`` bug on
Aura Jab's from-discard attach). Optional win-weighting down-weights decisions
from lost games (α<1) to bias the clone toward the teacher's winning play.

Run:  ``uv run --group dev python -m ptcg_ai.imitation.train <dataset.jsonl.gz>
       <deck.csv> <out_weights.json> <profile> [--seed N] [--alpha A]``
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation import features as F
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.imitation.deck_profiles import DeckProfile, get_profile
from ptcg_ai.imitation.kaggle_replay import ReplayDecision
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.observation.resolve import resolve_option
from ptcg_ai.state.game_state import GameState

_L2_GRID = (1e-4, 1e-3, 1e-2)
_MIN_ROWS = 250          # learn only contexts with at least this many train rows
_GREEDY_SKIP_ACC = 0.95  # skip contexts greedy already predicts this well
_DEGENERATE_FRAC = 0.99  # frac of decisions whose options are one card id
_MIN_LIFT = 0.05         # keep a learned context only if BC beats greedy by this
#: SelectKind values whose options are real cards — the only selects where an
#: all-zero "take-k by index" vector is a valid stand-in (CARD/ATTACHED_CARD/
#: CARD_OR_ATTACHED_CARD/ENERGY). COUNT/YES_NO are excluded (see verdict logic).
_CARD_LIKE_SELECTS = frozenset({1, 2, 3, 4})


@dataclass
class Decision:
    """One featurized decision: option matrix, chosen indices, and CE weight."""

    X: np.ndarray          # (n_options, feature_dim) float32
    chosen: list[int]      # logged order, legal, in range
    weight: float          # 1.0 for won games, α for lost


# -- pass 1: classify contexts (cheap; no featurization) ----------------------

def _classify_contexts(
    profile: DeckProfile,
    train_rows: list[ReplayDecision],
    val_rows: list[ReplayDecision],
    parser: ObservationParser,
    cards: CardDatabase,
    greedy: GreedyPolicy,
) -> dict[str, dict]:
    """Per context: n_train, degenerate fraction, greedy val accuracy, verdict."""
    info: dict[str, dict] = {}
    # train-side counts + degeneracy
    for r in train_rows:
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None or F.is_prize_pick(obs.select):
            continue
        ctx = obs.select.context.name
        d = info.setdefault(ctx, {"n_train": 0, "degen": 0, "g_ok": 0, "g_tot": 0, "select_type": 0})
        d["n_train"] += 1
        d["select_type"] = int(obs.select.type)
        ids = {resolve_option(o, obs).card_id for o in obs.select.option}
        if len(ids) <= 1:
            d["degen"] += 1
    # val-side greedy-as-predictor accuracy
    for r in val_rows:
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None or F.is_prize_pick(obs.select):
            continue
        ctx = obs.select.context.name
        if ctx not in info:
            continue
        n = len(obs.select.option)
        chosen = [a for a in r.action if 0 <= a < n]
        if not chosen:
            continue
        d = info[ctx]
        try:
            pred = greedy.choose(DecisionContext(raw=r.raw_observation, observation=obs, cards=cards))
        except Exception:
            pred = []
        d["g_ok"] += set(pred) == set(chosen)
        d["g_tot"] += 1
    for ctx, d in info.items():
        degen_frac = d["degen"] / max(d["n_train"], 1)
        g_acc = d["g_ok"] / d["g_tot"] if d["g_tot"] else 1.0
        d["degen_frac"] = degen_frac
        d["greedy_acc"] = g_acc
        # A "zeros" verdict (all-zero weights → live take-k by lowest index) is
        # only correct when the options are actual cards (CARD/ENERGY-like selects,
        # where the teacher takes k=min(maxCount,n) and identical cards make the
        # choice positional). NUMBER (COUNT=8) and YES/NO (YES_NO=9) selects also
        # resolve to card_id=None for every option, so they LOOK degenerate — but
        # take-k=index-0 is wrong there (draws the minimum / could pick NO). Route
        # those to greedy, whose _max_number / _is_first / _safe_default handlers
        # match the teacher (verified: DRAW_COUNT max 114/114, IS_FIRST YES 237/237).
        if degen_frac >= _DEGENERATE_FRAC and d["select_type"] in _CARD_LIKE_SELECTS:
            d["verdict"] = "zeros"
        elif d["n_train"] >= _MIN_ROWS and g_acc < _GREEDY_SKIP_ACC:
            d["verdict"] = "learn"
        else:
            d["verdict"] = "skip"
    return info


# -- pass 2: featurize the learned contexts -----------------------------------

def _featurize(
    profile: DeckProfile,
    rows: list[ReplayDecision],
    learn: set[str],
    parser: ObservationParser,
    cards: CardDatabase,
    alpha: float,
) -> dict[str, list[Decision]]:
    out: dict[str, list[Decision]] = {c: [] for c in learn}
    for r in rows:
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None:
            continue
        ctx = obs.select.context.name
        if ctx not in learn or F.is_prize_pick(obs.select):
            continue
        n = len(obs.select.option)
        chosen = [a for a in r.action if 0 <= a < n]
        if not chosen or len(set(chosen)) != len(chosen):
            continue
        gs = GameState.build(obs, cards)
        X = np.asarray(F.featurize_decision(profile, obs.select, obs, gs, cards), dtype=np.float32)
        out[ctx].append(Decision(X=X, chosen=chosen, weight=1.0 if r.won else alpha))
    return out


# -- vectorized single-pick training ------------------------------------------

def _pack(decisions: list[Decision]):
    bigX = np.concatenate([d.X for d in decisions], axis=0)
    lens = np.array([d.X.shape[0] for d in decisions], dtype=np.int64)
    starts = np.zeros(len(decisions), dtype=np.int64)
    np.cumsum(lens[:-1], out=starts[1:])
    chosen_global = starts + np.array([d.chosen[0] for d in decisions], dtype=np.int64)
    weights = np.array([d.weight for d in decisions], dtype=np.float64)
    return bigX, starts, lens, chosen_global, weights


def _seg_softmax(logits, starts, lens):
    seg_max = np.maximum.reduceat(logits, starts)
    ex = np.exp(logits - np.repeat(seg_max, lens))
    seg_sum = np.add.reduceat(ex, starts)
    return ex / np.repeat(seg_sum, lens)


def _train_single(decisions, dim, l2, epochs=400, lr=0.05):
    bigX, starts, lens, chosen_global, weights = _pack(decisions)
    per_option_w = np.repeat(weights, lens)
    chosen_sum = (weights[:, None] * bigX[chosen_global]).sum(axis=0)
    wsum = weights.sum()
    w = np.zeros(dim); m = np.zeros(dim); v = np.zeros(dim)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        probs = _seg_softmax(bigX @ w, starts, lens)
        grad = (bigX.T @ (per_option_w * probs) - chosen_sum) / wsum + l2 * w
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * (grad * grad)
        w -= lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
    return w


def _train_multi(decisions, dim, l2, epochs=400, lr=0.05):
    w = np.zeros(dim); m = np.zeros(dim); v = np.zeros(dim)
    b1, b2, eps = 0.9, 0.999, 1e-8
    wsum = sum(d.weight for d in decisions)
    for t in range(1, epochs + 1):
        grad = np.zeros(dim)
        for d in decisions:
            active = list(range(d.X.shape[0]))
            for a in d.chosen:
                sub = d.X[active]
                logits = sub @ w
                p = np.exp(logits - logits.max()); p /= p.sum()
                grad += d.weight * ((p @ sub) - d.X[a])
                active.remove(a)
        grad = grad / wsum + l2 * w
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * (grad * grad)
        w -= lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
    return w


# -- evaluation ---------------------------------------------------------------

def _bc_accuracy(decisions, w) -> float:
    if not decisions:
        return 0.0
    correct = 0
    for d in decisions:
        k = len(d.chosen)
        pred = set(np.argsort(-(d.X @ w))[:k].tolist())
        correct += pred == set(d.chosen)
    return correct / len(decisions)


def _random_expectation(decisions) -> float:
    if not decisions:
        return 0.0
    total = 0.0
    for d in decisions:
        n, k = d.X.shape[0], len(d.chosen)
        total += 1.0 / math.comb(n, k) if 0 < k <= n else 0.0
    return total / len(decisions)


def train_bc(
    dataset: Path, deck: Path, out_weights: Path, profile_name: str,
    seed: int = 0, alpha: float = 1.0,
) -> dict:
    profile = get_profile(profile_name)
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()
    greedy = GreedyPolicy(deck=None)

    rows = list(read_decision_dataset(dataset))
    train_rows, val_rows = split_by_game(rows, val_fraction=0.2, seed=seed)
    print(f"profile={profile.name} dim={profile.feature_dim} alpha={alpha}")
    print(f"rows={len(rows)}  train={len(train_rows)}  val={len(val_rows)}  (split by game)")

    cls = _classify_contexts(profile, train_rows, val_rows, parser, cards, greedy)
    learn = {c for c, d in cls.items() if d["verdict"] == "learn"}
    zeros = {c for c, d in cls.items() if d["verdict"] == "zeros"}
    print("\ncontext classification:")
    for ctx, d in sorted(cls.items(), key=lambda kv: -kv[1]["n_train"]):
        print(f"  {ctx:<24} n_tr={d['n_train']:>6} degen={d['degen_frac']:.2f} "
              f"greedy={d['greedy_acc']:.3f} -> {d['verdict']}")

    t0 = time.perf_counter()
    tr_by = _featurize(profile, train_rows, learn, parser, cards, alpha)
    va_by = _featurize(profile, val_rows, learn, parser, cards, alpha)
    print(f"\nfeaturized learned contexts in {time.perf_counter()-t0:.0f}s")

    weights: dict[str, list[float]] = {}
    metrics: dict[str, dict] = {}
    dim = profile.feature_dim
    print(f"\n{'context':<24}{'n_tr':>7}{'n_val':>7}{'BC':>8}{'greedy':>8}{'random':>8}{'l2':>7}")
    print("-" * 76)
    for ctx in sorted(learn):
        tr, va = tr_by[ctx], va_by[ctx]
        if len(tr) < _MIN_ROWS:
            continue
        multi = any(len(d.chosen) > 1 for d in tr)
        trainer = _train_multi if multi else _train_single
        best_w, best_acc, best_l2 = None, -1.0, _L2_GRID[0]
        for l2 in _L2_GRID:
            w = trainer(tr, dim, l2)
            acc = _bc_accuracy(va, w)
            if acc > best_acc:
                best_w, best_acc, best_l2 = w, acc, l2
        g_acc = cls[ctx]["greedy_acc"]
        r_acc = _random_expectation(va)
        if best_acc < g_acc + _MIN_LIFT:
            print(f"{ctx:<24}{len(tr):>7}{len(va):>7}{best_acc:>8.3f}{g_acc:>8.3f}{r_acc:>8.3f}"
                  f"   DROP (lift<{_MIN_LIFT})")
            continue
        weights[ctx] = [float(x) for x in best_w]
        metrics[ctx] = {"n_train": len(tr), "n_val": len(va), "bc_acc": best_acc,
                        "greedy_acc": g_acc, "random_acc": r_acc, "l2": best_l2, "multi_pick": multi}
        print(f"{ctx:<24}{len(tr):>7}{len(va):>7}{best_acc:>8.3f}{g_acc:>8.3f}{r_acc:>8.3f}{best_l2:>7.0e}")

    for ctx in sorted(zeros):
        weights[ctx] = [0.0] * dim
        print(f"{ctx:<24}{cls[ctx]['n_train']:>7}{'':>7}{'zeros':>8}  (degenerate: take-k)")

    tot = sum(m["n_val"] for m in metrics.values())
    bc_macro = sum(m["bc_acc"] * m["n_val"] for m in metrics.values()) / max(tot, 1)
    gr_macro = sum(m["greedy_acc"] * m["n_val"] for m in metrics.values()) / max(tot, 1)
    print("-" * 76)
    print(f"{'weighted (learned)':<24}{'':>7}{tot:>7}{bc_macro:>8.3f}{gr_macro:>8.3f}")

    out_weights.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1, "profile": profile.name, "deck": Path(deck).name,
        "feature_dim": dim, "seed": seed, "alpha": alpha,
        "contexts": weights, "metrics": metrics,
        "degenerate_contexts": sorted(zeros),
        "weighted_bc_acc": bc_macro, "weighted_greedy_acc": gr_macro,
    }
    import json
    out_weights.write_text(json.dumps(payload), encoding="utf-8")
    print(f"\nwrote {out_weights}  ({out_weights.stat().st_size/1024:.0f} KB)")
    return payload


def _parse_argv(argv: list[str]):
    if len(argv) < 4:
        print("usage: python -m ptcg_ai.imitation.train <dataset> <deck.csv> "
              "<out.json> <profile> [--seed N] [--alpha A]")
        raise SystemExit(2)
    dataset, deck, out, profile = argv[0], argv[1], argv[2], argv[3]
    seed, alpha = 0, 1.0
    rest = argv[4:]
    for i, tok in enumerate(rest):
        if tok == "--seed" and i + 1 < len(rest):
            seed = int(rest[i + 1])
        elif tok == "--alpha" and i + 1 < len(rest):
            alpha = float(rest[i + 1])
    return Path(dataset), Path(deck), Path(out), profile, seed, alpha


if __name__ == "__main__":
    ds, dk, out, prof, seed, alpha = _parse_argv(sys.argv[1:])
    train_bc(ds, dk, out, prof, seed=seed, alpha=alpha)
