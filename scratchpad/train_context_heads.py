"""M42 — train a scorer for ANY decision context, not just MAIN.

WHY THIS EXISTS. An audit of Yushin's 2,330-game corpus (190,731 decisions) against
the shipped payload found that **12.2% of all decisions (23,354) have no model at
all** — they fall through to GreedyPolicy, and the most frequent of them to
`_safe_default`, which returns `list(range(minCount))`: option 0, with no look at
the state. The biggest is ACTIVATE ("use this optional Ability?" — Psychic Draw,
Run Away Draw, Flip the Script) at **6.55 decisions/game, 14,750 rows, the 3rd most
common decision in the game**.

They were never trained because `train.py:284` drops a context unless it beats
greedy by `_MIN_LIFT = 0.05`. Greedy already sits at 0.934 on ACTIVATE, so the bar
demands 0.984 — unreachable by arithmetic, not by merit. This script removes that
one assumption and reports the honest number instead: TEST accuracy against **what
we actually ship today** for that context.

BASELINE DISCIPLINE (the M37 rule: "a sweep's baseline is the SHIPPED artefact
measured with the same instrument, never an internal arm"). For each context the
baseline is, in order:
  1. the spec in `--baseline-weights` if that payload has the context (scored under
     the spec's own `profile` override if it declares one), else
  2. GreedyPolicy — the real fallback, run through the same filters.

TIE-BREAK. Accuracy uses the LIVE rule `sorted(range(n), key=(-score, i))`
(`policy._rank_order`), not `np.argsort`. `train._bc_accuracy` uses the latter and
the two disagree materially on duplicate-heavy contexts (TO_HAND: ~55% of decisions
carry duplicate identical-feature options). Both are printed; the live one gates.

MULTI-PICK. k-pick decisions factorize into k single-pick Plackett-Luce stages
(M13's `train_mlp_v4._expand_stages`), which feeds the MAIN MLP trainer unchanged.
Evaluation stays on the ORIGINAL decisions with set-match top-k.

Writes nothing by default. `--out` dumps the winning spec per context, ready for
`build_fetch_weights.py` to graft onto the frozen champion payload.

Run:
  PYTHONPATH=src python scratchpad/train_context_heads.py --contexts ACTIVATE
  PYTHONPATH=src python scratchpad/train_context_heads.py --contexts TO_HAND \
      --profile ALAKAZAM_FETCH --out data/models/ctx_tohand.json
"""

from __future__ import annotations

import argparse
import json
import sys
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
from ptcg_ai.imitation.policy import _score
from ptcg_ai.imitation.train import (
    _L2_GRID,
    Decision,
    _bc_accuracy,
    _featurize,
    _train_multi,
    _train_single,
)
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

sys.path.insert(0, "scratchpad")
from train_mlp_main import _mlp_scores, _spec, _train_mlp  # noqa: E402
from train_mlp_v4 import _expand_stages  # noqa: E402

DATASET = Path("data/imitation/yushinito_full.jsonl.gz")
DECK = Path("decks/yushinito.csv")
CHAMPION = Path("data/models/bc_alakazam_fetch.json")


# -- evaluation with the LIVE tie-break ---------------------------------------

def _rank_order(scores) -> list[int]:
    """Mirrors policy._rank_order: score-desc, deterministic lowest-index tie-break."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def _setmatch(decisions: list[Decision], score_fn) -> float:
    """Set-match top-k accuracy under the live ranking rule."""
    if not decisions:
        return 0.0
    correct = 0
    for d in decisions:
        k = len(d.chosen)
        pred = set(_rank_order(score_fn(d.X))[:k])
        correct += pred == set(d.chosen)
    return correct / len(decisions)


def _linear_fn(w):
    return lambda X: (X @ w).tolist()


def _mlp_fn(members):
    return lambda X: sum(_mlp_scores(X, P)[0] for P in members).tolist()


# -- baselines ----------------------------------------------------------------

def _greedy_accuracy(rows, ctx_name, parser, cards, greedy) -> tuple[float, int]:
    """Greedy's set-match accuracy on `ctx_name`, under _featurize's EXACT filters.

    The filters must match or the denominator differs from the model's and the
    comparison is meaningless.
    """
    ok = tot = 0
    for r in rows:
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None:
            continue
        if obs.select.context.name != ctx_name or F.is_prize_pick(obs.select):
            continue
        n = len(obs.select.option)
        chosen = [a for a in r.action if 0 <= a < n]
        if not chosen or len(set(chosen)) != len(chosen):
            continue
        try:
            pred = greedy.choose(DecisionContext(raw=r.raw_observation, observation=obs, cards=cards))
        except Exception:  # noqa: BLE001 — greedy failing IS the baseline's behaviour
            pred = []
        ok += set(pred) == set(chosen)
        tot += 1
    return (ok / tot if tot else 0.0), tot


def _featurize_ctx(profile, rows, ctx_name, parser, cards) -> list[Decision]:
    return _featurize(profile, rows, {ctx_name}, parser, cards, 1.0)[ctx_name]


def _shipped_baseline(payload, ctx_name, rows, parser, cards):
    """(accuracy, label) for the shipped spec on this context, or None if absent."""
    spec = (payload or {}).get("contexts", {}).get(ctx_name)
    if spec is None:
        return None
    prof_name = spec.get("profile") if isinstance(spec, dict) else None
    prof = get_profile(prof_name or payload.get("profile", "TR_650"))
    te = _featurize_ctx(prof, rows, ctx_name, parser, cards)
    acc = _setmatch(te, lambda X: [_score(spec, x) for x in X])
    kind = spec.get("kind", "linear") if isinstance(spec, dict) else "linear-v1"
    return acc, f"shipped {kind} @ {prof.name}/{prof.feature_dim}", len(te)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="ACTIVATE", help="comma-separated context names")
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--profile", default="ALAKAZAM")
    ap.add_argument("--baseline-weights", type=Path, default=CHAMPION)
    ap.add_argument("--h", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None, help="write winning specs here")
    ap.add_argument("--bar", type=float, default=0.02, help="TEST lift bar over the baseline")
    ap.add_argument("--l2", type=float, default=None,
                    help="pin the linear L2 instead of sweeping _L2_GRID. MAIN's optimum is "
                         "already known (the shipped payload records l2=1e-4), so pinning it "
                         "cuts the linear arm 3x on a 62k-decision context.")
    ap.add_argument("--skip-linear", action="store_true",
                    help="skip the linear arm entirely. On MAIN, the per-option matrix (every "
                         "option of every decision, not just 62k rows) is large enough that "
                         "400-epoch full-batch descent thrashes swap on a memory-constrained "
                         "box (M46: >2h stuck on ONE l2 value). Use when only the MLP result "
                         "is needed for a decision (e.g. a k-sweep), not the linear baseline.")
    args = ap.parse_args()

    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()
    deck_ids = [int(x) for x in DECK.read_text(encoding="utf-8").split()]
    greedy = GreedyPolicy(deck=deck_ids)
    profile = get_profile(args.profile)
    dim = profile.feature_dim
    baseline_payload = (
        json.loads(args.baseline_weights.read_text(encoding="utf-8"))
        if args.baseline_weights and args.baseline_weights.exists()
        else None
    )

    rows = list(read_decision_dataset(args.dataset))
    # THREE-way split by game, seeds 0/1 — the project-wide convention so every
    # ALAKAZAM experiment stays comparable (train_mlp_alakazam.py:44-45).
    temp_rows, test_rows = split_by_game(rows, val_fraction=0.2, seed=0)
    tr_rows, va_rows = split_by_game(temp_rows, val_fraction=0.25, seed=1)
    print(f"dataset={args.dataset.name}  rows={len(rows)}  "
          f"train={len(tr_rows)} val={len(va_rows)} test={len(test_rows)}")
    print(f"profile={profile.name} dim={dim}  h={args.h} seeds={args.seeds}  bar=+{args.bar:.3f}\n")

    out_specs: dict[str, dict] = {}
    for ctx_name in [c.strip() for c in args.contexts.split(",") if c.strip()]:
        print("=" * 78)
        print(f"CONTEXT {ctx_name}")
        tr = _featurize_ctx(profile, tr_rows, ctx_name, parser, cards)
        va = _featurize_ctx(profile, va_rows, ctx_name, parser, cards)
        te = _featurize_ctx(profile, test_rows, ctx_name, parser, cards)
        if not tr or not te:
            print(f"  no usable rows (train={len(tr)} test={len(te)}) — SKIP\n")
            continue
        multi = any(len(d.chosen) > 1 for d in tr)
        print(f"  featurized: train={len(tr)} val={len(va)} test={len(te)}  multi_pick={multi}")

        # ---- baseline: what we ship TODAY for this context ----
        shipped = _shipped_baseline(baseline_payload, ctx_name, test_rows, parser, cards)
        g_acc, g_n = _greedy_accuracy(test_rows, ctx_name, parser, cards, greedy)
        if shipped is not None:
            base_acc, base_label, _ = shipped
        else:
            base_acc, base_label = g_acc, "greedy fallback (context absent from payload)"
        print(f"  BASELINE  {base_acc:.4f}   [{base_label}]")
        if shipped is not None:
            print(f"    (greedy on the same rows: {g_acc:.4f}, n={g_n})")

        # ---- linear ----
        if args.skip_linear:
            lin_test = None
            print("  LINEAR    SKIPPED (--skip-linear)")
        else:
            trainer = _train_multi if multi else _train_single
            best_w, best_va = None, -1.0
            for l2 in ([args.l2] if args.l2 is not None else _L2_GRID):
                w = trainer(tr, dim, l2)
                a = _setmatch(va, _linear_fn(w))
                print(f"    linear l2={l2:<7} val={a:.4f}")
                if a > best_va:
                    best_w, best_va, best_l2 = w, a, l2
            lin_test = _setmatch(te, _linear_fn(best_w))
            print(f"  LINEAR    {lin_test:.4f}   (val {best_va:.4f}, l2={best_l2}, "
                  f"argsort-metric {_bc_accuracy(te, best_w):.4f})   lift {lin_test-base_acc:+.4f}")

        # ---- MLP ensemble ----
        tr_e = _expand_stages(tr) if multi else tr
        va_e = _expand_stages(va) if multi else va
        members = []
        for s in range(args.seeds):
            best_P, best_p_va = None, -1.0
            for l2 in (1e-4, 1e-3):
                P, acc = _train_mlp(tr_e, va_e, dim, args.h, l2, seed=s)
                if acc > best_p_va:
                    best_P, best_p_va = P, acc
            members.append(best_P)
            print(f"    mlp seed {s}: stage-val {best_p_va:.4f}  "
                  f"test(solo) {_setmatch(te, _mlp_fn([best_P])):.4f}")
        mlp_va = _setmatch(va, _mlp_fn(members))
        mlp_test = _setmatch(te, _mlp_fn(members))
        print(f"  MLP x{args.seeds}    {mlp_test:.4f}   (val {mlp_va:.4f})   lift {mlp_test-base_acc:+.4f}")

        # ---- verdict ----
        if lin_test is None or mlp_test >= lin_test:
            win_acc, win_spec, win_label = mlp_test, {
                "kind": "mlp_ensemble", "members": [_spec(P) for P in members]
            }, f"mlp_ensemble(k={args.seeds},h={args.h})"
        else:
            win_acc, win_spec, win_label = lin_test, {
                "kind": "linear", "w": [float(x) for x in best_w]
            }, "linear"
        if profile.name != (baseline_payload or {}).get("profile"):
            win_spec["profile"] = profile.name
        lift = win_acc - base_acc
        verdict = "PASS" if lift >= args.bar else "FAIL"
        print(f"\n  WINNER {win_label}  TEST {win_acc:.4f}  vs baseline {base_acc:.4f}")
        print(f"  GATE (lift >= +{args.bar:.3f}): {verdict}  ({lift:+.4f})\n")
        if verdict == "PASS":
            out_specs[ctx_name] = win_spec
            out_specs.setdefault("_metrics", {})[ctx_name] = {
                "test_acc": win_acc, "baseline_acc": base_acc, "baseline": base_label,
                "lift": lift, "kind": win_label, "n_test": len(te), "multi_pick": multi,
                "profile": profile.name,
            }

    if args.out and out_specs:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_specs), encoding="utf-8")
        wrote = [k for k in out_specs if not k.startswith("_")]
        print(f"wrote {args.out} ({args.out.stat().st_size/1024:.0f} KB) contexts={wrote}")
    elif args.out:
        print("nothing passed the gate — not writing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
