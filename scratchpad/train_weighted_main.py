"""M39 — weight MAIN decisions by how much the combo clock depends on them.

WHY. Every architecture lever is closed (M31 width, M36 data, M37 set transformer:
+0.017 strict against a +0.030 bar). M38 measured the reason: the Bayes ceiling of this
representation is 0.968 and we sit at 0.786, but per-decision accuracy is the wrong
target -- at 49 MAIN decisions/game the clone leaves Yushin's line every ~4.7 decisions
and then plays where there is no data. DAgger would fix it; we have no oracle. Self-play
RL is the substitute and M38 measured it WORSE (0.438 head-to-head vs the frozen init at
n=300, after reweighting; never better than a tie across four evaluations).

This attacks the SYMPTOM instead. `_featurize` currently assigns every decision
weight 1.0 (won) or alpha (lost) -- a lost game's ATTACH on turn 2 counts exactly as much
as the play that sets up the win condition. M33 measured where the real gap is:

  * Yushin fires Powerful Hand on 99.6% of the turns it is LEGAL (mean delay +0.00
    turns), so there is no "when to attack" decision to learn at all;
  * the gap is ASSEMBLY SPEED -- Powerful Hand becomes legal on turn 4.56 for him and
    5.46 for the clone, and his own won/lost split is 4.21 vs 4.95.

So the decisions that decide games are the ones BEFORE Powerful Hand first becomes legal:
the Abra/Kadabra/Alakazam + Rare Candy + energy sequencing that buys that turn. Those get
up-weighted here; everything after gets the normal weight.

This is a patch on the symptom, not a cure -- the clone still diverges every ~4.7
decisions. The hypothesis is only that it will diverge on cheaper decisions.

Run:
  PYTHONPATH=src python -u scratchpad/train_weighted_main.py --boost 3.0
  PYTHONPATH=src python -u scratchpad/train_weighted_main.py --boost 1.0   # control
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import ptcg_ai.imitation.features as F
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.imitation.train import Decision, _bc_accuracy, _L2_GRID, _train_single
from ptcg_ai.observation.models import OptionKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState
from train_mlp_main import _mlp_accuracy, _train_mlp

POWERFUL_HAND = 1072          # verified in diagnose_mlp.py:47
DATASET = Path("data/imitation/yushinito_full.jsonl.gz")


def _ph_legal_turn(rows, parser):
    """Per game: the first turn Powerful Hand appears as a legal option.

    Mirrors analyze_ph_availability.py's detection exactly (option.type is ATTACK and
    option.attackId == POWERFUL_HAND) so the boost lands on the same decisions M33
    measured, not on a re-derived approximation.
    """
    first = {}
    for r in rows:
        if r.context != int(SelectContextKind.MAIN):
            continue
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None:
            continue
        if any(o.type is OptionKind.ATTACK and o.attackId == POWERFUL_HAND
               for o in obs.select.option):
            g = r.game_id
            t = obs.current.turn
            if g not in first or t < first[g]:
                first[g] = t
    return first


def featurize_weighted(profile, rows, parser, cards, alpha, boost, ph_turn):
    """Same as train._featurize for MAIN, but multiplies the weight of every decision
    taken BEFORE the game's Powerful-Hand-legal turn by `boost`."""
    out, n_boosted, n_total = [], 0, 0
    for r in rows:
        if r.context != int(SelectContextKind.MAIN):
            continue
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None or F.is_prize_pick(obs.select):
            continue
        n = len(obs.select.option)
        chosen = [a for a in r.action if 0 <= a < n]
        if not chosen or len(set(chosen)) != len(chosen):
            continue
        gs = GameState.build(obs, cards)
        X = np.asarray(F.featurize_decision(profile, obs.select, obs, gs, cards),
                       dtype=np.float32)
        w = 1.0 if r.won else alpha
        cutoff = ph_turn.get(r.game_id)
        n_total += 1
        if cutoff is not None and obs.current.turn < cutoff:
            w *= boost
            n_boosted += 1
        out.append(Decision(X=X, chosen=chosen, weight=w))
    return out, n_boosted, n_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boost", type=float, default=3.0,
                    help="weight multiplier for pre-Powerful-Hand decisions (1.0 = control)")
    ap.add_argument("--h", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    profile = get_profile("ALAKAZAM")
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()

    rows = list(read_decision_dataset(DATASET))
    # IDENTICAL 3-way split to train_mlp_alakazam.py, so numbers are comparable to the
    # champion's. Changing the split would make any delta uninterpretable.
    temp_rows, test_rows = split_by_game(rows, val_fraction=0.2, seed=0)
    tr_rows, va_rows = split_by_game(temp_rows, val_fraction=0.25, seed=1)

    print("computing Powerful-Hand-legal turn per game...", flush=True)
    ph_turn = _ph_legal_turn(rows, parser)
    print(f"  games with a PH-legal turn: {len(ph_turn)}", flush=True)

    print(f"featurizing (boost={args.boost})...", flush=True)
    tr, nb, nt = featurize_weighted(profile, tr_rows, parser, cards, args.alpha,
                                    args.boost, ph_turn)
    print(f"  train: {len(tr)} decisions, {nb}/{nt} boosted ({nb / max(nt,1):.1%})", flush=True)
    # val/test are NEVER weighted -- they measure plain accuracy, so a boost cannot
    # flatter its own evaluation.
    va, _, _ = featurize_weighted(profile, va_rows, parser, cards, 1.0, 1.0, {})
    te, _, _ = featurize_weighted(profile, test_rows, parser, cards, 1.0, 1.0, {})
    print(f"  val={len(va)} test={len(te)}\n", flush=True)

    dim = profile.feature_dim
    best_lin, best_va = None, -1.0
    for l2 in _L2_GRID:
        w = _train_single(tr, dim, l2)
        a = _bc_accuracy(va, w)
        if a > best_va:
            best_lin, best_va = w, a
    print(f"linear: val={best_va:.4f} test={_bc_accuracy(te, best_lin):.4f}", flush=True)

    members = []
    for s in range(args.seeds):
        P, va_acc = _train_mlp(tr, va, dim, args.h, 1e-4, seed=s)
        members.append(P)
        print(f"  mlp seed {s}: val={va_acc:.4f}", flush=True)
    print(f"\nMLP ensemble (h={args.h}, {args.seeds} seeds), boost={args.boost}")
    print(f"  VAL  = {_mlp_accuracy(va, members):.4f}")
    print(f"  TEST = {_mlp_accuracy(te, members):.4f}   <- compare against the control run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
