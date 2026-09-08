"""M31 Track A — data selection & weighting triage (LINEAR, dev-only). READ-ONLY wrt src/.

Every imitation-learning solution in the Lux AI S3 top-10 filtered its training data (keep won
games, drop decided/garbage states, drop trivial no-ops); the chess winner called data selection
"the most important aspect". We have never done it: 47.4% of our MAIN rows come from games Yushin
LOST, and `train.py`'s `--alpha` (down-weight lost-game decisions) has never been used.

This triages the variants with the LINEAR model (minutes) instead of the MLP (~4 h). Design:
  * featurize MAIN ONCE, reuse the matrices for every variant (featurization is the expensive part);
  * one FIXED L2 for all variants, so the comparison is like-for-like and costs 1 fit per variant;
  * variants change ONLY the training rows/weights — the evaluation sets are fixed and identical.

Reported per variant: accuracy on a fixed unfiltered TEST set AND on a won-games-only TEST set.
NOTE (see docs/m31_plan.md §3): filtering the training target changes what "fidelity" means, so a
drop on the unfiltered set is expected by construction for won-only variants — these numbers are
TRIAGE, and the real gate for Track A is head_to_head, not fidelity.

Run:  uv run --group dev python scratchpad/track_a_data_selection.py
"""

from __future__ import annotations

import collections
import time
from pathlib import Path

import numpy as np

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation import features as F
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.imitation.train import Decision, _bc_accuracy, _train_single
from ptcg_ai.observation.models import SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

DATASET = Path("data/imitation/yushinito_full.jsonl.gz")
PROFILE = get_profile("ALAKAZAM")
L2 = 1e-4          # fixed across variants (won the grid on this data) -> fair, 1 fit per variant
SEED = 0


class Rec:
    """One featurized MAIN decision plus the metadata the filters need."""

    __slots__ = ("X", "chosen", "won", "n_opt", "turn")

    def __init__(self, X, chosen, won, n_opt, turn):
        self.X, self.chosen, self.won, self.n_opt, self.turn = X, chosen, won, n_opt, turn


def featurize(rows, parser, cards) -> list[Rec]:
    out: list[Rec] = []
    for r in rows:
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None:
            continue
        if obs.select.context is not SelectContextKind.MAIN or F.is_prize_pick(obs.select):
            continue
        n = len(obs.select.option)
        chosen = [a for a in r.action if 0 <= a < n]
        if not chosen or len(set(chosen)) != len(chosen):
            continue
        gs = GameState.build(obs, cards)
        X = np.asarray(F.featurize_decision(PROFILE, obs.select, obs, gs, cards), dtype=np.float32)
        out.append(Rec(X, chosen, bool(r.won), n, int(obs.current.turn)))
    return out


def to_decisions(recs, weight_of) -> list[Decision]:
    """Build trainer inputs, dropping zero-weight rows (equivalent but cheaper)."""
    ds = []
    for r in recs:
        w = weight_of(r)
        if w > 0.0:
            ds.append(Decision(X=r.X, chosen=r.chosen, weight=w))
    return ds


def flatten_cap(recs, bucket_of, cap: int):
    """Chess-1st 'flatten the distribution': cap an over-represented bucket (deterministic)."""
    seen: collections.Counter = collections.Counter()
    keep = []
    for r in recs:
        b = bucket_of(r)
        seen[b] += 1
        if seen[b] <= cap:
            keep.append(r)
    return keep


def main() -> int:
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()

    rows = list(read_decision_dataset(DATASET))
    tr_rows, te_rows = split_by_game(rows, val_fraction=0.2, seed=SEED)
    t0 = time.perf_counter()
    tr = featurize(tr_rows, parser, cards)
    te = featurize(te_rows, parser, cards)
    te_won = [r for r in te if r.won]
    print(f"featurized MAIN in {time.perf_counter()-t0:.0f}s | train={len(tr)} "
          f"test={len(te)} test-won={len(te_won)} dim={PROFILE.feature_dim} L2={L2:g}", flush=True)
    print(f"train rows from WON games: {sum(r.won for r in tr)/max(len(tr),1):.1%}", flush=True)

    # fixed evaluation sets (never filtered)
    ev_all = to_decisions(te, lambda r: 1.0)
    ev_won = to_decisions(te_won, lambda r: 1.0)

    # turn>=20 is a 19.2% spike; cap it at the largest ordinary bucket for A4
    turn_counts = collections.Counter(min(r.turn, 20) for r in tr)
    ordinary_max = max(v for k, v in turn_counts.items() if k < 20) if len(turn_counts) > 1 else 0

    variants = [
        ("A0 baseline (all rows)", lambda rs: rs, lambda r: 1.0),
        ("A1 won-games only", lambda rs: rs, lambda r: 1.0 if r.won else 0.0),
        ("A2 alpha=0.5 (soft)", lambda rs: rs, lambda r: 1.0 if r.won else 0.5),
        ("A3 A2 + drop n_options<=2", lambda rs: [r for r in rs if r.n_opt > 2],
         lambda r: 1.0 if r.won else 0.5),
        ("A4 A2 + flatten turn>=20", lambda rs: flatten_cap(rs, lambda r: min(r.turn, 20), ordinary_max)
         if ordinary_max else rs, lambda r: 1.0 if r.won else 0.5),
    ]

    print(f"\n{'variant':<30}{'n_train':>9}{'TEST-all':>10}{'TEST-won':>10}")
    print("-" * 59)
    results = {}
    for name, filt, wfn in variants:
        sub = filt(tr)
        ds = to_decisions(sub, wfn)
        t1 = time.perf_counter()
        w = _train_single(ds, PROFILE.feature_dim, L2)
        a_all, a_won = _bc_accuracy(ev_all, w), _bc_accuracy(ev_won, w)
        results[name] = (a_all, a_won)
        print(f"{name:<30}{len(ds):>9}{a_all:>10.4f}{a_won:>10.4f}   [{time.perf_counter()-t1:.0f}s]",
              flush=True)

    base_all, base_won = results["A0 baseline (all rows)"]
    print("\ndeltas vs A0:")
    for name, (a, w_) in results.items():
        if name.startswith("A0"):
            continue
        print(f"  {name:<30} TEST-all {a-base_all:+.4f}   TEST-won {w_-base_won:+.4f}")
    print("\nREMINDER: TEST-all penalises won-only variants by construction (they deliberately stop "
          "modelling losing play). Treat as triage; the real gate is head_to_head (docs/m31_plan.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
