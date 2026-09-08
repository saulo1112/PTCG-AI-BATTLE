"""M13 Phase B gate G-B2: leave-archetype-out CV for the TO_HAND MLP.

Same honest-OOD instrument the residual trainer used for MAIN (train_residual_main.py):
hold out one opponent archetype at a time, train linear vs MLP on the rest, compare
held-out set-match accuracy. A TO_HAND MLP earns its place only if it beats linear in
mean OOD AND is not worse than linear on any single fold by > 0.01 (the bar that
correctly rejected the residual). Reuses archetype_map() verbatim.

Run:  uv run --group dev python scratchpad/cv_archetype_tohand.py
"""

from __future__ import annotations

import sys

import numpy as np

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation import features as F
from ptcg_ai.imitation.dataset import read_decision_dataset
from ptcg_ai.imitation.train import _bc_accuracy, _train_multi, Decision
from ptcg_ai.observation.models import SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

sys.path.insert(0, "scratchpad")
from train_residual_main import archetype_map, DATASET, PROFILE, FOLDS  # noqa: E402
from train_mlp_v4 import _train_mlp_ensemble, _mlp_setmatch  # noqa: E402


def featurize_tohand(parser, cards):
    """TO_HAND decisions as (game_id, Decision), multi-pick preserved."""
    out = []
    for r in read_decision_dataset(DATASET):
        if SelectContextKind(r.context) is not SelectContextKind.TO_HAND:
            continue
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None or F.is_prize_pick(obs.select):
            continue
        n = len(obs.select.option)
        chosen = [a for a in r.action if 0 <= a < n]
        if not chosen or len(set(chosen)) != len(chosen):
            continue
        gs = GameState.build(obs, cards)
        X = np.asarray(F.featurize_decision(PROFILE, obs.select, obs, gs, cards), dtype=np.float32)
        out.append((r.game_id, Decision(X=X, chosen=chosen, weight=1.0)))
    return out


def main() -> int:
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()
    dim = PROFILE.feature_dim

    amap = archetype_map()
    data = featurize_tohand(parser, cards)
    by_fold = {g: sum(1 for gid, _ in data if amap.get(gid, "other") == g) for g in FOLDS}
    print(f"TO_HAND decisions: {len(data)}  dim={dim}")
    print("held-out sizes:", by_fold)

    print(f"\n{'held-out archetype':<20}{'linear':>9}{'MLP':>9}")
    print("-" * 38)
    lin_accs, mlp_accs = [], []
    for held in FOLDS:
        tr = [d for gid, d in data if amap.get(gid, "other") != held]
        te = [d for gid, d in data if amap.get(gid, "other") == held]
        if not te:
            continue
        # linear baseline
        best_w, best_acc = None, -1.0
        for l2 in (1e-4, 1e-3, 1e-2):
            w = _train_multi(tr, dim, l2)
            a = _bc_accuracy(tr, w)  # in-fold selection proxy (no separate val here)
            if a > best_acc:
                best_w, best_acc = w, a
        a_lin = _bc_accuracy(te, best_w)
        # MLP ensemble (stage-expanded), val = train (small folds); report on held-out
        members = _train_mlp_ensemble(tr, tr, dim, h=16, n_seeds=3, expand=True)
        a_mlp = _mlp_setmatch(te, members)
        lin_accs.append(a_lin); mlp_accs.append(a_mlp)
        print(f"{held:<20}{a_lin:>9.4f}{a_mlp:>9.4f}")
    print("-" * 38)
    ml, mm = np.mean(lin_accs), np.mean(mlp_accs)
    print(f"{'MEAN OOD':<20}{ml:>9.4f}{mm:>9.4f}")
    worst = min(m - l for m, l in zip(mlp_accs, lin_accs))
    print(f"\nMLP vs linear: mean {mm - ml:+.4f}, worst single fold {worst:+.4f}")
    passed = mm >= ml and worst > -0.01
    print(f"G-B2 (MLP >= linear mean AND no fold worse than linear by >0.01)? "
          f"{'PASS' if passed else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
