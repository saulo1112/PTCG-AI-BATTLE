"""M48 Fase 2.1 -- fit a critic V(state) -> P(win) on REAL LADDER PLAY, not self-play.

WHY THIS EXISTS. M18 closed determinized search with a specific, named cause: the leaf
value was fit on TR-650 SELF-PLAY states and was mis-calibrated out of that distribution
(+0.084 vs Bellibolt, an energy-tempo deck like the self-play mix; -0.053/-0.067 vs
Lucario/Cinderace, which are not). M47 refit a critic (v_alakazam_v2, AUC 0.79) but ALSO
only on self-play against 3 fixed opponents (Grimmsnarl-clone/mirror/Kangaskhan-clone) --
repeating M18's exact setup with a different deck would be the same mistake in a new
costume.

This fits on the field the agent actually meets: Yushin's own 2330-game corpus (the
teacher, same deck, full archetype spread) plus every one of OUR real submissions'
replays with the yushinito deck. Held out: the 135 replays of `imitation-final`
(submissions 55438655 + 55438687) -- the exact agent, the exact field, never touched
during fitting. That is the condition M18 named for reopening this line.

EXCLUDED: replays/55203764 (M37: wrong deck shipped, 0/9 profile vocabulary cards --
those states are not this agent's play, mixing them in would corrupt the fit, not help it).

Extractor is `kaggle_replay.iter_player_decisions` -- the VETTED one. A raw sweep that
reads `cell['action']` from the SAME step reports the OPPOSITE of the real signal (M42's
method trap); this project has been burned by that exact bug once already.

Fits base-7 (`train_value.features`), not the 10-feature vf2 set M47 used for the RL
critic: M47 measured vf2 beating base-7 by only +0.0056 AUC on self-play states, and
`search_bc.LearnedEvaluator` (the un-subclassed base) scores base-7 -- using it avoids the
zip-truncation trap this milestone closed in `search_bc.py`, by construction, on the
critic that actually ships to search.

GATE: adopt only if held-out AUC on the ladder validation set beats `v_alakazam.json`
(fit on Yushin only) ON THE SAME rows. If it does not, search is not built -- the whole
premise was that a broader-field critic generalizes where the self-play one did not.

Run:  PYTHONPATH=src python -u scratchpad/m48_fit_ladder_critic.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_value import FEATURE_NAMES, _auc, _fit_logreg, features  # noqa: E402

from ptcg_ai.cards.database import CardDatabase  # noqa: E402
from ptcg_ai.config import load_config  # noqa: E402
from ptcg_ai.environment.sdk import load_sdk  # noqa: E402
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game  # noqa: E402
from ptcg_ai.imitation.kaggle_replay import iter_player_decisions  # noqa: E402
from ptcg_ai.observation.parser import ObservationParser  # noqa: E402

OUR = "Saulo Quiñones Góngora"
TEACHER_DATASET = Path("data/imitation/yushinito_full.jsonl.gz")
# Our own submissions' replays with the yushinito deck, in chronological order.
# 55203764 is DELIBERATELY excluded (M37: wrong deck shipped, not this agent's play).
TRAIN_REPLAY_DIRS = ["replays/55145833", "replays/55369527", "replays/55369569"]
HELDOUT_REPLAY_DIRS = ["replays/55438655", "replays/55438687"]   # imitation-final
INCUMBENT = "data/models/v_alakazam.json"
OUT = "data/models/v_alakazam_ladder.json"
MAIN_CTX = None  # resolved after ObservationParser import, see main()


def _build_from_teacher(parser, cards, val_fraction=0.15, seed=0):
    """Yushin's corpus already carries `won` per decision AND is context-tagged, so it
    can reuse the standard dataset reader instead of the replay-JSON walker."""
    rows = list(read_decision_dataset(TEACHER_DATASET))
    tr_rows, va_rows = split_by_game(rows, val_fraction=val_fraction, seed=seed)
    return tr_rows, va_rows


def _decisions_from_dir(folder: Path):
    for f in sorted(Path(folder).glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        yield from iter_player_decisions(f, OUR)


def _featurize(rows, parser, cards, ctx_filter=None):
    X, y = [], []
    for r in rows:
        if ctx_filter is not None and int(r.context) != ctx_filter:
            continue
        obs = parser.parse(r.raw_observation)
        f = features(obs, cards)
        if f is None:
            continue
        X.append(f)
        y.append(1.0 if r.won else 0.0)
    return np.asarray(X), np.asarray(y)


def main() -> int:
    from ptcg_ai.observation.models import SelectContextKind
    main_ctx = int(SelectContextKind.MAIN)

    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()

    t0 = time.perf_counter()
    print("building TRAIN set...")
    teacher_tr, teacher_va = _build_from_teacher(parser, cards)
    Xtr, ytr = _featurize(teacher_tr, parser, cards, ctx_filter=main_ctx)
    print(f"  teacher (train split): {len(ytr)} rows  win rate {ytr.mean():.3f}")
    Xva_t, yva_t = _featurize(teacher_va, parser, cards, ctx_filter=main_ctx)
    print(f"  teacher (val split, folded into train): {len(yva_t)} rows")
    Xtr, ytr = np.concatenate([Xtr, Xva_t]), np.concatenate([ytr, yva_t])

    for d in TRAIN_REPLAY_DIRS:
        rows = list(_decisions_from_dir(Path(d)))
        Xd, yd = _featurize(rows, parser, cards, ctx_filter=main_ctx)
        n_games = len({r.game_id for r in rows})
        print(f"  {d}: {n_games} games, {len(yd)} MAIN rows, win rate "
              f"{yd.mean() if len(yd) else float('nan'):.3f}")
        if len(yd):
            Xtr, ytr = np.concatenate([Xtr, Xd]), np.concatenate([ytr, yd])

    print(f"\nbuilding HELD-OUT set (imitation-final's own replays, never touched above)...")
    Xva, yva = [], []
    for d in HELDOUT_REPLAY_DIRS:
        rows = list(_decisions_from_dir(Path(d)))
        Xd, yd = _featurize(rows, parser, cards, ctx_filter=main_ctx)
        n_games = len({r.game_id for r in rows})
        print(f"  {d}: {n_games} games, {len(yd)} MAIN rows, win rate "
              f"{yd.mean() if len(yd) else float('nan'):.3f}")
        if len(yd):
            Xva.append(Xd); yva.append(yd)
    Xva, yva = np.concatenate(Xva), np.concatenate(yva)

    print(f"\nTOTAL: train={len(ytr)} rows, held-out={len(yva)} rows  "
          f"({time.perf_counter() - t0:.0f}s)\n")

    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    w, b = _fit_logreg((Xtr - mu) / sd, ytr)
    new_auc = _auc(((Xva - mu) / sd) @ w + b, yva)

    old = json.loads(Path(INCUMBENT).read_text(encoding="utf-8"))
    omu, osd = np.asarray(old["mean"]), np.asarray(old["std"])
    ow, ob = np.asarray(old["weights"]), float(old["bias"])
    old_auc = _auc(((Xva - omu) / osd) @ ow + ob, yva)

    print(f"held-out AUC (imitation-final's own {len(yva)} MAIN decisions):")
    print(f"  incumbent v_alakazam.json (fit on Yushin ONLY)  : {old_auc:.4f}")
    print(f"  candidate (fit on Yushin + our own ladder games): {new_auc:.4f}  "
          f"({new_auc - old_auc:+.4f})")

    for i, name in enumerate(FEATURE_NAMES):
        print(f"    {name:<10} {w[i]:+.3f}")

    if new_auc > old_auc:
        Path(OUT).write_text(json.dumps({
            "version": 1, "features": list(FEATURE_NAMES),
            "mean": mu.tolist(), "std": sd.tolist(), "weights": w.tolist(), "bias": float(b),
            "val_auc": new_auc,
            "trained_on": f"yushinito_full.jsonl.gz + {TRAIN_REPLAY_DIRS}, "
                          f"held out {HELDOUT_REPLAY_DIRS}",
        }), encoding="utf-8")
        print(f"\nADOPTED -> {OUT}")
        print("Fase 2.3/2.4 (cost probe, search arena) may proceed.")
    else:
        print(f"\nNOT adopted -- {Path(INCUMBENT).name} still wins on the ladder held-out set.")
        print("The premise (a broader-field critic generalizes where self-play did not) did")
        print("NOT hold. Do not build the search arena on this critic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
