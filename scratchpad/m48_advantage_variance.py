"""M48 — how much of the RL learning signal is the DECISION, and how much is the GAME?

THE QUESTION M47 LEFT OPEN. Every M47 candidate failed the arena, and
`m47_step_coherence.py` explained the mechanism: consecutive updates point in opposite
directions (mean cosine -0.283, efficiency 0.214 against 0.408 for an independent random
walk) so five sixths of the path length cancels. It did not explain WHY the direction
reverses.

The suspect is the advantage estimator. `rl_selfplay._featurize_traj` computes

    A_t = R - V(s_t)

with R the FINAL game outcome in {0, 1}. That is a Monte-Carlo return: every one of the
~45 MAIN decisions in a game is credited or blamed with the same R. If V(s_t) explains
little of R, then A_t is dominated by "which game did this come from", identically across
the whole game, and the gradient degenerates into "do more of whatever you did in games
you happened to win" -- a direction that re-rolls with every batch.

THE MEASUREMENT, and it needs no new games. Decompose the variance of A by game:

    Var(A) = Var_between(mean A per game) + Var_within(residual inside a game)

`between` is outcome noise -- it carries no information about WHICH decision was good.
`within` is the only part that can discriminate one decision from another. The ratio is
the honest ceiling on what this estimator can teach per decision.

Reported for BOTH critics, because M47 refit the critic (held-out AUC on self-play states
0.6526 -> 0.7902) and the interesting question is whether tripling the critic's quality
actually moved the credit-assignment needle, or just shaved the variance a little.

Also reports the same decomposition for the TD/GAE alternative once `gae_lambda` exists,
so the fix can be judged on the same scale as the defect.

Run:  PYTHONPATH=src python -u scratchpad/m48_advantage_variance.py [traj.jsonl.gz]
"""

from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rl_selfplay as R  # noqa: E402
from ptcg_ai.imitation import features as F  # noqa: E402
from ptcg_ai.observation.parser import ObservationParser  # noqa: E402

TRAJ = sys.argv[1] if len(sys.argv) > 1 else "data/rl_alakazam_final/calib_tau03.jsonl.gz"
CRITICS = [("old  v_alakazam", "data/models/v_alakazam.json"),
           ("new  v_alakazam_v2", "data/models/v_alakazam_v2.json")]


def _rows(traj_path):
    """(game_id, won, obs) per usable MAIN decision, in file order."""
    parser = ObservationParser()
    out = []
    with gzip.open(traj_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if int(r["context"]) != int(R.MAIN_CTX):
                continue
            obs = parser.parse(r["raw_observation"])
            if obs.select is None or obs.current is None or F.is_prize_pick(obs.select):
                continue
            n = len(obs.select.option)
            chosen = [a for a in r["action"] if 0 <= a < n]
            if len(chosen) != 1:
                continue
            out.append((r["game_id"], bool(r["won"]), obs))
    return out


def decompose(adv, game_ids, label):
    """Var(A) = between-game + within-game, printed as shares."""
    adv = np.asarray(adv, dtype=np.float64)
    by_game = collections.defaultdict(list)
    for a, g in zip(adv, game_ids):
        by_game[g].append(a)
    grand = adv.mean()
    means = np.array([np.mean(v) for v in by_game.values()])
    sizes = np.array([len(v) for v in by_game.values()], dtype=np.float64)
    between = float((sizes * (means - grand) ** 2).sum() / len(adv))
    within = float(sum(((np.asarray(v) - np.mean(v)) ** 2).sum() for v in by_game.values())
                   / len(adv))
    total = between + within
    print(f"{label:<22}{adv.std():>9.4f}{between / total:>13.1%}{within / total:>13.1%}"
          f"{len(by_game):>9}{len(adv) / len(by_game):>10.1f}")
    return between / total, within / total


def main() -> int:
    R.configure("alakazam_final")
    _, cards = R._sdk_cards()

    rows = _rows(TRAJ)
    game_ids = [g for g, _, _ in rows]
    wins = np.array([1.0 if w else 0.0 for _, w, _ in rows])
    p = float(np.mean([1.0 if w else 0.0
                       for w in {g: w for g, w, _ in rows}.values()]))
    print(f"\n{TRAJ}")
    print(f"{len(rows)} MAIN decisions over {len(set(game_ids))} games, game win rate {p:.3f}")
    print(f"pure-outcome std sqrt(p(1-p)) = {np.sqrt(p * (1 - p)):.4f}   "
          f"-- the std of A if the critic were a constant\n")

    print(f"{'estimator':<22}{'std(A)':>9}{'BETWEEN game':>13}{'within game':>13}"
          f"{'games':>9}{'dec/game':>10}")
    print("-" * 76)
    # Reference row: no critic at all (A = R - mean). Pure outcome noise by construction,
    # so its `within` share is the floor the critics have to beat.
    decompose(wins - wins.mean(), game_ids, "no critic (A = R - p)")

    for label, path in CRITICS:
        v = R._load_v650(path)
        adv = [(1.0 if won else 0.0) - R._value(obs, cards, v) for _, won, obs in rows]
        share_b, _ = decompose(adv, game_ids, label)
        explained = 1.0 - (np.std(adv) / np.sqrt(p * (1 - p))) ** 2
        print(f"{'':<22}{'':>9}explains {explained:.1%} of the outcome variance")

    print("-" * 76)
    print("BETWEEN-game share = the fraction of the learning signal that is identical for")
    print("every decision in a game, i.e. carries NO information about which decision was")
    print("good. A Monte-Carlo return cannot get this below the game's own outcome noise.")
    print("TD/GAE targets exactly this column: A_t = V(s_t+1) - V(s_t) is within-game by")
    print("construction, trading outcome noise for critic error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
