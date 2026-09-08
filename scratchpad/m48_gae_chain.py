"""M48 Fase 1.3 — the MECHANICAL gate: does TD/GAE stop the updates from cancelling?

THE SETUP IS A REPLAY, NOT A NEW EXPERIMENT. M47's brazo A chained six updates from the
shipped champion over six self-play batches with clip=0.2 / lr=1e-4 / tau=0.3 /
lambda_a=0.1 / adv_norm, and `m47_step_coherence.py` measured the result:

    efficiency 0.214  (0.408 = independent random steps at T=6)
    mean cosine between consecutive steps  -0.283   (all five pairs negative)

This re-runs that chain on the SAME six batches with the SAME hyperparameters, changing
exactly ONE thing: the advantage estimator. No new games are played.

WHY IT SHOULD HELP, in one line: `m48_advantage_variance.py` measured that 87.5% of the
variance of A = R - V(s) is BETWEEN games -- identical for all ~45 decisions of a game, so
it says nothing about which decision was good, and it re-rolls with every batch. TD
advantages are within-game by construction.

PRE-REGISTERED GATE (fixed before running, so a near-miss cannot be talked into a pass):

    mean cosine  -0.283  ->  >= 0.00      the steps stop cancelling
    efficiency    0.214  ->  >= 0.40      net travel reaches at least a random walk

If no lambda clears both, the RL line closes HERE -- with no arena time and no new games.
That is the cheap-signal-first rule M47 broke at a cost of 95 minutes.

HONEST CAVEAT to carry into the write-up either way: GAE trades outcome noise for CRITIC
ERROR, and lambda=0 is the maximum-bias end of that trade. This gate therefore measures
COHERENCE (did the direction become consistent?), not fidelity or strength. Only the n=600
arena decides strength.

Run:  PYTHONPATH=src python -u scratchpad/m48_gae_chain.py [--lambdas 0.0,0.5,0.95]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rl_selfplay as R  # noqa: E402
from ptcg_ai.observation.parser import ObservationParser  # noqa: E402

BATCHES = [f"data/rl_alakazam_final/iter{i}_traj.jsonl.gz" for i in range(1, 7)]
CRITIC = "data/models/v_alakazam_v2.json"
# M47 brazo A, verbatim. Changing any of these would stop this being a controlled replay.
HP = dict(lambda_a=0.10, tau=0.3, lr=1e-4, epochs=10, clip=0.2, adv_norm=True)
BASELINE_COS, BASELINE_EFF = -0.283, 0.214
GATE_COS, GATE_EFF = 0.00, 0.40


def _flat(members):
    return np.concatenate([np.asarray(P[k], dtype=np.float64).ravel()
                           for P in members for k in ("W1", "b1", "w2", "b2")])


def _coherence(vecs):
    steps = [vecs[i + 1] - vecs[i] for i in range(len(vecs) - 1)]
    path = sum(float(np.linalg.norm(s)) for s in steps)
    displ = float(np.linalg.norm(vecs[-1] - vecs[0]))
    cos = [float(steps[i - 1] @ steps[i] /
                 (np.linalg.norm(steps[i - 1]) * np.linalg.norm(steps[i]) + 1e-12))
           for i in range(1, len(steps))]
    return displ / max(path, 1e-12), (float(np.mean(cos)) if cos else float("nan")), path, displ


def _between_share(prepared, adv):
    by_game = collections.defaultdict(list)
    for (gid, *_), a in zip(prepared, adv):
        by_game[gid].append(a)
    adv = np.asarray(adv)
    grand = adv.mean()
    sizes = np.array([len(v) for v in by_game.values()], dtype=np.float64)
    means = np.array([np.mean(v) for v in by_game.values()])
    between = float((sizes * (means - grand) ** 2).sum() / len(adv))
    within = float(sum(((np.asarray(v) - np.mean(v)) ** 2).sum() for v in by_game.values())
                   / len(adv))
    return between / max(between + within, 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", default="0.0,0.5,0.95")
    ap.add_argument("--tag", default="gae")
    args = ap.parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",")]

    R.configure("alakazam_final")
    _, cards = R._sdk_cards()
    parser = ObservationParser()
    v = R._load_v650(CRITIC)
    base_payload, members_bc = R._load_main_members(R.BASE_CKPT)

    missing = [b for b in BATCHES if not Path(b).is_file()]
    if missing:
        raise SystemExit(f"missing batches: {missing}")

    # Featurise ONCE. None of it depends on lambda, and it is ~90% of the wall clock.
    print("featurising the six batches (shared across every lambda arm)...", flush=True)
    prepared = []
    for b in BATCHES:
        t0 = time.perf_counter()
        p = R._prepare_traj(b, cards, parser, v)
        prepared.append(p)
        print(f"  {Path(b).name:<28} {len(p):>7} decisions  {time.perf_counter() - t0:>5.0f}s",
              flush=True)

    print(f"\nBASELINE (M47 brazo A, Monte-Carlo advantage): "
          f"efficiency {BASELINE_EFF:.3f}, mean cos {BASELINE_COS:+.3f}")
    print(f"GATE: efficiency >= {GATE_EFF:.2f} AND mean cos >= {GATE_COS:+.2f}\n")

    results = []
    for lam in lambdas:
        label = "MC (control)" if lam is None else f"lambda={lam}"
        members = [{k: np.copy(np.asarray(P[k], dtype=np.float64)) for k in P}
                   for P in members_bc]
        vecs = [_flat(members)]
        shares = []
        t0 = time.perf_counter()
        for i, p in enumerate(prepared, start=1):
            decs = R.advantages(p, gae_lambda=lam)
            shares.append(_between_share(p, [d.weight for d in decs]))
            members = R.rl_update(members, members_bc, decs, **HP)
            vecs.append(_flat(members))
        eff, cos, path, displ = _coherence(vecs)
        passed = (cos >= GATE_COS) and (eff >= GATE_EFF)
        out = f"data/models/rl_{args.tag}_lam{lam}_iter6.json"
        R._write_ckpt(base_payload, members, out,
                      meta={**HP, "gae_lambda": lam, "chained_over": BATCHES,
                            "critic": CRITIC})
        results.append((lam, eff, cos, np.mean(shares), out, passed))
        print(f"{label:<14} between-game share {np.mean(shares):>6.1%}   "
              f"efficiency {eff:>6.3f}   mean cos {cos:>+7.3f}   "
              f"{'PASS' if passed else 'fail'}   ({time.perf_counter() - t0:.0f}s)",
              flush=True)

    print("\n" + "=" * 92)
    print(f"{'lambda':>8}{'between-game':>15}{'efficiency':>13}{'mean cos':>11}"
          f"{'vs M47 cos':>13}  checkpoint")
    print("-" * 92)
    print(f"{'MC (M47)':>8}{'87.5%':>15}{BASELINE_EFF:>13.3f}{BASELINE_COS:>+11.3f}"
          f"{'--':>13}  data/models/rl_alakazam_final_iter6.json")
    for lam, eff, cos, share, out, passed in results:
        print(f"{lam:>8.2f}{share:>15.1%}{eff:>13.3f}{cos:>+11.3f}"
              f"{cos - BASELINE_COS:>+13.3f}  {Path(out).name}")
    print("=" * 92)
    winners = [r for r in results if r[5]]
    if winners:
        best = max(winners, key=lambda r: r[2])
        print(f"GATE PASSED by lambda={best[0]} -> take {best[4]} to the n=600 arena.")
    else:
        print("GATE FAILED for every lambda. The steps still cancel, so the advantage")
        print("estimator was NOT the binding constraint. Close the RL line here: no arena,")
        print("no new games, and write the post-mortem with these numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
