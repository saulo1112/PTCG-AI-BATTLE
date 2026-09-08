"""M47 — is the RL travelling, or just jittering? Two hypotheses, one cheap measurement.

THE QUESTION. Both M47 arms plateau at ~2-3% behavioural displacement from the BC init and
stop moving. Two very different causes fit that equally well from the outside:

  (A) THE ANCHOR BINDS. The L2 term contributes lambda*(P - P_bc), zero at the first
      update and growing with displacement, so the loop settles where it balances the
      policy gradient. Fix: lower lambda (the M47 brazo-B arm).

  (B) THE GRADIENT IS NOISE. 800 games give a policy-gradient estimate whose direction is
      mostly sampling noise, so consecutive steps partly cancel and net displacement grows
      like sqrt(T) instead of T. Fix: more games per batch (or pooling batches) -- lowering
      lambda would do nothing at all.

THE DISCRIMINATOR, and it needs no new games. Compare
    PATH LENGTH   sum_t |P_t - P_{t-1}|      (how far each step went)
    DISPLACEMENT  |P_T - P_0|                (how far we actually got)
Their ratio is the classic "efficiency" of a walk: ~1.0 means every step pointed the same
way (travelling), ~1/sqrt(T) means the steps were independent (jittering). Consecutive-step
cosine similarity says the same thing locally: ~0 is a random walk, >0 is coherent drift.

This reads checkpoints already on disk. Cost: seconds, no engine, no games.

Run:  PYTHONPATH=src python scratchpad/m47_step_coherence.py data/models/bc_alakazam_final.json \
          data/models/rl_alakazam_final_iter{1..6}.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def flat_main(path: str) -> np.ndarray:
    """Every MAIN parameter of every ensemble member, as one vector."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))["contexts"]["MAIN"]
    parts = []
    for m in spec["members"]:
        for k in ("W1", "b1", "w2", "b2"):
            parts.append(np.asarray(m[k], dtype=np.float64).ravel())
    return np.concatenate(parts)


def main() -> int:
    paths = sys.argv[1:]
    if len(paths) < 3:
        raise SystemExit(f"usage: {sys.argv[0]} <base.json> <iter1.json> ... (>=2 iters)")

    vecs = [flat_main(p) for p in paths]
    steps = [vecs[i + 1] - vecs[i] for i in range(len(vecs) - 1)]
    T = len(steps)

    print(f"{len(vecs[0])} MAIN parameters, {T} steps\n")
    print(f"{'step':<10}{'|step|':>12}{'|cum displ|':>14}{'cos(prev)':>12}")
    print("-" * 48)
    path_len = 0.0
    for i, s in enumerate(steps, start=1):
        path_len += float(np.linalg.norm(s))
        displ = float(np.linalg.norm(vecs[i] - vecs[0]))
        if i == 1:
            cos = float("nan")
        else:
            a, b = steps[i - 2], steps[i - 1]
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        cos_txt = "     -" if math.isnan(cos) else f"{cos:>+12.3f}"
        print(f"{Path(paths[i]).stem[-6:]:<10}{np.linalg.norm(s):>12.5f}{displ:>14.5f}{cos_txt}")

    displ = float(np.linalg.norm(vecs[-1] - vecs[0]))
    eff = displ / max(path_len, 1e-12)
    rw = 1.0 / math.sqrt(T)
    cosines = [float(steps[i - 1] @ steps[i] /
                     (np.linalg.norm(steps[i - 1]) * np.linalg.norm(steps[i]) + 1e-12))
               for i in range(1, T)]
    mean_cos = float(np.mean(cosines)) if cosines else float("nan")

    print("-" * 48)
    print(f"path length          {path_len:.5f}")
    print(f"net displacement     {displ:.5f}")
    print(f"EFFICIENCY           {eff:.3f}   (1.0 = perfectly straight, "
          f"{rw:.3f} = independent random steps at T={T})")
    print(f"mean cos(consecutive){mean_cos:>+8.3f}   (0 = no direction memory)")
    print()
    if eff > 2.5 * rw and mean_cos > 0.15:
        print("VERDICT: the steps are COHERENT -- the policy is travelling in a consistent")
        print("  direction, just slowly. That is consistent with hypothesis (A), the anchor")
        print("  (or the step size) limiting how far it gets, and lowering lambda / raising")
        print("  lr should show up as more displacement.")
    elif eff < 1.6 * rw or mean_cos < 0.05:
        print("VERDICT: the steps are essentially a RANDOM WALK -- consecutive updates point")
        print("  in unrelated directions and cancel. That is hypothesis (B): the gradient")
        print("  estimate from this batch size is dominated by sampling noise. Lowering")
        print("  lambda cannot help; only more games per update (or pooling batches) can.")
    else:
        print("VERDICT: AMBIGUOUS -- between a random walk and coherent drift. Treat the")
        print("  lambda arm as the tiebreaker rather than concluding from this alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
