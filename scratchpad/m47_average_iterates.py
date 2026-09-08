"""M47 — Polyak-average the RL iterates, because they were measured to oscillate.

WHY. `m47_step_coherence.py` measured, on the six brazo-A checkpoints:

    efficiency (net displacement / path length) = 0.214, against 0.408 for INDEPENDENT
    random steps at T=6, and mean cosine between consecutive steps = -0.283 (all five
    pairs negative).

That is not slow travel and it is not noise: it is systematic oscillation. Adam
normalises the gradient, so every iteration takes a step of nearly the same length
(0.138-0.149 here) whatever the signal is; when the direction reverses each time, five
sixths of the path length cancels out.

The textbook response to an oscillating iterate sequence is to average it: the
oscillating component cancels and the coherent drift survives. It costs one pass over
files already on disk and zero games, so it is the cheapest remaining shot -- and unlike
another training run it CANNOT fail for a new reason, it either extracts a signal that is
already there or shows there was none.

The average is taken over MAIN only, member-by-member and parameter-by-parameter, and
written onto the champion payload, so the other 11 contexts, `profile_overrides` and
`count_heads` come through untouched (verify with m47_verify_payload.py).

Averaging weights across independently-trained nets would be meaningless; averaging along
ONE optimisation trajectory is not -- these iterates are all within 0.18 of the same
starting point in a 95k-dimensional space, i.e. the same basin.

Run:
  PYTHONPATH=src python scratchpad/m47_average_iterates.py \
      --out data/models/rl_alakazam_final_avg.json \
      data/models/rl_alakazam_final_iter{1,2,3,4,5,6}.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHAMPION = "data/models/bc_alakazam_final.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=CHAMPION,
                    help="payload donating every context except MAIN")
    ap.add_argument("--include-base", action="store_true",
                    help="include the base's own MAIN in the average (shrinks toward the "
                         "champion; off by default so the average is purely of RL iterates)")
    args = ap.parse_args()

    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    paths = list(args.checkpoints)
    if args.include_base:
        paths = [args.base] + paths

    specs = []
    for p in paths:
        spec = json.loads(Path(p).read_text(encoding="utf-8"))["contexts"]["MAIN"]
        if spec.get("kind") != "mlp_ensemble":
            raise SystemExit(f"{p}: MAIN is {spec.get('kind')!r}, expected mlp_ensemble")
        specs.append(spec)

    n_members = len(specs[0]["members"])
    if any(len(s["members"]) != n_members for s in specs):
        raise SystemExit("checkpoints disagree on ensemble size")

    members = []
    for mi in range(n_members):
        avg = {}
        for key in ("W1", "b1", "w2", "b2"):
            stack = [np.asarray(s["members"][mi][key], dtype=np.float64) for s in specs]
            if any(a.shape != stack[0].shape for a in stack):
                raise SystemExit(f"member {mi} param {key}: shape mismatch across checkpoints")
            avg[key] = np.mean(stack, axis=0)
        members.append({
            "kind": "mlp", "h": int(np.shape(avg["b1"])[0]),
            "W1": avg["W1"].tolist(), "b1": avg["b1"].tolist(),
            "w2": avg["w2"].tolist(), "b2": float(avg["b2"]),
        })

    # How far the average sits from the champion, in the same units step_coherence prints.
    def flat(spec):
        return np.concatenate([np.asarray(m[k], dtype=np.float64).ravel()
                               for m in spec["members"] for k in ("W1", "b1", "w2", "b2")])

    champ_vec = flat(base["contexts"]["MAIN"])
    avg_vec = flat({"members": members})
    print(f"averaged {len(paths)} checkpoints over {n_members} ensemble members")
    for p, s in zip(paths, specs):
        print(f"  {Path(p).name:<40} |P - champion| = {np.linalg.norm(flat(s) - champ_vec):.5f}")
    print(f"  {'AVERAGE':<40} |P - champion| = {np.linalg.norm(avg_vec - champ_vec):.5f}")

    payload = dict(base)
    payload["contexts"] = dict(base["contexts"])
    payload["contexts"]["MAIN"] = {"kind": "mlp_ensemble", "members": members}
    payload["rl_meta"] = {"averaged_over": [Path(p).name for p in paths],
                          "note": "Polyak average of oscillating RL iterates (M47)"}
    Path(args.out).write_text(json.dumps(payload), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print("Now: scratchpad/m47_verify_payload.py then scratchpad/m47_arena.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
