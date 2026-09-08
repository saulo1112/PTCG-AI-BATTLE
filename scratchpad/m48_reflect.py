"""M48 — reflect a MEASURED-BAD weight direction through the champion.

THE REASONING, and why this is not fishing. The GAE(lambda=0) chain produced a policy that
the n=600 mirror scored at 0.448 with a 95% CI of [0.409, 0.488] against a control of
0.487 -- entirely below 0.500, i.e. confidently WORSE. That makes the displacement

    d = W_candidate - W_champion

a direction in weight space whose effect on playing strength has been MEASURED, with
significance, rather than hypothesised. To first order the objective along a small step is
linear, so moving -d from the same origin should carry the opposite sign. The step is
small enough for that to be a reasonable approximation: |d| = 0.470 against |W| = 45.49,
i.e. **1.0%**, in a 95,043-dimensional space.

WHAT WOULD MAKE THIS FISHING, and how it is avoided: generating a hypothesis from a
measurement and then confirming it on the SAME measurement. So the gate is pre-registered
before running -- mirror CI entirely above 0.500 with the control in 0.48-0.52 -- and a
pass must then REPLICATE on an independent arena run before anything is built or shipped.
The engine is stochastic (M38), so a second run is genuinely independent evidence.

WHAT IT WOULD MEAN IF IT WORKS: the self-play objective points AWAY from ladder strength,
and its negation is informative. Which is a strange thing to ship, and would need the
held-out matchup rows before anyone believed it.

Run:
  PYTHONPATH=src python scratchpad/m48_reflect.py \
      --candidate data/models/rl_gae_lam0.0_iter6.json \
      --alpha 1.0 --out data/models/rl_gae_reflected.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHAMPION = "data/models/bc_alakazam_final.json"


def _members(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))["contexts"]["MAIN"]
    if spec.get("kind") != "mlp_ensemble":
        raise SystemExit(f"{path}: MAIN is {spec.get('kind')!r}, expected mlp_ensemble")
    return spec["members"]


def _flat(members):
    return np.concatenate([np.asarray(m[k], dtype=np.float64).ravel()
                           for m in members for k in ("W1", "b1", "w2", "b2")])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="W_out = W_champ - alpha*(W_cand - W_champ). 1.0 is the exact "
                         "reflection; <1 is a shorter step in the same reversed direction.")
    ap.add_argument("--base", default=CHAMPION)
    args = ap.parse_args()

    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    champ_m, cand_m = _members(args.base), _members(args.candidate)
    if len(champ_m) != len(cand_m):
        raise SystemExit("ensemble sizes differ")

    out_members = []
    for mi, (c, k) in enumerate(zip(champ_m, cand_m)):
        m = {"kind": "mlp", "h": int(np.shape(np.asarray(c["b1"]))[0])}
        for key in ("W1", "b1", "w2", "b2"):
            cw = np.asarray(c[key], dtype=np.float64)
            kw = np.asarray(k[key], dtype=np.float64)
            if cw.shape != kw.shape:
                raise SystemExit(f"member {mi} param {key}: shape mismatch")
            val = cw - args.alpha * (kw - cw)
            m[key] = float(val) if key == "b2" else val.tolist()
        out_members.append(m)

    cf, kf, of = _flat(champ_m), _flat(cand_m), _flat(out_members)
    print(f"|champion|                 {np.linalg.norm(cf):.4f}")
    print(f"|candidate - champion|     {np.linalg.norm(kf - cf):.4f}  "
          f"({100 * np.linalg.norm(kf - cf) / np.linalg.norm(cf):.2f}% of |champion|)")
    print(f"|reflected - champion|     {np.linalg.norm(of - cf):.4f}")
    cos = float((of - cf) @ (kf - cf) /
                (np.linalg.norm(of - cf) * np.linalg.norm(kf - cf) + 1e-12))
    print(f"cos(reflected, candidate)  {cos:+.4f}   (must be -1.000 at alpha=1)")

    payload = dict(base)
    payload["contexts"] = dict(base["contexts"])
    payload["contexts"]["MAIN"] = {"kind": "mlp_ensemble", "members": out_members}
    payload["rl_meta"] = {"reflected_from": Path(args.candidate).name, "alpha": args.alpha,
                          "note": "M48: negation of a direction measured confidently WORSE "
                                  "(mirror 0.448 [0.409,0.488] vs control 0.487)"}
    Path(args.out).write_text(json.dumps(payload), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
