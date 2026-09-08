"""M37 Fase 3 — convert the Colab `setxf2` state_dict dump into a shippable payload.

LOCAL ONLY. Never bundled; it exists to produce `data/models/bc_alakazam_setxf2.json`.

The trainer writes `--emit-raw` as a straight `state_dict` -> nested-list dump: torch key
names, `[out][in]` nested lists, and floats at full float64 repr (19.6 chars each). That
is 100 MB for 7 members and it is not the format `setnet.py` reads. Three changes here:

1. **Rename** torch keys to setnet's flat names (`opt_proj.weight` -> `opt_w`, ...).
2. **Flatten** row-major. setnet slices rows out of a flat list; nested lists would need
   a different (slower) indexing path.
3. **Quantise to 7 significant digits.** The weights were TRAINED in float32, so digits
   past the 7th are noise from the float64 JSON repr, not signal. Uses `f"{v:.7g}"`, NOT
   `round(v, 6)` — the latter is ABSOLUTE rounding and would flush small-magnitude
   weights (1e-8) to zero. Measured: 19.6 -> 8.4 chars/float, so ~100 MB -> ~45 MB at
   k=7, and `json.load` drops from 2.2 s to ~1 s at agent import.

The result is grafted as `contexts["MAIN"]` onto a COPY of the base payload, leaving
TO_HAND (ALAKAZAM_FETCH, dim 750) and every other context untouched.

Usage:
    python scratchpad/convert_setxf2.py --k 3
    python scratchpad/convert_setxf2.py --k 7 --out data/models/bc_alakazam_setxf2_k7.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAW = Path("data/models/setxf2_k7.json")
BASE = Path("data/models/bc_alakazam_fetch.json")

#: setnet's expected slice bounds. These are class constants on SetTransformerV2 and are
#: NOT stored in the raw dump, so they are re-declared here and cross-checked against the
#: actual tensor shapes below (opt_proj is [d, OPT_END], state_enc.0 is [d, STATE span]).
OPT_END, STATE_START, STATE_END = 69, 69, 103


def q(v: float) -> float:
    """Quantise to float32's ~7 significant digits. Relative, not absolute."""
    return float(f"{v:.7g}")


def flat(t: list) -> list[float]:
    """Row-major flatten of a 1-D or 2-D nested list, quantised."""
    if t and isinstance(t[0], list):
        return [q(v) for row in t for v in row]
    return [q(v) for v in t]


def convert_member(sd: dict) -> dict:
    """One torch state_dict -> one setnet member dict."""
    out: dict = {
        "opt_w": flat(sd["opt_proj.weight"]),
        "opt_b": flat(sd["opt_proj.bias"]),
        "state0_w": flat(sd["state_enc.0.weight"]),
        "state0_b": flat(sd["state_enc.0.bias"]),
        "state2_w": flat(sd["state_enc.2.weight"]),
        "state2_b": flat(sd["state_enc.2.bias"]),
        "film_w": flat(sd["film.weight"]),
        "film_b": flat(sd["film.bias"]),
        "lnout_w": flat(sd["ln_out.weight"]),
        "lnout_b": flat(sd["ln_out.bias"]),
        # nn.Linear(d, 1): weight is [1, d] -> flattens to d; bias is a 1-list -> scalar
        "head_w": flat(sd["head.weight"]),
        "head_b": q(sd["head.bias"][0]),
        "blocks": [],
    }
    n_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    for i in range(n_layers):
        p = f"blocks.{i}."
        out["blocks"].append({
            "ln1_w": flat(sd[p + "ln1.weight"]),
            "ln1_b": flat(sd[p + "ln1.bias"]),
            "ln2_w": flat(sd[p + "ln2.weight"]),
            "ln2_b": flat(sd[p + "ln2.bias"]),
            "in_proj_weight": flat(sd[p + "attn.in_proj_weight"]),
            "in_proj_bias": flat(sd[p + "attn.in_proj_bias"]),
            "out_proj_weight": flat(sd[p + "attn.out_proj.weight"]),
            "out_proj_bias": flat(sd[p + "attn.out_proj.bias"]),
            "ff0_w": flat(sd[p + "ff.0.weight"]),
            "ff0_b": flat(sd[p + "ff.0.bias"]),
            "ff2_w": flat(sd[p + "ff.2.weight"]),
            "ff2_b": flat(sd[p + "ff.2.bias"]),
        })
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="how many ensemble members to emit")
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    if raw.get("kind") != "setxf2_ensemble":
        raise SystemExit(f"{args.raw} is kind={raw.get('kind')!r}, expected setxf2_ensemble")
    cfg = raw["config"]
    members = raw["members"]
    if not (1 <= args.k <= len(members)):
        raise SystemExit(f"--k {args.k} outside 1..{len(members)}")

    # cross-check the hardcoded slice bounds against the real tensor shapes, so a
    # retrain with different bounds fails here instead of silently mis-slicing at inference
    d = int(cfg["d_model"])
    if len(members[0]["opt_proj.weight"][0]) != OPT_END:
        raise SystemExit(
            f"opt_proj expects {len(members[0]['opt_proj.weight'][0])} inputs, "
            f"OPT_END={OPT_END} — the trained slice bounds changed")
    if len(members[0]["state_enc.0.weight"][0]) != STATE_END - STATE_START:
        raise SystemExit("state_enc input width != STATE_END-STATE_START")

    spec = {
        "kind": "setxf2_ensemble",
        "d_model": d,
        "layers": int(cfg["layers"]),
        "heads": int(cfg["heads"]),
        "opt_end": OPT_END,
        "state_start": STATE_START,
        "state_end": STATE_END,
        "members": [convert_member(m) for m in members[: args.k]],
    }

    payload = json.loads(args.base.read_text(encoding="utf-8"))
    if int(payload.get("feature_dim", -1)) != int(raw["dim"]):
        raise SystemExit(
            f"base payload feature_dim={payload.get('feature_dim')} != trained dim {raw['dim']}")
    # the MLP MAIN scorer we are replacing becomes the time-guard's cheap fallback,
    # travelling INSIDE the spec (a phantom top-level context would never be looked up)
    prev_main = payload["contexts"].get("MAIN")
    if prev_main is not None:
        spec["fallback"] = prev_main
    payload["contexts"]["MAIN"] = spec

    out = args.out or Path(f"data/models/bc_alakazam_setxf2_k{args.k}.json")
    out.write_text(json.dumps(payload), encoding="utf-8")
    mb = out.stat().st_size / (1024 * 1024)
    n_par = sum(
        len(v) for m in spec["members"] for k2, v in m.items() if isinstance(v, list)
    ) + sum(
        len(v) for m in spec["members"] for b in m["blocks"] for v in b.values()
    )
    print(f"wrote {out} ({mb:.1f} MB)  k={args.k}  ~{n_par:,} weights")
    print(f"  contexts: {sorted(payload['contexts'])}")
    print(f"  fallback carried: {'yes' if 'fallback' in spec else 'NO — base had no MAIN'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
