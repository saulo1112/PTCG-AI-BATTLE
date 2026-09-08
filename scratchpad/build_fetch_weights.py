"""M34 — compose the shipping payload: champion + a re-profiled TO_HAND head.

The payload format carries ONE DeckProfile for every context, so historically
adopting ALAKAZAM_FETCH (dim 750) would have forced a retrain of MAIN's MLP
ensemble too -- the champion's single best asset (M28: 0.603 -> 0.780) and hours
of compute we do not have before the upload cutoff.

policy.py now resolves a DeckProfile PER CONTEXT (generalising what mlp_switch
already did for its specialist head), so this script writes a payload where:

  MAIN     = the champion's mlp_ensemble, copied BYTE-FOR-BYTE, still ALAKAZAM/658
  TO_HAND  = a linear vector under ALAKAZAM_FETCH/750  <- the only change
  others   = the champion's vectors, untouched

The MAIN block is hashed before and after and asserted identical: a silent MAIN
regression is the exact footgun build_mlp_alakazam_weights.py:66 has (it rebuilds
MAIN from the linear base), and it would be invisible in every offline metric.

M42 EXTENSION — two changes, both backward compatible (the --fetch-payload path
below still produces the shipped champion byte-for-byte):

  1. `--heads FILE` grafts ready-made spec DICTS emitted by train_context_heads.py
     (linear OR mlp_ensemble, each carrying its own `profile` when it needs one).
     The original path only accepted a flat linear vector and hard-coded ONE
     profile for every swapped context, which cannot express "ACTIVATE gets an MLP
     under ALAKAZAM/658 while TO_HAND gets one under ALAKAZAM_FETCH/750".
  2. The integrity guard now covers EVERY context, not just MAIN. With one context
     being swapped that was enough; with several it is not — an untouched head
     silently changing is exactly the class of bug the MAIN hash was added to catch.

Run (M34, unchanged):
  uv run --group dev python scratchpad/build_fetch_weights.py \
      --fetch-payload data/models/bc_alakazam_fetch_full.json \
      --out data/models/bc_alakazam_fetch.json

Run (M42):
  PYTHONPATH=src python scratchpad/build_fetch_weights.py \
      --base data/models/bc_alakazam_fetch.json \
      --heads data/models/ctx_activate.json \
      --out data/models/bc_alakazam_ctx.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.imitation.policy import ImitationPolicy, load_weights

BASE = Path("data/models/bc_alakazam_mlp.json")
FETCH_PROFILE = "ALAKAZAM_FETCH"


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def _assert_spec_dim(ctx: str, spec: dict, dim: int) -> None:
    """Fail loudly on a dim mismatch instead of letting policy._decide swallow it.

    ImitationPolicy._decide catches every scoring exception and falls back to greedy
    (policy.py:416), so a wrong-width head ships as a silent no-op that still reports
    bc_failures>0 only if somebody looks. Check here, where it aborts the build.
    """
    kind = spec.get("kind")
    if kind == "linear":
        got = len(spec["w"])
        if got != dim:
            raise SystemExit(f"{ctx}: linear w is dim {got}, profile says {dim}")
    elif kind == "mlp_ensemble":
        for i, m in enumerate(spec["members"]):
            got = len(m["W1"])
            if got != dim:
                raise SystemExit(f"{ctx}: mlp member {i} W1 is dim {got}, profile says {dim}")
    elif kind == "mlp":
        got = len(spec["W1"])
        if got != dim:
            raise SystemExit(f"{ctx}: mlp W1 is dim {got}, profile says {dim}")
    else:
        raise SystemExit(f"{ctx}: unsupported spec kind {kind!r} for grafting")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-payload", default="data/models/bc_alakazam_fetch_full.json",
                    help="a full linear payload trained under ALAKAZAM_FETCH; only its "
                         "TO_HAND vector is used")
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--out", default="data/models/bc_alakazam_fetch.json")
    ap.add_argument("--contexts", default="TO_HAND",
                    help="comma-separated contexts to re-profile from --fetch-payload")
    ap.add_argument("--heads", default=None, action="append",
                    help="M42: JSON of ready-made spec dicts from train_context_heads.py or "
                         "train_count_head.py. Repeatable — pass once per arm.")
    args = ap.parse_args()

    base = json.loads(Path(args.base).read_text(encoding="utf-8"))

    # Snapshot EVERY context before touching anything, so the guard below can prove
    # that exactly the intended heads changed and nothing else drifted.
    before = {c: _hash(s) for c, s in base["contexts"].items()}
    swapped: list[str] = []
    overrides: dict[str, str] = dict(base.get("profile_overrides") or {})

    # -- path 1 (M34): re-profile a flat linear vector out of a full payload ----
    if args.heads is None:
        fetch = json.loads(Path(args.fetch_payload).read_text(encoding="utf-8"))
        fetch_dim = get_profile(FETCH_PROFILE).feature_dim
        if int(fetch.get("feature_dim", -1)) != fetch_dim:
            raise SystemExit(f"{args.fetch_payload} is dim {fetch.get('feature_dim')}, "
                             f"expected {FETCH_PROFILE} dim {fetch_dim}")
        for ctx in [c.strip() for c in args.contexts.split(",") if c.strip()]:
            vec = fetch["contexts"].get(ctx)
            if vec is None:
                print(f"  !! {ctx} absent from the fetch payload (trainer dropped it) — skipped")
                continue
            if not isinstance(vec, list):
                raise SystemExit(f"{ctx} in the fetch payload is not a flat linear vector")
            base["contexts"][ctx] = {"kind": "linear", "profile": FETCH_PROFILE, "w": vec}
            base.setdefault("metrics", {}).setdefault(ctx, {})
            base["metrics"][ctx].update(fetch.get("metrics", {}).get(ctx, {}))
            base["metrics"][ctx]["profile"] = FETCH_PROFILE
            overrides[ctx] = FETCH_PROFILE
            swapped.append(ctx)

    # -- path 2 (M42): graft spec dicts, each with its own profile --------------
    else:
        count_heads: dict[str, dict] = dict(base.get("count_heads") or {})
        for hp in args.heads:
            heads = json.loads(Path(hp).read_text(encoding="utf-8"))
            metrics = heads.get("_metrics", {})
            # Two emitted shapes: flat {CTX: spec} (train_context_heads.py) and nested
            # {"contexts": {...}, "count_heads": {...}} (train_count_head.py, which must
            # ship a ranking head AND a count head together — a count head is dead code
            # unless the context also has a scorer, since _decide routes on `contexts`).
            specs = heads.get("contexts") if "contexts" in heads else {
                k: v for k, v in heads.items() if not k.startswith("_")
            }
            for ctx, spec in specs.items():
                if not isinstance(spec, dict) or "kind" not in spec:
                    raise SystemExit(f"{ctx} in {hp} is not a spec dict with a 'kind'")
                prof = spec.get("profile") or base.get("profile")
                _assert_spec_dim(ctx, spec, get_profile(prof).feature_dim)
                base["contexts"][ctx] = spec
                base.setdefault("metrics", {}).setdefault(ctx, {}).update(metrics.get(ctx, {}))
                if spec.get("profile"):
                    overrides[ctx] = spec["profile"]
                else:
                    overrides.pop(ctx, None)
                swapped.append(ctx)
            for ctx, head in (heads.get("count_heads") or {}).items():
                if ctx not in base["contexts"]:
                    raise SystemExit(f"count_head {ctx} has no ranking scorer — it would never fire")
                count_heads[ctx] = head
        if count_heads:
            base["count_heads"] = count_heads

    if not swapped:
        raise SystemExit("nothing swapped — aborting rather than writing a no-op payload")

    # -- integrity guard over EVERY context, not just MAIN ---------------------
    after = {c: _hash(s) for c, s in base["contexts"].items()}
    drifted = [c for c in before if c not in swapped and before[c] != after.get(c)]
    if drifted:
        raise SystemExit(f"untouched contexts changed — refusing to write: {drifted}")
    if "MAIN" in swapped:
        raise SystemExit("refusing to swap MAIN through this script (see M14)")
    dropped = [c for c in before if c not in after]
    if dropped:
        raise SystemExit(f"contexts vanished — refusing to write: {dropped}")

    base["profile_overrides"] = overrides
    out = Path(args.out)
    out.write_text(json.dumps(base), encoding="utf-8")

    # load through the real validator + instantiate the real policy
    payload = load_weights(out)
    ImitationPolicy(out, deck=[int(x) for x in
                               Path("decks/yushinito.csv").read_text().split()])

    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  payload profile : {payload['profile']} (dim {payload['feature_dim']})")
    print(f"  MAIN            : {payload['contexts']['MAIN'].get('kind')} — UNCHANGED "
          f"(sha {before['MAIN']})")
    for c in swapped:
        spec = payload["contexts"][c]
        prof = overrides.get(c, payload["profile"])
        cnt = " + count_head" if c in (payload.get("count_heads") or {}) else ""
        print(f"  {c:<16}: {spec.get('kind')} under {prof} "
              f"(dim {get_profile(prof).feature_dim}){cnt}{'  [NEW]' if c not in before else ''}")
    others = [c for c in payload["contexts"] if c not in swapped and c != "MAIN"]
    print(f"  untouched       : {', '.join(sorted(others))}  (all sha-verified identical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
