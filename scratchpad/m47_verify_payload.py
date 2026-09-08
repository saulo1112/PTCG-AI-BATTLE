"""M47 — prove an RL checkpoint changed MAIN and NOTHING else, before it is bundled.

`rl_selfplay._write_ckpt` shallow-copies the base payload and swaps `contexts["MAIN"]`.
That SHOULD leave the other 11 contexts, `profile_overrides` and `count_heads` untouched
— but "should" is exactly what cost this project submission 55203764 (M37 shipped the
SDK's sample deck next to Alakazam weights; `validate_submission` passed, the entrypoint
smoke test passed with `bc_failures: 0`, and the agent was blind). The rule since then is
that the artefact gets verified, not the intention.

This is the same sha guard `build_fetch_weights.py` and `graft_main_k7.py` apply, pointed
at an RL checkpoint: every context except MAIN must hash identically to the champion, MAIN
must NOT, and the auxiliary blocks must survive.

Run:  PYTHONPATH=src python scratchpad/m47_verify_payload.py data/models/rl_alakazam_final_iter12.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

CHAMPION = "data/models/bc_alakazam_final.json"


def _sha(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def main() -> int:
    cand_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not cand_path:
        raise SystemExit(f"usage: {sys.argv[0]} <rl_checkpoint.json>")
    champ = json.loads(Path(CHAMPION).read_text(encoding="utf-8"))
    cand = json.loads(Path(cand_path).read_text(encoding="utf-8"))

    problems = []
    print(f"champion : {CHAMPION}")
    print(f"candidate: {cand_path}\n")

    champ_ctx, cand_ctx = champ["contexts"], cand["contexts"]
    if set(champ_ctx) != set(cand_ctx):
        problems.append(f"context set differs: champion-only={set(champ_ctx) - set(cand_ctx)}, "
                        f"candidate-only={set(cand_ctx) - set(champ_ctx)}")

    print(f"{'context':<26}{'champion':>18}{'candidate':>18}   verdict")
    print("-" * 76)
    for ctx in sorted(set(champ_ctx) | set(cand_ctx)):
        a, b = _sha(champ_ctx.get(ctx)), _sha(cand_ctx.get(ctx))
        if ctx == "MAIN":
            ok = a != b
            verdict = "CHANGED (expected)" if ok else "IDENTICAL -- RL changed nothing!"
        else:
            ok = a == b
            verdict = "identical" if ok else "DIFFERS -- must not happen"
        if not ok:
            problems.append(f"{ctx}: {verdict}")
        print(f"{ctx:<26}{a:>18}{b:>18}   {verdict}")

    print()
    for key in ("profile", "feature_dim", "profile_overrides", "count_heads"):
        a, b = champ.get(key), cand.get(key)
        same = _sha(a) == _sha(b)
        print(f"{key:<26}{'identical' if same else 'DIFFERS'}   {json.dumps(b)[:60]}")
        if not same:
            problems.append(f"{key} differs between champion and candidate")

    meta = cand.get("rl_meta")
    print(f"\nrl_meta: {json.dumps(meta)}")
    if not meta:
        problems.append("no rl_meta on the candidate -- is this really an RL checkpoint?")
    elif "reflected_from" in meta:
        # M48: a reflection through the champion of a direction measured confidently
        # worse. Carries provenance, not training hyperparameters.
        print(f"  (reflected payload, alpha={meta.get('alpha')}, "
              f"from {meta.get('reflected_from')})")
        if not meta.get("reflected_from"):
            problems.append("reflected payload with no source recorded")
    elif "averaged_over" in meta:
        # A Polyak average (m47_average_iterates.py) carries the list it averaged, not
        # training hyperparameters. Check the provenance instead: every source must be a
        # clipped checkpoint, which is verified where those were written, so here we only
        # assert the record is non-empty and plural.
        srcs = meta.get("averaged_over") or []
        print(f"  (averaged payload over {len(srcs)} checkpoints)")
        if len(srcs) < 2:
            problems.append(f"averaged_over lists {len(srcs)} checkpoint(s) -- not an average")
    elif meta.get("clip") is None:
        problems.append(f"rl_meta says clip={meta.get('clip')} -- this checkpoint was trained "
                        "WITHOUT the PPO clip, which is the whole point of M47")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK: MAIN changed, every other context and auxiliary block is byte-identical "
          "to the champion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
