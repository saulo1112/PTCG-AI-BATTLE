"""M47 — audit the COUNT dimension across every context, not just the one M42 fixed.

WHY THIS, AND WHY NOW. The only intervention that has ever cleared this project's decisive
arena is M42/M43's: give a model to contexts that were being decided BLINDLY or by a rule
that cannot express the teacher's behaviour. The single largest lift ever measured
(+0.5098, SETUP_BENCH_POKEMON) came from the COUNT dimension specifically -- M42 found that
`ImitationPolicy._top_k` takes `min(maxCount, n)` floored at `minCount`, so **no ranking
model at any capacity or depth can express "take fewer"**. That was a structural
impossibility, not a fidelity gap, and fixing it was worth more than every capacity
experiment combined.

M42 then applied `count_heads` to exactly ONE context and stopped. Nobody has ever checked
whether the same impossibility exists elsewhere. This checks all of them, over Yushin's
full corpus, with no model and no training -- it is pure arithmetic on the data:

    hi = min(maxCount, n)      what the shipped policy ALWAYS takes
    k  = len(chosen)           what the teacher actually took
    mismatch = k != hi         a decision our agent cannot reproduce at any capacity

A context with many decisions per game AND a high mismatch rate is a structural blind spot
of the same kind as SETUP_BENCH. A context where k == hi almost always has no count
dimension to learn, and the question is closed for it forever.

Uses `read_decision_dataset`, the vetted extractor: a raw sweep that reads `cell['action']`
from the SAME step reports the OPPOSITE result (M42's method trap, docs/handoff.md).

Run:  PYTHONPATH=src python -u scratchpad/m47_count_audit.py
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

from ptcg_ai.imitation import features as F
from ptcg_ai.imitation.dataset import read_decision_dataset
from ptcg_ai.observation.parser import ObservationParser

DATASET = Path("data/imitation/yushinito_full.jsonl.gz")
#: contexts that already carry a count head in the shipped payload
ALREADY_FIXED = {"SETUP_BENCH_POKEMON"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET)
    args = ap.parse_args()

    parser = ObservationParser()
    stats: dict[str, dict] = collections.defaultdict(
        lambda: {"n": 0, "mismatch": 0, "fewer": 0, "more": 0,
                 "multi": 0, "games": set(), "k_dist": collections.Counter(),
                 "hi_dist": collections.Counter()})

    n_rows = 0
    for r in read_decision_dataset(args.dataset):
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None or F.is_prize_pick(obs.select):
            continue
        n = len(obs.select.option)
        if n == 0:
            continue
        chosen = [a for a in r.action if 0 <= a < n]
        if len(set(chosen)) != len(chosen):
            continue
        n_rows += 1
        ctx = obs.select.context.name
        s = stats[ctx]
        lo = min(max(obs.select.minCount, 0), n)
        hi = min(obs.select.maxCount, n)
        k = len(chosen)
        s["n"] += 1
        s["games"].add(r.game_id)
        s["k_dist"][k] += 1
        s["hi_dist"][hi] += 1
        if hi > lo:
            s["multi"] += 1          # the count is genuinely free here
        if k != hi:
            s["mismatch"] += 1
            if k < hi:
                s["fewer"] += 1
            else:
                s["more"] += 1       # should be impossible; a canary for a parsing bug

    n_games = len(set().union(*(s["games"] for s in stats.values()))) if stats else 0
    print(f"{n_rows} decisions over {n_games} games\n")
    print(f"{'context':<24}{'/game':>7}{'n':>8}{'free-k%':>9}{'MISMATCH%':>11}"
          f"{'fewer':>8}{'more':>7}  status")
    print("-" * 88)

    rows = sorted(stats.items(), key=lambda kv: -kv[1]["mismatch"])
    for ctx, s in rows:
        per_game = s["n"] / max(n_games, 1)
        free = s["multi"] / max(s["n"], 1)
        mm = s["mismatch"] / max(s["n"], 1)
        if ctx in ALREADY_FIXED:
            status = "count head SHIPPED (M42)"
        elif s["mismatch"] == 0:
            status = "no count dimension -- closed"
        elif mm * per_game >= 0.20:
            status = ">>> STRUCTURAL BLIND SPOT <<<"
        else:
            status = "present but low volume"
        print(f"{ctx:<24}{per_game:>7.2f}{s['n']:>8}{free:>9.1%}{mm:>11.1%}"
              f"{s['fewer']:>8}{s['more']:>7}  {status}")

    print("-" * 88)
    print("MISMATCH% = decisions where the teacher's count differs from min(maxCount, n),")
    print("which is what `ImitationPolicy._top_k` always takes. Those decisions cannot be")
    print("reproduced by ANY ranking model -- that is what made SETUP_BENCH worth +0.5098.")
    print("The shortlist ranks by mismatches PER GAME (rate x frequency), because a 40%")
    print("mismatch on 0.05 decisions/game is worth nothing and 6% on 12/game is not.")
    print()
    for ctx, s in rows:
        if ctx in ALREADY_FIXED or s["mismatch"] == 0:
            continue
        per_game = s["n"] / max(n_games, 1)
        mm = s["mismatch"] / max(s["n"], 1)
        print(f"  {ctx:<24} {mm * per_game:>6.3f} corrigible decisions/game   "
              f"k dist {dict(s['k_dist'].most_common(4))}  hi dist {dict(s['hi_dist'].most_common(4))}")
    print("\n`more` must be 0: taking MORE than min(maxCount, n) is illegal. Any nonzero")
    print("value means the extractor or the parse is wrong, and nothing above is trustworthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
