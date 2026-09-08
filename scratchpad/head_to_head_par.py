"""M37 — head_to_head across worker processes, plus the slices the macro WR hides.

WHY PARALLEL. The set-transformer candidate costs ~45 s per gauntlet game against the
MLP baseline's ~1 s. Over the 72-deck field at n=12 that is ~10 h serially. The field
loop is embarrassingly parallel: `run_candidate` builds a fresh BattleEnvironment,
BattleRunner and pair of policies for EVERY deck, so decks are independent units of work.

CORRECTION (2026-08-04). An earlier version of this docstring claimed the split was
"verified byte-identical to serial". **That verification was invalid** — it compared a
field slice where the baseline won 1.000 on every deck, so both runs matched trivially.
Measured properly on unsaturated win rates, **the engine is STOCHASTIC**: two identical
serial runs in the same process return different per-deck results (it shuffles decks from
entropy). Parallelism therefore cannot be, and need not be, bit-identical — each deck is
still an independent sample from the same distribution. What it does mean: any paired
comparison here needs enough games to see through that run-to-run noise, which is the
same reason M25 found n=6 signals evaporating at n=12.

Deliberately NOT done: swapping in a numpy scorer. It is ~7x faster and agrees with the
shipped stdlib kernel to 1.33e-15, but that is not zero — it could flip a near-tie and
silently change a decision. Parallelism gives a comparable speedup without touching the
arithmetic at all.

WHY THE SLICES. `_paired_delta`'s macro is an UNWEIGHTED mean over distinct decks, while
the ladder plays archetypes at their real frequency. For imitation-fetch's actual field
those disagree badly: Grimmsnarl is 30.8% of episodes but only 9 of 72 distinct decks
(12.5%). A candidate could lose the matchup that decides our ladder and still show a
healthy macro. So this reports three numbers: unweighted macro (comparable to every
previous milestone), frequency-weighted, and the Grimmsnarl subset alone.

The gate stays a VETO, never a promoter (M25: signals at n=6 evaporated at n=12; M31: a
+0.009 fidelity win lost head_to_head with a fully negative 90% CI).

Run:
  PYTHONPATH=src python -u scratchpad/head_to_head_par.py \
      decks/yushinito.csv data/models/bc_alakazam_setxf2_k3.json 12 \
      --baseline decks/yushinito.csv data/models/bc_alakazam_fetch.json \
      --label setxf2-k3 --baseline-label mlp-fetch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

REPLAY_FOLDER = Path("replays/55145833")


def _worker(args):
    """Run one candidate over a SLICE of the field. Each process loads its own SDK —
    the handle wraps a native library and is not picklable."""
    label, kind, deck_csv, weights, field_slice, n = args
    from field_gauntlet import _cards, run_candidate
    sdk, cards = _cards()
    return run_candidate(label, kind, deck_csv, weights, field_slice, n, sdk, cards)


def _run_parallel(label, kind, deck_csv, weights, field, n, workers):
    """Split the field across processes; merge the per-deck dicts."""
    chunks: list[list] = [[] for _ in range(workers)]
    for i, item in enumerate(field):           # round-robin: even cost per worker
        chunks[i % workers].append(item)
    jobs = [(label, kind, deck_csv, weights, c, n) for c in chunks if c]
    merged: dict[str, float] = {}
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        for part in pool.map(_worker, jobs):
            merged.update(part)
    print(f"  [{label}] {len(merged)} decks in {time.perf_counter() - started:.0f}s "
          f"across {len(jobs)} workers  macro={sum(merged.values()) / len(merged):.3f}")
    return merged


def _episode_counts(field, cards) -> dict[str, int]:
    """How often each field deck was ACTUALLY faced, for the frequency-weighted view.

    Reads the same replays the field was extracted from and matches on the deck's card
    multiset, so a deck seen 20 times weighs 20x a deck seen once — which is what the
    ladder does and what the unweighted macro does not.
    """
    from field_gauntlet import OUR_NAME, _first_deck
    by_key = {tuple(sorted(dids)): name for name, dids in field}
    counts: Counter = Counter()
    for fp in sorted(REPLAY_FOLDER.glob("*.json")):
        if "metadata" in fp.name:
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        agents = data.get("info", {}).get("Agents", [])
        ours = {i for i, a in enumerate(agents)
                if a.get("Name", "").lower() == OUR_NAME.lower()}
        if len(ours) == len(agents):
            continue                                   # self-play validation episode
        for seat in range(len(agents)):
            if seat in ours:
                continue
            deck = _first_deck(data, seat)
            if not deck:
                continue
            name = by_key.get(tuple(sorted(deck)))
            if name:
                counts[name] += 1
    return dict(counts)


def _grimmsnarl_decks(field) -> set[str]:
    from ptcg_ai.imitation import deck_profiles as DP
    g = set(DP.GRIMMSNARL.deck_ids)
    return {name for name, dids in field if len(set(dids) & g) / len(g) > 0.5}


def _report(tag, a, b, la, lb, keys=None):
    from field_gauntlet import _paired_delta
    if keys is not None:
        a = {k: v for k, v in a.items() if k in keys}
        b = {k: v for k, v in b.items() if k in keys}
    if not a:
        print(f"  {tag:<22} (no decks in this slice)")
        return
    mean, lo, hi, wins, losses, n = _paired_delta(a, b)
    verdict = "CANDIDATE WORSE" if hi < 0 else ("candidate better" if lo > 0 else "tie")
    print(f"  {tag:<22} {la} {sum(a.values())/len(a):.3f} vs {lb} "
          f"{sum(b.values())/len(b):.3f} | delta {mean:+.3f} "
          f"[90% {lo:+.3f}, {hi:+.3f}] {wins}W/{losses}L over {n} decks -> {verdict}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("deck_csv")
    ap.add_argument("weights")
    ap.add_argument("n", nargs="?", type=int, default=12)
    ap.add_argument("--baseline", nargs=2, metavar=("DECK_CSV", "WEIGHTS"), required=True)
    ap.add_argument("--label", default="candidate")
    ap.add_argument("--baseline-label", dest="baseline_label", default="baseline")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                    help="default = PHYSICAL cores; this box is 4c/8t and the work is "
                         "CPU-bound pure Python, so SMT threads buy little")
    ap.add_argument("--limit", type=int, default=None, help="first N field decks (smoke)")
    args = ap.parse_args(argv)

    from field_gauntlet import _cards, extract_field
    sdk, cards = _cards()
    field = extract_field(cards)
    if args.limit:
        field = field[: args.limit]
    counts = _episode_counts(field, cards)
    grim = _grimmsnarl_decks(field)
    print(f"field: {len(field)} decks | {len(grim)} Grimmsnarl-like | "
          f"{sum(counts.values())} episodes matched | n={args.n} | workers={args.workers}")

    cand = _run_parallel(args.label, "imitation", args.deck_csv, args.weights,
                         field, args.n, args.workers)
    base = _run_parallel(args.baseline_label, "imitation", args.baseline[0],
                         args.baseline[1], field, args.n, args.workers)

    print("\nPAIRED DELTAS (veto, not promoter):")
    _report("macro (unweighted)", cand, base, args.label, args.baseline_label)
    _report("Grimmsnarl only", cand, base, args.label, args.baseline_label, keys=grim)
    _report("non-Grimmsnarl", cand, base, args.label, args.baseline_label,
            keys=set(cand) - grim)

    if counts:
        tot = sum(counts.get(k, 0) for k in cand)
        if tot:
            wc = sum(cand[k] * counts.get(k, 0) for k in cand) / tot
            wb = sum(base[k] * counts.get(k, 0) for k in base) / tot
            print(f"  {'frequency-weighted':<22} {args.label} {wc:.3f} vs "
                  f"{args.baseline_label} {wb:.3f} | delta {wc - wb:+.3f}  "
                  f"(no CI: weights are fixed, not resampled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
