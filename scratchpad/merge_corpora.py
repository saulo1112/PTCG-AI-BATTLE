"""M37 Fase 0 — pool Yushin's two submissions into one training corpus.

The old submission `54486275` (511 games) sat unused on disk: only a 60-game probe
dataset was ever built from it (m31_plan.md:195 planned the rest and it never ran).
Its decklist is 59/60 identical to the current one and BOTH differing cards (1266
Nighttime Mine, 1081 Enhanced Hammer) are already members of `_ALAKAZAM_DECK_IDS`,
so the rows featurize under the shipped ALAKAZAM profile with no dim change and no
overflow into the OOV bucket. That is ~+24% more MAIN decisions for free.

THE METHODOLOGICAL POINT THIS SCRIPT EXISTS TO ENFORCE
------------------------------------------------------
`split_by_game` hashes the game_id, so naively pooling would scatter old-submission
games across train/val/TEST. That would be wrong twice over:

  1. **It breaks comparability.** The gate for M37 is the GPU baseline TEST 0.7831,
     measured on the NEW corpus's held-out 20%. Change the test set and the number
     stops meaning anything.
  2. **It corrupts model selection.** The old bot is a WEAKER version of Yushin (he
     played it at lower elo). Selecting on a val set that mixes old-weak and
     new-strong behaviour optimises for the wrong target — we want to imitate the
     Yushin who is at 1191 today, not the one from two weeks ago.

So: old games are **train-only**. TEST and VAL stay exactly the new-corpus games
they were. Pooling adds training signal without touching the measuring stick.

The id ranges overlap in time (old 87533500..88669932, new 87657625..89191727 — he
ran both submissions concurrently), so they cannot be separated by an id threshold.
Hence the explicit manifest this script writes.

Run:
  uv run --group dev python scratchpad/merge_corpora.py
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

NEW = Path("data/imitation/yushinito_full.jsonl.gz")
OLD = Path("data/imitation/yushin_old_full.jsonl.gz")
OUT = Path("data/imitation/yushinito_pooled.jsonl.gz")
MANIFEST = Path("data/imitation/pooled_manifest.json")


def _stream(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def _game_ids(path: Path) -> set[str]:
    return {json.loads(line)["game_id"] for line in _stream(path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path, default=NEW)
    ap.add_argument("--old", type=Path, default=OLD)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    for p in (args.new, args.old):
        if not p.exists():
            print(f"missing {p}")
            return 2

    new_ids = _game_ids(args.new)
    old_ids = _game_ids(args.old)
    overlap = new_ids & old_ids
    if overlap:
        print(f"REFUSING: {len(overlap)} game_ids appear in BOTH corpora — "
              f"pooling would double-count them. Sample: {sorted(overlap)[:5]}")
        return 1

    n_new = n_old = 0
    with gzip.open(args.out, "wt", encoding="utf-8") as out:
        for line in _stream(args.new):
            out.write(line + "\n")
            n_new += 1
        for line in _stream(args.old):
            out.write(line + "\n")
            n_old += 1

    MANIFEST.write_text(json.dumps({
        "train_only_games": sorted(old_ids),
        "note": "old submission 54486275 — TRAIN ONLY. Never let these into val/test: "
                "it is a weaker version of the same pilot, and the M37 gate (GPU TEST "
                "0.7831) is defined on the new corpus's held-out 20%.",
        "new_games": len(new_ids),
        "old_games": len(old_ids),
    }, indent=2), encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"  new corpus : {len(new_ids):>5} games / {n_new:>7} rows  (train+val+TEST as before)")
    print(f"  old corpus : {len(old_ids):>5} games / {n_old:>7} rows  (TRAIN ONLY)")
    print(f"  pooled     : {len(new_ids) + len(old_ids):>5} games / {n_new + n_old:>7} rows")
    print(f"wrote {MANIFEST}  ({len(old_ids)} train-only game ids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
