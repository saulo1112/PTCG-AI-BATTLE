"""Pick the next N unscreened candidates by score — with reliable dedup.

`data/candidates_joined.json`'s own `screened` flag is unreliable (it was never
updated across M21/M22/M23). The authoritative record of "already screened" is on
disk: `quick_screen.py` writes `decks/{slug}.csv` for EVERY candidate it touches
(even DISCARD_DECK ones, before the strength gate), so a candidate whose slug has a
deck csv has been screened. DISCARD_DECK candidates never get a `_screen.json`, so
the deck csv — not the model json — is the complete marker. Prior-round batch files
(`candidates_m22*.txt`) record the sids too, as a backstop.

Modes:
  python tools/next_batch.py                 # list next 20 unscreened by score
  python tools/next_batch.py --n 20
  python tools/next_batch.py --backfill       # persist the disk-reconciled flag
  python tools/next_batch.py --mark 54611538 54767753 ...   # flag sids done

Always run with PYTHONIOENCODING=utf-8 (team names carry emoji/CJK).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

CANDIDATES = Path("data/candidates_joined.json")


def _slug(name: str) -> str:
    """Identical to quick_screen._slug so deck-csv stems line up with team names."""
    return "".join(c for c in name.lower() if c.isalnum()) or "candidate"


def _screened_on_disk() -> tuple[set[str], set[int]]:
    """(deck-csv slugs, sids referenced in candidates_m22*.txt)."""
    slugs = {Path(p).stem for p in glob.glob("decks/*.csv")}
    sids: set[int] = set()
    for fn in glob.glob("candidates_m22*.txt"):
        for line in Path(fn).read_text(encoding="utf-8").splitlines():
            m = re.search(r"replays[\\/](\d+)", line)
            if m:
                sids.add(int(m.group(1)))
    return slugs, sids


def _is_screened(cand: dict, disk_slugs: set[str], disk_sids: set[int]) -> bool:
    return bool(
        cand.get("screened")
        or _slug(cand["team"]) in disk_slugs
        or cand["sid"] in disk_sids
    )


def _load() -> list[dict]:
    return json.loads(CANDIDATES.read_text(encoding="utf-8"))


def _save(data: list[dict]) -> None:
    CANDIDATES.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--backfill", action="store_true",
                    help="persist screened=True for everything reconciled from disk")
    ap.add_argument("--mark", type=int, nargs="+", default=None,
                    help="set screened=True for these sids and save")
    args = ap.parse_args()

    data = _load()

    if args.mark:
        by_sid = {c["sid"]: c for c in data}
        hit = 0
        for sid in args.mark:
            if sid in by_sid:
                by_sid[sid]["screened"] = True
                hit += 1
        _save(data)
        print(f"marked {hit}/{len(args.mark)} sids screened=True; saved {CANDIDATES}")
        return 0

    disk_slugs, disk_sids = _screened_on_disk()
    for c in data:
        c["_screened_eff"] = _is_screened(c, disk_slugs, disk_sids)

    if args.backfill:
        n0 = sum(1 for c in data if c.get("screened"))
        for c in data:
            if c.pop("_screened_eff"):
                c["screened"] = True
        _save(data)
        n1 = sum(1 for c in data if c.get("screened"))
        print(f"backfill: screened flag {n0} -> {n1} of {len(data)} (saved {CANDIDATES})")
        return 0

    n_done = sum(1 for c in data if c["_screened_eff"])
    todo = sorted((c for c in data if not c["_screened_eff"]),
                  key=lambda x: -x["score"])[: args.n]
    print(f"pool={len(data)}  screened(disk-reconciled)={n_done}  "
          f"remaining={len(data)-n_done}  showing next {len(todo)} by score\n")
    print(f"{'rank':>5} {'score':>7} {'seen':>7} {'sid':>10}  team")
    for c in todo:
        print(f"{c['rank']:>5} {c['score']:>7.1f} {c['seen']:>7.1f} {c['sid']:>10}  {c['team']}")
    print("\n# download line:")
    print("python tools/download_competitors.py --submissions "
          + " ".join(str(c["sid"]) for c in todo)
          + " --limit 40 --cookies tools/cookies.txt.json --out replays/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
