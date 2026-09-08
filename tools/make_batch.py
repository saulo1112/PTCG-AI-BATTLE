"""Turn downloaded replay folders into a ready-to-run quick_screen --batch file.

Closes the last manual step of the candidate pipeline: the episode API exposes
submissionId but NOT the player name, while quick_screen needs the name (it filters
that player's decisions out of the replays). The name IS in the downloaded replay
JSONs (`info.Agents[].Name`) — the submission's owner is the agent that appears in
(nearly) every episode of its own folder, while opponents each appear ~once.

Usage:
  python tools/make_batch.py replays/54611538 replays/54708568 ... > candidates.txt
  python tools/make_batch.py --all replays/            # every subfolder
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


def owner_name(folder: Path, sample: int = 40) -> tuple[str | None, int, int]:
    """Most frequent agent name in the folder = the submission's own player.
    Returns (name, appearances, files_scanned)."""
    names: collections.Counter = collections.Counter()
    files = [f for f in sorted(folder.glob("*.json")) if f.name != "metadata.json"]
    scanned = 0
    for f in files[:sample]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        scanned += 1
        for a in data.get("info", {}).get("Agents", []):
            n = a.get("Name")
            if n:
                names[n] += 1
    if not names:
        return None, 0, scanned
    name, count = names.most_common(1)[0]
    return name, count, scanned


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folders", nargs="*", help="replay folders (replays/<submission_id>)")
    ap.add_argument("--all", metavar="ROOT", default=None,
                    help="use every subfolder of ROOT instead of listing folders")
    ap.add_argument("--min-share", type=float, default=0.8,
                    help="owner must appear in at least this share of scanned files")
    args = ap.parse_args()

    folders = ([p for p in sorted(Path(args.all).iterdir()) if p.is_dir()]
               if args.all else [Path(f) for f in args.folders])
    if not folders:
        ap.print_help()
        return 2

    for folder in folders:
        name, count, scanned = owner_name(folder)
        if name is None or scanned == 0:
            print(f"# {folder}: no replays parsed — skipped", file=sys.stderr)
            continue
        share = count / scanned
        if share < args.min_share:
            print(f"# {folder}: ambiguous owner ({name!r} in {share:.0%} of {scanned}) — "
                  f"check manually", file=sys.stderr)
            continue
        print(f"{folder}\t{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
