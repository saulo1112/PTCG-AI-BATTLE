"""Discover clone-teacher candidates from our OWN episode history — no leaderboard needed.

The bottleneck for the imitation lever (m15_findings.md: "would need a genuinely NEW,
simple-deck ~700-900 teacher not yet sourced") was never the screen speed, it was
FINDING candidates. This automates that: every episode we played records both agents'
`submissionId` and their Kaggle ladder rating (`updatedScore`), so our own match history
is already a ranked directory of opponents — with exactly the submission_ids that
tools/download_competitors.py consumes.

Pipeline:
  1. python tools/find_candidates.py --ours 54555926 54556007 --min-score 700 --max-score 900
  2. python tools/download_competitors.py --submissions <ids from step 1> --out replays/
  3. uv run --group dev python scratchpad/quick_screen.py --batch <batch file from step 1>

Note the Kaggle leaderboard REST/internal endpoints need a different auth (API token /
different payload) than our cookie session; mining episodes needs no extra auth because
tools/kaggle_api.py already talks to the EpisodeService successfully.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

from kaggle_api import list_episodes, load_session


def mine(session, our_subs: list[int]) -> dict[int, dict]:
    """Aggregate every opponent seen across our submissions' episodes."""
    our_teams: set[int] = set()
    for sub in our_subs:
        for e in list_episodes(session, sub):
            for a in e.get("agents", []):
                if a.get("submissionId") == sub:
                    our_teams.add(a.get("teamId"))

    opp: dict[int, dict] = collections.defaultdict(
        lambda: {"games": 0, "best": 0.0, "last": 0.0, "team": None, "their_wins": 0})
    for sub in our_subs:
        for e in list_episodes(session, sub):
            for a in e.get("agents", []):
                sid = a.get("submissionId")
                if sid is None or sid in our_subs or a.get("teamId") in our_teams:
                    continue
                score = a.get("updatedScore") or a.get("initialScore") or 0.0
                d = opp[sid]
                d["games"] += 1
                d["team"] = a.get("teamId")
                d["best"] = max(d["best"], score)
                d["last"] = score
                if (a.get("reward") or 0) > 0:
                    d["their_wins"] += 1
    return dict(opp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", type=int, nargs="+", required=True,
                    help="our own submission_ids to mine (e.g. 54555926 54556007)")
    ap.add_argument("--cookies", default="tools/cookies.txt.json")
    ap.add_argument("--min-score", type=float, default=700.0)
    ap.add_argument("--max-score", type=float, default=900.0,
                    help="upper bound: M6 showed top-tier decks are too complex for our "
                         "heuristics to pilot, so absurdly-rated agents are poor clone targets")
    ap.add_argument("--top", type=int, default=10, help="how many to emit")
    ap.add_argument("--out", default=None, help="write a quick_screen --batch file here")
    args = ap.parse_args()

    session = load_session(args.cookies)
    opp = mine(session, args.ours)
    band = {s: d for s, d in opp.items() if args.min_score <= d["best"] <= args.max_score}
    ranked = sorted(band.items(), key=lambda kv: -kv[1]["best"])[: args.top]

    print(f"opponents seen: {len(opp)}   in band [{args.min_score:.0f},{args.max_score:.0f}]: {len(band)}")
    print(f"\n{'submissionId':>13}{'bestScore':>11}{'games':>7}{'theirWins':>10}")
    for sid, d in ranked:
        print(f"{sid:>13}{d['best']:>11.1f}{d['games']:>7}{d['their_wins']:>10}")

    ids = " ".join(str(s) for s, _ in ranked)
    print(f"\nNext:\n  python tools/download_competitors.py --submissions {ids} "
          f"--cookies {args.cookies} --out replays/")
    if args.out:
        # player NAME is not in the episode API; fill it after download (the replay JSONs
        # carry info.Agents[].Name). Emitted as a template to complete.
        lines = [f"replays/{sid}\t<PLAYER NAME — read from replays/{sid}/*.json info.Agents>"
                 for sid, _ in ranked]
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  wrote batch template -> {args.out} (fill in player names after download)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
