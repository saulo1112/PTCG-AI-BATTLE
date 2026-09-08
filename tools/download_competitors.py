"""Download replay folders for several submission_ids in one command.

Usage:
    python tools/download_competitors.py --submissions 53922595 54347432 --cookies tools/cookies.txt.json --out replays/
"""

from __future__ import annotations

import argparse

from download_replays import download_submission_replays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submissions", type=int, nargs="+", required=True, help="Kaggle submission_ids")
    parser.add_argument("--cookies", default="tools/cookies.txt.json", help="Path to exported cookie JSON")
    parser.add_argument("--out", default="replays", help="Output directory root")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include in-progress/unsuccessful episodes too (default: completed only)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="max NEW replays per submission this run (screening needs ~100-150; "
             "incremental, so re-run without it to top up a candidate that passes)",
    )
    args = parser.parse_args()

    for submission_id in args.submissions:
        download_submission_replays(
            submission_id=submission_id,
            cookies_path=args.cookies,
            output_dir=args.out,
            successful_only=not args.all,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
