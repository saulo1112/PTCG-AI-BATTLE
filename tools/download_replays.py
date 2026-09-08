"""Download all replay JSONs for a single Kaggle submission_id, incrementally.

Usage:
    python tools/download_replays.py --submission 53922595 --cookies tools/cookies.txt.json --out replays/
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from kaggle_api import fetch_replay, list_episodes, load_session

MAX_RETRIES = 5
REQUEST_DELAY_SECONDS = 0.6


def download_submission_replays(
    submission_id: int,
    cookies_path: str,
    output_dir: str,
    successful_only: bool = True,
    limit: int | None = None,
) -> None:
    """Download a submission's replays, skipping any already on disk.

    ``limit`` caps how many NEW files this run fetches — the screening pipeline only
    needs ~100-150 games (decklist extraction stabilises in ~30; the clonability check
    wants a few thousand decisions), while top-ranked submissions can have 600-900
    episodes. Because the download is incremental, a candidate that passes the screen
    can simply be re-run without ``--limit`` to top up the full set."""
    session = load_session(cookies_path)
    episodes = list_episodes(session, submission_id, successful_only=successful_only)

    submission_dir = Path(output_dir) / str(submission_id)
    submission_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed: list[int] = []

    for episode in episodes:
        if limit is not None and downloaded >= limit:
            print(f"  reached --limit {limit}; stopping (re-run without it to top up)")
            break
        episode_id = episode["id"]
        dest = submission_dir / f"{episode_id}.json"
        if dest.exists():
            skipped += 1
            continue

        time.sleep(REQUEST_DELAY_SECONDS)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                replay = fetch_replay(session, episode_id)
                dest.write_text(json.dumps(replay), encoding="utf-8")
                downloaded += 1
                break
            except requests.exceptions.HTTPError as exc:
                is_rate_limited = exc.response is not None and exc.response.status_code == 429
                if attempt == MAX_RETRIES:
                    print(f"  FAILED episode {episode_id}: {exc}")
                    failed.append(episode_id)
                elif is_rate_limited:
                    retry_after = float(exc.response.headers.get("Retry-After", 0))
                    wait = max(retry_after, 2**attempt)
                    print(f"  rate limited on episode {episode_id}, waiting {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    time.sleep(1)
            except Exception as exc:  # noqa: BLE001 - retry loop, report and move on
                if attempt == MAX_RETRIES:
                    print(f"  FAILED episode {episode_id}: {exc}")
                    failed.append(episode_id)
                else:
                    time.sleep(1)

        if downloaded % 10 == 0 and downloaded > 0:
            print(f"  ...{downloaded} downloaded so far")

    metadata = {
        "submission_id": submission_id,
        "total_episodes": len(episodes),
        "downloaded_this_run": downloaded,
        "skipped_existing": skipped,
        "failed": failed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (submission_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"submission {submission_id}: {downloaded} downloaded, "
        f"{skipped} already present, {len(failed)} failed "
        f"(out of {len(episodes)} episodes) -> {submission_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=int, required=True, help="Kaggle submission_id")
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
        help="max NEW replays to fetch this run (screening needs ~100-150; incremental, "
             "so re-running without it tops up the full set)",
    )
    args = parser.parse_args()

    download_submission_replays(
        submission_id=args.submission,
        cookies_path=args.cookies,
        output_dir=args.out,
        successful_only=not args.all,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
