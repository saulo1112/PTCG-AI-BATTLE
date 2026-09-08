"""Thin client for the Kaggle internal API used to list and fetch simulation replays.

Not part of the shipped submission — this only talks to kaggle.com to pull
down episode replay JSONs for offline analysis / imitation-learning data.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

LEADERBOARD_ORIGIN = "https://www.kaggle.com"
LIST_EPISODES_URL = f"{LEADERBOARD_ORIGIN}/api/i/competitions.EpisodeService/ListEpisodes"
REPLAY_URL_TEMPLATE = f"{LEADERBOARD_ORIGIN}/competitions/episodes/{{episode_id}}/replay.json"


def load_session(cookies_path: str | Path) -> requests.Session:
    """Build an authenticated requests.Session from an exported cookie-array JSON file.

    Expects the format produced by Chrome cookie-export extensions (a JSON list of
    objects with at least "name", "value", "domain", "path"), not Netscape cookies.txt.
    """
    cookies_path = Path(cookies_path)
    cookies = json.loads(cookies_path.read_text(encoding="utf-8"))

    session = requests.Session()
    xsrf_token = None
    for cookie in cookies:
        domain = cookie["domain"].lstrip(".")
        session.cookies.set(
            cookie["name"],
            cookie["value"],
            domain=domain,
            path=cookie.get("path", "/"),
        )
        if cookie["name"] == "XSRF-TOKEN":
            xsrf_token = cookie["value"]

    if xsrf_token is None:
        raise ValueError(f"No XSRF-TOKEN cookie found in {cookies_path}")

    session.headers.update(
        {
            "content-type": "application/json",
            "accept": "application/json",
            "x-xsrf-token": xsrf_token,
            "origin": LEADERBOARD_ORIGIN,
        }
    )
    return session


def list_episodes(
    session: requests.Session,
    submission_id: int,
    successful_only: bool = True,
    include_in_progress: bool = False,
) -> list[dict]:
    """Return the list of episode dicts for a given submission_id."""
    payload = {
        "ids": [],
        "submissionId": submission_id,
        "successfulOnly": successful_only,
        "includeInProgress": include_in_progress,
    }
    response = session.post(LIST_EPISODES_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    return data.get("episodes", [])


def fetch_replay(session: requests.Session, episode_id: int) -> dict:
    """Download the full replay JSON for a single episode."""
    response = session.get(REPLAY_URL_TEMPLATE.format(episode_id=episode_id))
    response.raise_for_status()
    return response.json()
