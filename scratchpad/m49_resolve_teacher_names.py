"""M49 — derive the teacher's display names from submissionId, not from memory.

THE BUG THIS EXISTS TO PREVENT, caught live in M49. The whole imitation pipeline keys on
a DISPLAY NAME: `build_decision_dataset(log_dir, player_name, out)` ->
`player_seats(data, player_name)` matches `info.Agents[].Name` case-insensitively. But a
Kaggle team can RENAME itself, and this teacher did -- twice. The 2,330 replays from the
2026-08-02 snapshot say `Yushin Ito`; the ~738 new ones say `AlphaStarmie` (and at least
one episode shows `AlphaTCG`). Same team, same submission 54773249, different string.

Left alone, that is a SILENT failure, not a loud one: rebuilding the corpus with
"Yushin Ito" would have matched zero seats in every new replay and written a corpus the
same size as the old one, with no error and no warning -- and the context heads would
have been retrained believing they had +32% data while having none. Exactly the shape of
M37's wrong-deck submission, which passed every structural check.

`submissionId` is the stable identifier and the API carries it per episode agent. This
resolves the seat that way and reports which display names that seat actually used, so
the extraction can be driven by a VERIFIED name set instead of a remembered one.

It also cross-checks the other direction: any name found on the teacher's seat must NEVER
appear on the opponent seat of another episode, otherwise using it as a filter would
silently pull in someone else's decisions.

Reads only the first 4 KB of each replay (the `info` block sits at ~byte 187), so a
several-GB folder is scanned in seconds instead of minutes.

Read-only apart from the audit JSON it writes.

Run:  PYTHONIOENCODING=utf-8 python -u scratchpad/m49_resolve_teacher_names.py
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "tools")
from kaggle_api import list_episodes, load_session  # noqa: E402

SUBMISSION = 54773249
REPLAY_DIR = Path("replays/54773249")
COOKIES = "tools/cookies.txt.json"
OUT = Path("data/imitation/teacher_name_audit.json")
#: Known from the pre-API era: every replay in the 2026-08-02 snapshot uses this.
LEGACY_NAME = "Yushin Ito"

_AGENTS_RE = re.compile(r'"Agents": \[(.*?)\], "EpisodeId"', re.S)
_NAME_RE = re.compile(r'"Name": "(.*?)"')


def _header_names(path: Path) -> list[str] | None:
    try:
        head = path.open("rb").read(4096).decode("utf-8", "replace")
    except OSError:
        return None
    m = _AGENTS_RE.search(head)
    if not m:
        return None
    names = _NAME_RE.findall(m.group(1))
    return names if len(names) == 2 else None


def main() -> int:
    session = load_session(COOKIES)
    episodes = list_episodes(session, SUBMISSION)
    print(f"API returned {len(episodes)} episodes for submission {SUBMISSION}")

    seat_of: dict[int, int] = {}
    for e in episodes:
        for a in e.get("agents", []):
            if a.get("submissionId") == SUBMISSION:
                seat_of[e["id"]] = int(a.get("index", 0))
                break

    ours = collections.Counter()
    theirs = collections.Counter()
    on_disk = missing_header = 0

    for f in sorted(REPLAY_DIR.glob("*.json")):
        if f.name == "metadata.json":
            continue
        try:
            eid = int(f.stem)
        except ValueError:
            continue
        names = _header_names(f)
        if names is None:
            missing_header += 1
            continue
        on_disk += 1
        seat = seat_of.get(eid)
        if seat is None:
            # Outside the API's most-recent-1000 window (the old snapshot). Its seat
            # cannot be resolved from the API; it is covered by LEGACY_NAME instead.
            continue
        ours[names[seat]] += 1
        theirs[names[1 - seat]] += 1

    print(f"{on_disk} replays scanned ({missing_header} unreadable headers)")
    print(f"{len(seat_of)} episodes resolvable via the API window\n")

    print("=" * 70)
    print("DISPLAY NAMES USED BY THE TEACHER'S OWN SEAT (resolved by submissionId)")
    print("=" * 70)
    if not ours:
        print("  NONE RESOLVED — cannot proceed. Either the download has not reached the")
        print("  API window yet, or the seat mapping is wrong. Do NOT guess a name.")
        return 1
    for n, c in ours.most_common():
        print(f"  {c:5d}  {n!r}")

    collisions = {n for n in ours if n in theirs}
    print("\n" + "=" * 70)
    print("COLLISION CHECK — a name on our seat must never appear on an opponent seat")
    print("=" * 70)
    if collisions:
        print("  *** UNSAFE AS A FILTER ***")
        for n in sorted(collisions):
            print(f"    {n!r}: ours={ours[n]} opponent={theirs[n]}")
        print("  Filtering by these names would pull in another player's decisions.")
        return 1
    print("  clean — none of the teacher's names appear on any opponent seat")

    name_set = sorted(set(ours) | {LEGACY_NAME})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "submission_id": SUBMISSION,
        "names": name_set,
        "resolved_counts": dict(ours),
        "legacy_name": LEGACY_NAME,
        "note": "Names resolved from the API by submissionId, not from memory. The team "
                "renamed at least twice; the display name is NOT a stable key.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"NAME SET for corpus extraction  ->  {OUT}")
    print("=" * 70)
    for n in name_set:
        src = f"{ours[n]} API-resolved" if n in ours else "legacy snapshot era"
        print(f"  {n!r}  ({src})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
