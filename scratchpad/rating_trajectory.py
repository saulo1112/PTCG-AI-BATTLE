"""M44 — rating trajectory + implied rating for any submission, from the Kaggle API.

WHY THIS EXISTS. Every ladder question in this project ("did it converge?", "is the
decline real or the opponent draw?", "is the whole ladder deflating?") was answered
with throwaway inline scripts in M29/M43. This makes it a tool.

THE TWO NUMBERS THAT MATTER:

  implied rating = opp_mean + 400*log10(WR/(1-WR))

    The raw leaderboard score mixes skill with WHO you drew. M29 proved the SAME
    bundle scored 773 / 857 / 899 across uploads purely from the opponent field.
    The implied rating corrects for that and is the only number comparable across
    submissions.

  slope over the last half of episodes

    Distinguishes "still climbing" from "converged". A converged agent will not
    improve with more time no matter how long you wait; a climbing one that gets
    cut off (as 55145833 was, at 899 and rising) leaves elo on the table.

`--by-day` prints the daily series, which is how you see BOTH the episode-rate decay
(a new submission gets a burst then starves) and any long-run drift of the ladder
itself — point it at a competitor who has not touched their bot in weeks and the
drift you see is the SYSTEM, not the agent.

Read-only: it only calls ListEpisodes. Nothing is written.

Run:
  PYTHONIOENCODING=utf-8 python scratchpad/rating_trajectory.py 55438687 55438655
  PYTHONIOENCODING=utf-8 python scratchpad/rating_trajectory.py 54773249 --by-day
"""

from __future__ import annotations

import argparse
import collections
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "tools")
from kaggle_api import list_episodes, load_session  # noqa: E402

COOKIES = "tools/cookies.txt.json"


def fetch(session, sid: int, retries: int = 3) -> list[dict]:
    """Episodes for `sid`, chronological, with our side and the opponent split out.

    Kaggle drops the connection under rapid repeated calls, so this retries with a
    pause rather than failing a multi-submission run halfway through.
    """
    eps = None
    for attempt in range(retries):
        try:
            eps = list_episodes(session, sid)
            break
        except Exception as exc:  # noqa: BLE001 — transient connection resets
            if attempt == retries - 1:
                raise
            print(f"  (reintento {attempt + 1}: {exc})")
            time.sleep(15)
    rows = []
    for e in eps or []:
        mine = opp = None
        for a in e.get("agents", []):
            if a.get("submissionId") == sid:
                mine = a
            else:
                opp = a
        if mine is None or opp is None:
            continue
        rows.append({
            "t": e.get("createTime") or "",
            "reward": mine.get("reward"),
            "score": mine.get("updatedScore"),
            "init": mine.get("initialScore"),
            "opp": opp.get("updatedScore") or opp.get("initialScore"),
            "opp_sid": opp.get("submissionId"),
            "opp_team": opp.get("teamId"),
        })
    rows.sort(key=lambda r: r["t"])
    return rows


def implied(rows) -> tuple[int, int, int, float, float, float]:
    w = sum(1 for r in rows if (r["reward"] or 0) > 0)
    l = sum(1 for r in rows if (r["reward"] or 0) < 0)
    d = len(rows) - w - l
    wr = (w + 0.5 * d) / max(len(rows), 1)
    opps = [r["opp"] for r in rows if r["opp"] is not None]
    om = statistics.mean(opps) if opps else 0.0
    # clamp so a 0% or 100% run does not blow up the logit
    wrc = min(max(wr, 1e-3), 1 - 1e-3)
    return w, l, d, wr, om, om + 400 * math.log10(wrc / (1 - wrc))


def slope(values: list[float]) -> float:
    """Least-squares slope per episode over the last half (the converged tail)."""
    half = values[len(values) // 2:] if len(values) > 4 else values
    n = len(half)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(half) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, half)) / max(den, 1e-9)


def report(sid: int, rows: list[dict], by_day: bool) -> None:
    if not rows:
        print(f"\n=== {sid}: sin episodios ===")
        return
    w, l, d, wr, om, imp = implied(rows)
    scores = [r["score"] for r in rows if r["score"] is not None]
    sl = slope(scores)
    verdict = "SUBIENDO" if sl > 0.3 else ("BAJANDO" if sl < -0.3 else "PLANO/CONVERGIDO")
    print(f"\n=== submission {sid} ===")
    print(f"  n={len(rows)}   {w}W-{l}L-{d}D   WR={wr:.3f}")
    print(f"  rival medio        : {om:.1f}")
    print(f"  RATING IMPLICITO   : {imp:.1f}")
    print(f"  score reportado    : {scores[-1]:.1f}   (arranco en {rows[0]['init']:.1f})")
    print(f"  pendiente 2a mitad : {sl:+.2f}/episodio  -> {verdict}")

    q = max(len(scores) // 4, 1)
    quarters = [scores[i * q:(i + 1) * q] if i < 3 else scores[3 * q:] for i in range(4)]
    print("  por cuartos (score): " + "  ".join(
        f"Q{i+1} {statistics.mean(s):.0f}" for i, s in enumerate(quarters) if s))
    opps = [r["opp"] for r in rows if r["opp"] is not None]
    oq = [opps[i * q:(i + 1) * q] if i < 3 else opps[3 * q:] for i in range(4)]
    print("  por cuartos (rival): " + "  ".join(
        f"Q{i+1} {statistics.mean(s):.0f}" for i, s in enumerate(oq) if s))

    if by_day:
        per_day = collections.defaultdict(list)
        for r in rows:
            per_day[r["t"][:10]].append(r)
        print(f"\n  {'dia':<12}{'eps':>5}{'WR':>7}{'rival':>8}{'score fin':>11}")
        for day in sorted(per_day):
            rs = per_day[day]
            dw = sum(1 for r in rs if (r["reward"] or 0) > 0)
            do = [r["opp"] for r in rs if r["opp"] is not None]
            ds = [r["score"] for r in rs if r["score"] is not None]
            print(f"  {day:<12}{len(rs):>5}{dw/len(rs):>7.2f}"
                  f"{statistics.mean(do) if do else 0:>8.0f}{ds[-1] if ds else 0:>11.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("submissions", nargs="+", type=int)
    ap.add_argument("--by-day", action="store_true", help="serie diaria completa")
    ap.add_argument("--cookies", default=COOKIES)
    args = ap.parse_args()

    session = load_session(Path(args.cookies))
    results = []
    for i, sid in enumerate(args.submissions):
        if i:
            time.sleep(8)  # be gentle; Kaggle resets the connection under rapid calls
        rows = fetch(session, sid)
        report(sid, rows, args.by_day)
        if rows:
            results.append((sid, *implied(rows), len(rows)))

    if len(results) > 1:
        print(f"\n{'submission':>12}{'n':>6}{'WR':>8}{'rival':>9}{'IMPLICITO':>11}")
        for sid, w, l, d, wr, om, imp, n in results:
            print(f"{sid:>12}{n:>6}{wr:>8.3f}{om:>9.1f}{imp:>11.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
