"""M29 — why did the "same" Kanga score 773 (54911514) vs 900 (54988272)? READ-ONLY.

Verifies same-agent identity (byte-identical 60-card deck) and dissects the ladder gap:
record/WR, opponent-archetype field, opponent-name overlap, and EpisodeId-ordered W/L
(chronology proxy — replays carry no wall-clock, but EpisodeId is monotonic). Answers:
is the score gap time/field-correlated or luck, and how noisy is a single converged score.

Run:  uv run --group dev python scratchpad/compare_submissions.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from ptcg_ai.imitation.kaggle_replay import player_seats

import diagnose_kanga as dk
import extract_top_decks as et

OUR = dk.OUR
CANON = tuple(sorted(int(x) for x in Path("decks/懒惰的金枪鱼.csv").read_text(encoding="utf-8").split()))
SUBS = [("54911514 (~773)", Path("replays/54911514")),
        ("54988272 (~900)", Path("replays/54988272"))]


def analyze(folder: Path) -> dict:
    files = sorted(p for p in folder.glob("*.json") if p.name != "metadata.json")
    w = l = selfm = 0
    our_decks: collections.Counter = collections.Counter()
    opp_arch: collections.Counter = collections.Counter()
    opp_names: collections.Counter = collections.Counter()
    seq = []  # (episodeId, 'W'/'L', opp_name, archetype)
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        seats = player_seats(data, OUR)
        eid = data.get("info", {}).get("EpisodeId")
        if len(seats) != 1:
            selfm += 1
            if seats:
                d = et.extract_deck(data, seats[0])
                if d:
                    our_decks[tuple(sorted(d))] += 1
            continue
        our = seats[0]
        rew = (data.get("rewards") or [0, 0])[our]
        res = "W" if rew > 0 else "L"
        w += rew > 0
        l += rew <= 0
        d = et.extract_deck(data, our)
        if d:
            our_decks[tuple(sorted(d))] += 1
        opp_seat = 1 - our
        agents = data.get("info", {}).get("Agents", [])
        opp_name = agents[opp_seat].get("Name") if opp_seat < len(agents) else "?"
        opp_names[opp_name] += 1
        od = et.extract_deck(data, opp_seat)
        arch = "other"
        if od:
            labs = [lab for cid, lab in dk.ARCHETYPE_MARKERS.items() if cid in od]
            arch = labs[0] if labs else "other"
        opp_arch[arch] += 1
        seq.append((eid if eid is not None else 0, res, opp_name, arch))
    return dict(files=len(files), w=w, l=l, selfm=selfm, our_decks=our_decks,
                opp_arch=opp_arch, opp_names=opp_names, seq=seq)


def main() -> int:
    res = {name: analyze(folder) for name, folder in SUBS}

    print("=" * 74)
    print("PHASE B — SAME AGENT? (our fixed 60-card deck)")
    top_decks = {}
    for name, r in res.items():
        top, cnt = (r["our_decks"].most_common(1) or [((), 0)])[0]
        top_decks[name] = top
        print(f"  {name}: {len(r['our_decks'])} distinct decks; top deck in {cnt} games; "
              f"== canonical kanga? {top == CANON}")
    names = list(res)
    print(f"  deck(54911514) == deck(54988272)? {top_decks[names[0]] == top_decks[names[1]]}")

    print("\n" + "=" * 74)
    print("PHASE C — RECORD & FIELD")
    for name, r in res.items():
        tot = r["w"] + r["l"]
        print(f"\n[{name}]  {r['files']} files | {r['w']}W-{r['l']}L "
              f"(WR {r['w']/max(tot,1):.1%}) | self-matches {r['selfm']}")
        print(f"  opponent archetypes: {dict(r['opp_arch'].most_common())}")
        print(f"  distinct opponents: {len(r['opp_names'])} | top: "
              f"{dict(r['opp_names'].most_common(6))}")

    # opponent-name overlap between the two runs
    o1, o2 = set(res[names[0]]["opp_names"]), set(res[names[1]]["opp_names"])
    print(f"\n  opponent OVERLAP: {len(o1 & o2)} shared, "
          f"{len(o1 - o2)} only in 54911514, {len(o2 - o1)} only in 54988272")

    print("\n" + "=" * 74)
    print("PHASE C — CHRONOLOGY (W/L ordered by EpisodeId; first-half vs second-half WR)")
    for name, r in res.items():
        s = sorted(r["seq"], key=lambda t: t[0])
        results = [x[1] for x in s]
        n = len(results)
        if n == 0:
            continue
        h = n // 2
        fh = results[:h].count("W") / max(h, 1)
        sh = results[h:].count("W") / max(n - h, 1)
        streak = "".join("W" if x == "W" else "." for x in results)
        print(f"\n[{name}]  n={n}  first-half WR {fh:.0%}  second-half WR {sh:.0%}")
        print(f"  {streak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
