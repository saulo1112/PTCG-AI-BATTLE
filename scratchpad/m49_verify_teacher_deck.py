"""M49 Fase 2 — did the teacher change decks, and is his decline the pilot or the field?

WHY THIS RUNS BEFORE ANY TRAINING. The champion clones Yushin Ito (submission 54773249),
whose corpus on disk is a 2026-08-02 snapshot of 2,330 games. The Kaggle API shows the
submission is still live (33 episodes on 2026-08-14), so ~738 new games are available --
a +32% corpus. But two things must be checked before a single row of that data is mixed
in, because both have precedent IN THIS PROJECT:

  (1) HE MAY HAVE SWITCHED DECKS. M45 caught Majkel1337 (world #1) swapping Alakazam ->
      Mega Lucario near the deadline; M46 caught やる気元気ミワハルキ swapping Grimmsnarl ->
      Mega Lopunny. If Yushin did the same, the new games are a DIFFERENT AGENT and
      merging them corrupts the corpus rather than enlarging it -- every context head
      would be fitting two policies at once, the same failure M24 measured when pooling
      three pilots of one deck (-0.024 fidelity).

  (2) HE IS DECLINING. Score 1196.8 (Aug 2) -> 1043.4 (Aug 14); WR 0.26 on Aug 13; mean
      opponent rating falling 1143 -> 1006, i.e. he is sliding DOWN the ladder. Two
      readings compete and they imply OPPOSITE actions: if he now loses to the SAME
      archetypes he used to beat, the new data is lower-quality pilot play and the recent
      tail should be excluded; if the FIELD merely shifted (M32/M43 measured Grimmsnarl
      going 17% -> 51% of the top band), his play is unchanged and all the data is good.

SCOPE, deliberately narrow: only files newer than the snapshot are parsed. The old era is
NOT re-parsed -- it is ~20 GB and ~25 minutes for numbers M33 already published on the
same 1,313-game corpus (embedded below as `M33_BASELINE`). Re-deriving them would cost
half an hour to reproduce a table we already trust.

The deck check compares the 60-card submission action as a MULTISET against
decks/yushinito.csv, so card COUNTS matter, not just which cards appear -- the same method
M45/M46 used to confirm/reject candidates.

Read-only. Writes nothing.

Run:  PYTHONIOENCODING=utf-8 python -u scratchpad/m49_verify_teacher_deck.py
"""

from __future__ import annotations

import collections
import json
from datetime import datetime, timezone
from pathlib import Path

REPLAY_DIR = Path("replays/54773249")
DECK_CSV = Path("decks/yushinito.csv")
CARD_DATA = Path("build/colab_export/card_data.json")
#: Resolved from the Kaggle API by submissionId, NOT from memory. The team renamed
#: TWICE (Yushin Ito -> AlphaStarmie -> AlphaTCG), and matching a single stale name
#: yields zero seats in the newer replays without raising anything.
NAME_AUDIT = Path("data/imitation/teacher_name_audit.json")
DECK_LEN = 60
#: Files at/before this are the corpus we already trained on (downloaded 2026-08-02 18:50).
SNAPSHOT = datetime(2026, 8, 3, tzinfo=timezone.utc)

#: M33, measured over his 1,313 non-mirror games. archetype -> (WR, share of field).
M33_BASELINE = {
    "marnie_grimmsnarl": (0.519, 0.581),
    "team_rocket":       (0.262, 0.096),
    "mega_kangaskhan":   (0.725, 0.100),
    "alakazam(mirror)":  (0.875, 0.049),
    "mega_lucario":      (0.808, 0.020),
}

#: Archetype markers, id -> label (the project's inherited table; M32 added several).
#: Order matters: the first marker found in the opponent deck wins.
MARKERS = {
    648: "marnie_grimmsnarl", 756: "mega_kangaskhan", 121: "dragapult",
    1191: "archaludon", 104: "mega_lucario", 400: "team_rocket", 743: "alakazam(mirror)",
}


def _deck_ids() -> collections.Counter:
    return collections.Counter(int(x) for x in DECK_CSV.read_text(encoding="utf-8").split())


def _master_names() -> list[str]:
    if not NAME_AUDIT.is_file():
        raise SystemExit(
            f"missing {NAME_AUDIT} — run scratchpad/m49_resolve_teacher_names.py first.\n"
            "Deliberately NO hardcoded fallback: a stale name matches zero seats in the "
            "newer replays and this script would then 'pass' on an empty sample.")
    return json.loads(NAME_AUDIT.read_text(encoding="utf-8"))["names"]


def _seats(data: dict, names: list[str]) -> list[int]:
    wanted = {n.lower() for n in names}
    agents = data.get("info", {}).get("Agents", [])
    return [i for i, a in enumerate(agents) if (a.get("Name") or "").lower() in wanted]


def _deck_action(steps, seat):
    for row in steps[:4]:
        cell = row[seat] if seat < len(row) else None
        if not cell:
            continue
        action = cell.get("action")
        if isinstance(action, list) and len(action) == DECK_LEN:
            return action
    return None


def main() -> int:
    if not REPLAY_DIR.is_dir():
        raise SystemExit(f"missing {REPLAY_DIR}")
    ref = _deck_ids()
    master_names = _master_names()
    names = {}
    if CARD_DATA.is_file():
        names = {c["cardId"]: c["name"]
                 for c in json.loads(CARD_DATA.read_text(encoding="utf-8"))["cards"]}

    allf = [f for f in sorted(REPLAY_DIR.glob("*.json")) if f.name != "metadata.json"]
    new = [f for f in allf
           if datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc) > SNAPSHOT]
    print(f"{len(allf)} replays on disk; {len(new)} newer than the snapshot -- parsing "
          f"ONLY those ({len(new) * 8.7 / 1024:.1f} GB)\n", flush=True)
    if not new:
        print("no new replays found. Did the download finish?")
        return 1

    matched = mismatched = skipped = 0
    mismatch_examples = []
    n_by = collections.Counter()
    w_by = collections.Counter()

    for i, f in enumerate(new, 1):
        if i % 100 == 0:
            print(f"  ...{i}/{len(new)}", flush=True)
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue
        seats = _seats(data, master_names)
        steps = data.get("steps") or []
        rewards = data.get("rewards") or []
        if not seats or len(steps) < 2 or len(rewards) != 2:
            skipped += 1
            continue
        seat = seats[0]
        deck = _deck_action(steps, seat)
        if deck is None:
            skipped += 1
            continue

        got = collections.Counter(deck)
        if got == ref:
            matched += 1
        else:
            mismatched += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append((f.name, got - ref, ref - got))

        opp_seat = 1 - seat
        opp = set(_deck_action(steps, opp_seat) or [])
        label = next((lab for cid, lab in MARKERS.items() if cid in opp), "other")
        r_me, r_op = rewards[seat], rewards[opp_seat]
        if r_me is None or r_op is None or r_me == r_op:
            continue
        n_by[label] += 1
        w_by[label] += int(r_me > r_op)

    print("\n" + "=" * 78)
    print("2.1  DECK IDENTITY — is the new data the SAME agent?")
    print("=" * 78)
    print(f"  new games parsed        : {matched + mismatched}   (skipped {skipped})")
    print(f"  deck matches yushinito  : {matched}")
    print(f"  deck DIFFERS            : {mismatched}")
    if matched + mismatched == 0:
        # THE GUARD THIS FILE LEARNED THE HARD WAY. With nothing parsed,
        # `mismatched == 0` is trivially true, and the first version of this script
        # printed "OK, safe to merge" after skipping all 125 files it looked at --
        # a pass declared on zero evidence, the same shape as M37's wrong-deck bundle
        # clearing every structural check.
        print("\n  *** NOTHING PARSED — this is a FAILURE, not a pass. ***")
        print("  Every new replay was skipped, so the deck was never actually compared.")
        print("  Most likely the name set is stale (the team renamed) or the download")
        print("  has not reached the new episodes yet. Do NOT proceed to training.")
        return 1
    if mismatched == 0:
        print("\n  OK — every new game submits the exact same 60-card multiset.")
        print("  Same agent, same deck. Safe to merge and retrain.")
    else:
        share = mismatched / max(matched + mismatched, 1)
        print(f"\n  *** {share:.1%} of new games use a DIFFERENT deck ***")
        for fn, extra, missing in mismatch_examples:
            print(f"    {fn}")
            for cid, k in extra.items():
                print(f"        +{k}x {cid} {names.get(cid, '?')}")
            for cid, k in missing.items():
                print(f"        -{k}x {cid} {names.get(cid, '?')}")
        print("\n  He switched decks (the M45/M46 pattern). STOP: do not merge the new")
        print("  games wholesale. Either filter to deck-matching games only, or abandon")
        print("  the enlargement and keep the 2,330-game corpus as it stands.")

    print("\n" + "=" * 78)
    print("2.2  THE DECLINE — pilot, or field?")
    print("=" * 78)
    total = sum(n_by.values())
    print(f"{'archetype':<22}{'new n':>7}{'new WR':>8}{'new %':>8}"
          f"{'M33 WR':>9}{'M33 %':>8}{'dWR':>8}")
    print("-" * 78)
    for lab, n in n_by.most_common():
        wr = w_by[lab] / n
        share = n / max(total, 1)
        base = M33_BASELINE.get(lab)
        if base:
            print(f"{lab:<22}{n:>7}{wr:>8.3f}{share:>8.1%}"
                  f"{base[0]:>9.3f}{base[1]:>8.1%}{wr - base[0]:>+8.3f}")
        else:
            print(f"{lab:<22}{n:>7}{wr:>8.3f}{share:>8.1%}{'--':>9}{'--':>8}{'--':>8}")
    print("-" * 78)
    overall = sum(w_by.values()) / max(total, 1)
    print(f"{'OVERALL':<22}{total:>7}{overall:>8.3f}{'':>8}{0.569:>9.3f}"
          f"{'':>8}{overall - 0.569:>+8.3f}   (M33 overall 56.9%)")
    print()
    print("READ IT THIS WAY:")
    print("  * per-archetype dWR ~flat, but the MIX shifted toward hard matchups")
    print("      -> the field moved, the pilot is fine. Train on ALL the new data.")
    print("  * per-archetype dWR clearly NEGATIVE on archetypes he used to beat")
    print("      -> pilot decline. Consider excluding the recent tail.")
    print("  Small per-archetype n makes single rows noisy — weigh the big buckets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
