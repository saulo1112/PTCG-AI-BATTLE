"""M32 discriminator: is the Grimmsnarl wall a DECK weakness or a CLONE gap?

M32 found imitation-mlp goes 10W-17L (37%) vs Marnie's Grimmsnarl while going 69%
vs the rest of the field. Two incompatible explanations:

  (a) DECK/ARCHETYPE weakness — Yushin himself also loses that matchup with this
      deck. Then no amount of piloting fixes it; the lever is a different deck.
  (b) CLONE gap — Yushin BEATS Grimmsnarl but the clone doesn't. Then the clone
      fails to reproduce matchup-specific play, and the lever is training, not deck.

We have Yushin's own 1284 episodes on disk, so this is directly measurable: compute
HIS win rate per opponent archetype and put it beside ours.

Seat detection is by DECK IDENTITY (his 60-card list), not by player name, so it
works regardless of how the replay labels the agents.

READ-ONLY. Run:
  uv run --group dev python scratchpad/diagnose_mlp_teacher_matchup.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import diagnose_mlp as dm  # noqa: E402
import extract_top_decks as et  # noqa: E402

TEACHER_FOLDER = Path("replays/54773249")


def teacher_record() -> tuple[dict[str, list[int]], dict[str, collections.Counter], int, int]:
    """His W/L per opponent archetype, seat found by matching his 60-card list."""
    yushin = tuple(sorted(int(x) for x in dm.DECK_CSV.read_text(encoding="utf-8").split()))
    by_arch: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    reasons: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    skipped = mirror = 0
    for f in dm._files(TEACHER_FOLDER):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue
        decks = [et.extract_deck(data, s) for s in (0, 1)]
        seats = [s for s in (0, 1) if decks[s] and tuple(sorted(decks[s])) == yushin]
        if len(seats) != 1:          # 0 = deck not recoverable, 2 = true mirror
            if len(seats) == 2:
                mirror += 1
            else:
                skipped += 1
            continue
        our = seats[0]
        arch = dm._archetype(decks[1 - our])
        rew = (data.get("rewards") or [0, 0])[our]
        if rew > 0:
            by_arch[arch][0] += 1
        else:
            by_arch[arch][1] += 1
            reason, _ = dm.loss_reason(data, our)
            reasons[arch][reason] += 1
    return by_arch, reasons, skipped, mirror


def our_record() -> dict[str, list[int]]:
    out: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    from ptcg_ai.imitation.kaggle_replay import player_seats
    for folder in dm.LIVE_FOLDERS:
        for f in dm._files(folder):
            data = json.loads(f.read_text(encoding="utf-8"))
            seats = player_seats(data, dm.OUR)
            if len(seats) != 1:
                continue
            our = seats[0]
            arch = dm._archetype(et.extract_deck(data, 1 - our))
            rew = (data.get("rewards") or [0, 0])[our]
            out[arch][0 if rew > 0 else 1] += 1
    return out


def main() -> int:
    them, reasons, skipped, mirror = teacher_record()
    us = our_record()

    tw = sum(v[0] for v in them.values()); tl = sum(v[1] for v in them.values())
    uw = sum(v[0] for v in us.values()); ul = sum(v[1] for v in us.values())

    print("=" * 88)
    print("Win rate by opponent archetype — TEACHER (Yushin, 54773249) vs CLONE (imitation-mlp)")
    print(f"(teacher: {tw}W-{tl}L = {tw/max(tw+tl,1):.1%} over {tw+tl} games; "
          f"{mirror} true mirrors + {skipped} unrecoverable excluded)")
    print(f"(clone:   {uw}W-{ul}L = {uw/max(uw+ul,1):.1%} over {uw+ul} games)")
    print()
    print(f"  {'archetype':<22}{'teacher W-L':>14}{'WR':>7} | {'clone W-L':>11}{'WR':>7}{'delta':>8}")
    print("  " + "-" * 76)
    keys = sorted(set(them) | set(us), key=lambda k: -(them.get(k, [0, 0])[0] + them.get(k, [0, 0])[1]))
    for k in keys:
        t = them.get(k, [0, 0]); u = us.get(k, [0, 0])
        tn, un = t[0] + t[1], u[0] + u[1]
        twr = t[0] / tn if tn else None
        uwr = u[0] / un if un else None
        d = f"{uwr - twr:+.0%}" if (twr is not None and uwr is not None) else "—"
        print(f"  {k:<22}{f'{t[0]}-{t[1]}':>14}{(f'{twr:.0%}' if twr is not None else '—'):>7} | "
              f"{f'{u[0]}-{u[1]}':>11}{(f'{uwr:.0%}' if uwr is not None else '—'):>7}{d:>8}")
    print()
    print("  teacher loss reasons, top archetypes:")
    for k in keys[:6]:
        if reasons.get(k):
            print(f"    {k:<22}{dict(reasons[k])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
