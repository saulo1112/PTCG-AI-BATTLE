"""M32 follow-up: WHAT is the 'other' archetype bucket that beats imitation-mlp?

FASE 0b of diagnose_mlp.py found the champion's losses concentrate almost entirely
in decks our ARCHETYPE_MARKERS don't label: 'other' = 17W-31L (35%) while every
named archetype is 70-89%. This clusters the opponent decks in that bucket by
their exact 60-card list and names the key Pokemon, so the marker table can be
extended and the real bad matchups identified.

READ-ONLY. Run:
  uv run --group dev python scratchpad/diagnose_mlp_other.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.kaggle_replay import player_seats

import diagnose_mlp as dm  # noqa: E402
import extract_top_decks as et  # noqa: E402


def main() -> int:
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))

    def is_pokemon(cid: int) -> bool:
        c = cards.get_card(cid)
        return c is not None and bool(getattr(c, "is_basic_pokemon", False) or c.cardType == 0
                                      or getattr(c, "hp", 0))

    def label_cards(key: tuple[int, ...], k: int = 6) -> str:
        counts = collections.Counter(key)
        mons = [(c, n) for c, n in counts.items() if is_pokemon(c)]
        mons.sort(key=lambda cn: (-cn[1], cn[0]))
        return ", ".join(f"{cards.name(c)}x{n}" for c, n in mons[:k])

    # cluster every opponent deck (labelled or not) by its exact card multiset
    clusters: dict[tuple[int, ...], list[int]] = collections.defaultdict(lambda: [0, 0])
    cluster_label: dict[tuple[int, ...], str] = {}
    cluster_reasons: dict[tuple[int, ...], collections.Counter] = collections.defaultdict(collections.Counter)
    for folder in dm.LIVE_FOLDERS:
        for f in dm._files(folder):
            data = json.loads(f.read_text(encoding="utf-8"))
            seats = player_seats(data, dm.OUR)
            if len(seats) != 1:
                continue
            our = seats[0]
            deck = et.extract_deck(data, 1 - our)
            if not deck:
                continue
            key = tuple(sorted(deck))
            cluster_label[key] = dm._archetype(deck)
            rew = (data.get("rewards") or [0, 0])[our]
            if rew > 0:
                clusters[key][0] += 1
            else:
                clusters[key][1] += 1
                reason, _ = dm.loss_reason(data, our)
                cluster_reasons[key][reason] += 1

    rows = sorted(clusters.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))

    # aggregate by marker with the EXTENDED table (variant lists collapse into one archetype)
    by_marker: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    marker_reasons: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for key, (w, l) in rows:
        lab = cluster_label[key]
        by_marker[lab][0] += w
        by_marker[lab][1] += l
        marker_reasons[lab].update(cluster_reasons.get(key, {}))
    print("=" * 92)
    print("Per-archetype record with the EXTENDED marker table")
    print(f"  {'archetype':<22}{'n':>4}{'W':>4}{'L':>4}{'WR':>7}   loss reasons")
    for lab, (w, l) in sorted(by_marker.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        n = w + l
        print(f"  {lab:<22}{n:>4}{w:>4}{l:>4}{w/max(n,1):>6.0%}   {dict(marker_reasons[lab]) or ''}")
    print()
    print("=" * 92)
    print("Opponent decks faced by imitation-mlp, clustered by exact 60-card list")
    print(f"{'n':>4} {'W':>3} {'L':>3} {'WR':>6}  {'marker':<14} key Pokemon / loss reasons")
    print("-" * 92)
    unlabeled_w = unlabeled_l = 0
    for key, (w, l) in rows:
        n = w + l
        lab = cluster_label[key]
        top = [label_cards(key)]
        if lab == "other":
            unlabeled_w += w
            unlabeled_l += l
        reasons = dict(cluster_reasons.get(key, {}))
        print(f"{n:>4} {w:>3} {l:>3} {w/max(n,1):>5.0%}  {lab:<14} {', '.join(top)}")
        if reasons:
            print(f"{'':>30}  losses: {reasons}")
    print("-" * 92)
    print(f"unlabeled ('other') total: {unlabeled_w}W-{unlabeled_l}L "
          f"= {unlabeled_w/max(unlabeled_w+unlabeled_l,1):.0%}")
    print(f"distinct opponent decks: {len(rows)}")

    # full decklist of the worst matchups (>=2 games, WR <= 40%)
    print("\n" + "=" * 92)
    print("FULL DECKLISTS — worst matchups (n>=2, WR<=40%)")
    for key, (w, l) in rows:
        n = w + l
        if n < 2 or w / n > 0.40:
            continue
        print(f"\n[{w}W-{l}L] marker={cluster_label[key]}")
        for c, k in sorted(collections.Counter(key).items(), key=lambda ck: (-ck[1], ck[0])):
            print(f"    {k}x {cards.name(c):<34} ({c})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
