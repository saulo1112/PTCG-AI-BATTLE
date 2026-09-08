"""Build + validate cheap-attacker Psychic decks (v2, diagnosed fix).

The v1 all-ex deck failed: attackers cost 3 energy, our pilot attaches 1/turn,
so they never charged (5 attacks in 6 games). v2 uses CHEAP attackers that our
1-attach pilot can actually power:
  - Iron Boulder (971): 170 dmg for PSY+COL, 140hp, NON-ex (1 prize)
  - Hop's Cramorant (311): 120 dmg for 1 COL, 110hp, NON-ex (1 prize)
Two variants: 'engine' adds Fezandipiti ex draw engine; 'pure' is all-non-ex.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

from ptcg_ai.agent.deck import Deck, validate_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk

DECKS = {
    # cheap attackers + Fezandipiti ex draw engine
    "engine": [
        (971, 4),   # Iron Boulder    170/2 PSY+COL, 140hp, 1-prize
        (311, 4),   # Hop's Cramorant 120/1 COL, 110hp, 1-prize
        (140, 4),   # Fezandipiti ex  210hp, "draw 3" ability + 100 snipe
        (1102, 4),  # Dusk Ball
        (1152, 4),  # Poke Pad
        (1227, 4),  # Lillie's Determination
        (1205, 2),  # Cyrano
        (5, 34),    # Basic {P} Energy
    ],
    # pure non-ex (zero prize liability); more attackers, trainer draw only
    "pure": [
        (971, 4),   # Iron Boulder
        (311, 4),   # Hop's Cramorant
        (216, 3),   # Mesprit  160/2 (needs bench) + Full Heart energy accel; filler body
        (1102, 4),  # Dusk Ball
        (1152, 4),  # Poke Pad
        (1227, 4),  # Lillie's Determination
        (1205, 3),  # Cyrano
        (5, 34),    # Basic {P} Energy
    ],
}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "engine"
    spec = DECKS[which]
    cfg = load_config(profile="benchmark")
    sdk = load_sdk(cfg.paths.sdk_dir)
    cards = CardDatabase.from_sdk(sdk)
    ids: list[int] = []
    for cid, n in spec:
        ids.extend([cid] * n)
    print(f"[{which}] total cards: {len(ids)}")
    for cid, n in sorted(collections.Counter(ids).items(), key=lambda kv: -kv[1]):
        info = cards.get_card(cid)
        nm = info.name if info else "??"
        typ = info.cardType.name if info else "?"
        print(f"  {n}x [{cid}] {nm} ({typ})")
    problems = validate_deck(Deck(card_ids=tuple(ids)), cards)
    print(f"legal? {'YES' if not problems else problems}")
    if not problems:
        out = Path(f"decks/psychic_{which}.csv")
        out.write_text("\n".join(str(c) for c in ids) + "\n", encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
