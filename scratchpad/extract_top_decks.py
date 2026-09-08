"""Reconstruct the top players' actual decks from their replay logs.

Each game's deck submission is the first 60-length action for that player, so
extraction is exact (no card-visibility inference needed). Confirms the player
runs one consistent list, names the cards, validates legality, and writes the
most-used deck to decks/<name>.csv.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from ptcg_ai.agent.deck import Deck, validate_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk

LOGS = Path("Logs/Higher ranking logs")


def player_index(data: dict, name: str) -> int | None:
    agents = data.get("info", {}).get("Agents", [])
    for i, a in enumerate(agents):
        if a.get("Name", "").lower() == name.lower():
            return i
    return None


def extract_deck(data: dict, idx: int) -> tuple[int, ...] | None:
    for step in data.get("steps", []):
        if idx < len(step):
            action = step[idx].get("action")
            if isinstance(action, list) and len(action) == 60 and all(isinstance(x, int) for x in action):
                return tuple(action)
    return None


def analyze_folder(folder_name: str, name: str, cards: CardDatabase) -> tuple[int, ...] | None:
    folder = LOGS / folder_name
    decks = collections.Counter()
    files = sorted(folder.glob("*.json"))
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f.name}: {e}")
            continue
        idx = player_index(data, name)
        if idx is None:
            continue
        deck = extract_deck(data, idx)
        if deck is not None:
            decks[tuple(sorted(deck))] += 1
    print(f"\n=== {name}: {len(files)} games, {len(decks)} distinct decklists ===")
    for deck, count in decks.most_common(3):
        print(f"  used {count}x")
    if not decks:
        return None
    best = decks.most_common(1)[0][0]
    print(f"\n  Most-used deck ({decks.most_common(1)[0][1]}/{sum(decks.values())} games):")
    counts = collections.Counter(best)
    for cid, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        info = cards.get_card(cid)
        nm = info.name if info else "??"
        typ = info.cardType.name if info else "?"
        print(f"    {n}x  [{cid:4d}] {nm}  ({typ})")
    problems = validate_deck(Deck(card_ids=best), cards)
    print(f"  legal? {'YES' if not problems else problems}")
    return best


def write_csv(deck: tuple[int, ...], path: Path) -> None:
    path.write_text("\n".join(str(c) for c in deck) + "\n", encoding="utf-8")
    print(f"  wrote {path} ({len(deck)} cards)")


def main():
    config = load_config(profile="benchmark")
    sdk = load_sdk(config.paths.sdk_dir)
    cards = CardDatabase.from_sdk(sdk)
    # Generalized: `python extract_top_decks.py "<folder>" <player> <out.csv>`
    # extracts one player's deck; with no args, reproduces the Vibechu/Majkel run.
    if len(sys.argv) >= 4:
        jobs = ((sys.argv[1], sys.argv[2], sys.argv[3]),)
    else:
        jobs = (
            ("Vibechu", "vibechu", "decks/vibechu.csv"),
            ("Majkel", "Majkel1337", "decks/majkel.csv"),
        )
    for folder, agent_name, out in jobs:
        deck = analyze_folder(folder, agent_name, cards)
        if deck is not None:
            problems = validate_deck(Deck(card_ids=deck), cards)
            if not problems:
                write_csv(deck, Path(out))
            else:
                print(f"  NOT writing {out}: {problems}")


if __name__ == "__main__":
    main()
