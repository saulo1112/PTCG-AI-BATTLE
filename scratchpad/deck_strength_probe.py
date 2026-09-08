"""Is a candidate DECK worth cloning? Probe raw deck strength vs the field.

v3 (Bellibolt) failed the gauntlet not because the clone was bad (its pilot-lift
over greedy was +0.30) but because the DECK underperforms v1's Team Rocket against
the field. Before investing in cloning another teacher, cheaply check whether their
deck is even competitive: pilot each candidate deck with GREEDY (same pilot for
all — isolates deck strength from clone quality) against the same greedy-piloted
~97-deck field, and compare macro WR to Team Rocket (v1's deck) as the baseline.

A deck only clears if greedy-piloting it beats greedy-piloting Team Rocket vs the
field — otherwise cloning it can't beat v1 (a good clone lifts ~+0.30 over greedy,
but so would a clone of Team Rocket).

Run:  uv run python scratchpad/deck_strength_probe.py [n]   # n games/deck, default 8
"""

from __future__ import annotations

import sys
from pathlib import Path

from field_gauntlet import _cards, extract_field, run_candidate

# (label, deck_csv) — all greedy-piloted. Team Rocket is the baseline to beat.
PROBE_DECKS = [
    ("greedy_TeamRocket(v1deck)", "decks/greengreenpurple.csv"),
    ("greedy_Bellibolt(v3deck)", "decks/kenn2439.csv"),
    ("greedy_Alakazam(uninc2000)", "decks/uninc2000.csv"),
    ("greedy_Cinderace(Yoshiki)", "decks/yoshiki.csv"),
]


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    sdk, cards = _cards()
    field = extract_field(cards)
    print(f"field: {len(field)} distinct legal decks")

    results = {}
    for label, deck_csv in PROBE_DECKS:
        if not Path(deck_csv).is_file():
            print(f"  [{label}] SKIP: {deck_csv} missing")
            continue
        per_deck = run_candidate(label, "greedy", deck_csv, None, field, n, sdk, cards)
        results[label] = sum(per_deck.values()) / max(len(per_deck), 1)

    print("\n=== raw deck strength (all greedy-piloted) vs field ===")
    baseline = results.get("greedy_TeamRocket(v1deck)")
    for label, macro in sorted(results.items(), key=lambda kv: -kv[1]):
        delta = f"  (vs TeamRocket: {macro - baseline:+.3f})" if baseline is not None else ""
        verdict = ""
        if baseline is not None and label != "greedy_TeamRocket(v1deck)":
            verdict = "  WORTH CLONING" if macro > baseline else "  weaker than v1's deck"
        print(f"  {label:<32} macro WR {macro:.3f}{delta}{verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
