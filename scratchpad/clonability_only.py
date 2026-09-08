"""Clonability check WITHOUT the greedy-pilotability gate.

Why this file exists: `quick_screen.py` refuses to measure clonability unless the
deck first clears `DECK_STRENGTH_MIN = 0.40` piloted by the hand-coded GreedyPolicy.
That gate is anti-correlated with what we actually want. It rejected **Yushin Ito
at 0.250** (m23_findings.md:44, "DISCARD (deck)") — and M28 then cloned Yushin
anyway and produced the best agent the project has ever had (the current champion,
~905 elo). A deck that REQUIRES skilled piloting scores badly under a dumb
heuristic by construction, so the gate systematically discards the strongest decks.

This runs step [2/2] only: build the dataset, auto-derive a generic profile, train
the linear BC clone, report per-context held-out accuracy. Read-only apart from the
dataset/payload it writes under data/.

    uv run --group dev python scratchpad/clonability_only.py replays/55058847 Luca
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ptcg_ai.imitation.deck_profiles as dp
from ptcg_ai.imitation.dataset import build_decision_dataset
from ptcg_ai.imitation.train import train_bc
from quick_screen import _cards, _extract_deck, _slug


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("player")
    args = ap.parse_args()

    sdk, cards = _cards()
    slug = _slug(args.player)
    folder = Path(args.folder)

    top_deck, top_count, n_games = _extract_deck(str(folder), args.player)
    if not top_deck:
        print("no fixed deck found")
        return 2
    deck_csv = Path(f"decks/{slug}.csv")
    deck_csv.write_text("\n".join(str(c) for c in top_deck) + "\n", encoding="utf-8")
    print(f"deck: {n_games} games, {top_count / max(n_games, 1):.0%} on the modal 60 "
          f"-> {deck_csv}")

    t = time.perf_counter()
    ds = Path(f"data/imitation/{slug}_screen.jsonl.gz")
    summary = build_decision_dataset(folder, args.player, ds)
    print(f"dataset: {summary.n_rows} rows / {summary.n_games} games  [{time.perf_counter()-t:.0f}s]")

    profile = dp.build_generic_profile(f"GENERIC_{slug.upper()}", top_deck, cards)
    dp.PROFILES[profile.name] = profile
    print(f"generic profile dim={profile.feature_dim}\n")

    train_bc(ds, deck_csv, Path(f"data/models/{slug}_screen.json"), profile.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
