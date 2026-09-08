"""One-shot cheap screen for a new clone candidate — no hand-authored DeckProfile.

Two cheap numbers, no arena/gauntlet, no manual card-reading:
  1. Deck strength: the deck piloted by GREEDY vs the same ~97-deck field used
     by field_gauntlet.py — isolates raw deck strength from clone quality.
  2. Clonability: trains a quick clone using an auto-built generic DeckProfile
     (deck_profiles.build_generic_profile — no per-card tuning) and reads its
     offline MAIN-context accuracy (BC vs greedy-as-predictor vs random).

Verdict thresholds (see docs/m9_findings.md's lesson): deck strength < 0.40 ->
discard without even training; MAIN accuracy < 0.75 -> discard. Only a candidate
clearing BOTH is worth: (a) a hand-authored DeckProfile (real damage_fn/wants_fn
for its actual cards, usually a small win over the generic numbers here), and
(b) the full two-stage promotion gate (arena + field_gauntlet.py).

SPEED (2026-07-20): a screen used to take ~1 h. Three changes, no threshold moved:
  * the 97-deck field is now cached to disk (field_gauntlet.extract_field) instead
    of re-parsing ~355 replay JSONs every run;
  * deck strength is screened PROGRESSIVELY — the field is played in shuffled
    chunks and we stop as soon as the 95% CI over decks clears or misses the 0.40
    bar. This is a THRESHOLD test, not a measurement, so clear pass/fail candidates
    resolve on a fraction of the field; only ambiguous ones pay the full 582 games;
  * --batch screens many candidates in ONE process, amortizing the SDK / card DB /
    field load across all of them.

Run (after downloading the candidate's replays into replays/<submission_id>/):
  uv run --group dev python scratchpad/quick_screen.py "replays/<submission_id>" "<player name>"

  ... --batch candidates.txt   # one "folder<TAB>player" (or "folder|player") per line
  ... --full-field             # disable early stopping (play all decks)
  ... --no-cache               # force a field rebuild
  ... --n 6                    # games per field deck
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

from ptcg_ai.imitation import deck_profiles as dp
from ptcg_ai.imitation.dataset import build_decision_dataset
from ptcg_ai.imitation.train import train_bc

import extract_top_decks as et
from field_gauntlet import _cards, extract_field, run_candidate

DECK_STRENGTH_MIN = 0.40
MAIN_ACC_MIN = 0.75
CHUNK = 12          # decks per progressive chunk
MIN_DECKS = 12      # never decide on fewer than this many decks
SCREEN_GAMES = 60   # games used for the clonability estimate (0 = full set)
BORDERLINE_MARGIN = 0.10  # below the bar by less than this on a subsample -> escalate


def _slug(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum()) or "candidate"


def _subsample_games(src: Path, n_games: int, slug: str) -> tuple[Path, int, int]:
    """Write a screening subset keeping only the first ``n_games`` distinct games.

    The clonability check is the expensive half of the screen, because cost scales with
    decision rows. A subsample is useful for TRIAGE — but only for triage.

    **It is NOT conservative.** This function was originally justified by "training on
    less data can only LOWER accuracy, so a subsampled pass is safe". That reasoning is
    wrong and the data falsified it (m21_findings.md): 懒惰的金枪鱼 scored **0.686 on 60
    games and 0.646 on all 461**. Subsampling shrinks the *validation* set too, making it
    narrower and easier, so the estimate is **optimistic**. A candidate landing just above
    the bar on a subsample would be called PROMISING in error. Always re-run a candidate
    you might act on with ``--screen-games 0``.
    Returns (path, kept_rows, kept_games)."""
    dst = src.with_name(f"{slug}_screen_sub.jsonl.gz")
    seen: dict[str, None] = {}
    kept = 0
    with gzip.open(src, "rt", encoding="utf-8") as fin, \
            gzip.open(dst, "wt", encoding="utf-8") as fout:
        for line in fin:
            try:
                gid = json.loads(line)["game_id"]
            except Exception:
                continue
            if gid not in seen:
                if len(seen) >= n_games:
                    continue
                seen[gid] = None
            fout.write(line)
            kept += 1
    return dst, kept, len(seen)


def _extract_deck(folder: str, player: str):
    """Most-frequent 60-card decklist for `player` in `folder` (None if none found)."""
    decks: dict[tuple[int, ...], int] = {}
    total = 0
    for f in sorted(Path(folder).glob("*.json")):
        if f.name == "metadata.json":
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        idx = et.player_index(data, player)
        if idx is None:
            continue
        total += 1
        deck = et.extract_deck(data, idx)
        if deck is not None:
            key = tuple(sorted(deck))
            decks[key] = decks.get(key, 0) + 1
    if not decks:
        return None, 0, total
    top_deck, top_count = max(decks.items(), key=lambda kv: kv[1])
    return top_deck, top_count, total


def _screen_strength(slug, deck_csv, field, sdk, cards, n=6, full=False, seed=0):
    """Progressive deck-strength screen against the greedy-piloted field.

    Plays the field in deterministically-shuffled chunks and stops as soon as the
    95% CI of the per-deck win rates excludes DECK_STRENGTH_MIN — i.e. as soon as
    the pass/fail answer is statistically settled. Returns (per_deck, macro_WR)."""
    order = list(field)
    random.Random(seed).shuffle(order)
    per_deck: dict[str, float] = {}
    i = 0
    while i < len(order):
        part = order[i:i + CHUNK]
        per_deck.update(run_candidate(
            f"greedy_{slug}[{min(i + len(part), len(order))}/{len(order)}]",
            "greedy", str(deck_csv), None, part, n, sdk, cards))
        i += len(part)
        vals = list(per_deck.values())
        mean = sum(vals) / len(vals)
        if full or i >= len(order) or len(vals) < MIN_DECKS:
            continue
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        se = sd / math.sqrt(len(vals))
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
        if lo > DECK_STRENGTH_MIN or hi < DECK_STRENGTH_MIN:
            print(f"  early stop after {len(vals)}/{len(order)} decks: "
                  f"{mean:.3f} [{lo:.3f},{hi:.3f}] vs bar {DECK_STRENGTH_MIN}")
            break
    return per_deck, sum(per_deck.values()) / max(len(per_deck), 1)


def screen_one(folder, player, field, sdk, cards, n=6, full=False,
               screen_games=SCREEN_GAMES) -> dict:
    """Screen a single candidate. Returns a small result dict (also printed)."""
    slug = _slug(player)
    res = {"player": player, "slug": slug, "folder": folder}
    print(f"\n{'='*70}\n=== SCREEN: {player!r}  ({folder}) ===")

    # 1. fixed decklist
    t0 = time.perf_counter()
    top_deck, top_count, total = _extract_deck(folder, player)
    if top_deck is None:
        print(f"no decklists found for {player!r} in {folder} — check the name/folder")
        res["verdict"] = "NO_DECK"
        return res
    print(f"deck: {top_count}/{total} games fixed ({top_count/max(total,1):.0%})"
          f"  [{time.perf_counter()-t0:.0f}s]")
    deck_csv = Path(f"decks/{slug}.csv")
    deck_csv.parent.mkdir(parents=True, exist_ok=True)
    deck_csv.write_text("\n".join(str(c) for c in top_deck) + "\n", encoding="utf-8")

    # 2. deck strength (progressive, early-stopping)
    t1 = time.perf_counter()
    per_deck, strength = _screen_strength(slug, deck_csv, field, sdk, cards, n=n, full=full)
    ok = strength >= DECK_STRENGTH_MIN
    res.update({"strength": round(strength, 4), "decks_played": len(per_deck)})
    print(f"\n[1/2] deck strength (greedy vs {len(per_deck)}/{len(field)} field decks): "
          f"{strength:.3f}  {'ok' if ok else f'FAIL (< {DECK_STRENGTH_MIN})'}"
          f"  [{time.perf_counter()-t1:.0f}s]")
    if not ok:
        print("\nVERDICT: DISCARD — deck too weak under greedy; skip the clonability check.")
        res["verdict"] = "DISCARD_DECK"
        return res

    # 3. clonability: quick clone with an auto-built generic profile
    t2 = time.perf_counter()
    dataset_path = Path(f"data/imitation/{slug}_screen.jsonl.gz")
    build_decision_dataset(Path(folder), player, dataset_path)
    print(f"  dataset built  [{time.perf_counter()-t2:.0f}s]")
    if screen_games:
        dataset_path, n_rows, n_games = _subsample_games(dataset_path, screen_games, slug)
        print(f"  screening subset: {n_rows} rows from {n_games} games "
              f"(--screen-games {screen_games}; 0 = full)")
    t3 = time.perf_counter()
    profile = dp.build_generic_profile(f"GENERIC_{slug.upper()}", top_deck, cards)
    dp.PROFILES[profile.name] = profile  # register in-process only; never shipped
    payload = train_bc(dataset_path, deck_csv, Path(f"data/models/{slug}_screen.json"), profile.name)
    print(f"  trained  [{time.perf_counter()-t3:.0f}s]")
    main_metrics = payload["metrics"].get("MAIN")
    if main_metrics is None:
        print("\n[2/2] MAIN context wasn't learned (too few rows, or greedy already "
              "near-perfect there) — inconclusive, read the printed table above by hand.")
        res["verdict"] = "INCONCLUSIVE"
        return res
    main_acc = main_metrics["bc_acc"]
    res["main_acc"] = round(main_acc, 4)
    sub = bool(screen_games)
    print(f"\n[2/2] clonability (offline MAIN accuracy, generic profile): {main_acc:.3f}"
          f"{'  [SUBSAMPLED -> OPTIMISTIC, see m21_findings.md]' if sub else '  [full data]'}")
    print()
    if main_acc >= MAIN_ACC_MIN:
        if sub:
            # m21: the subsample OVERstates accuracy (0.686@60 vs 0.646@461). A pass here
            # is a lead, not a verdict.
            print(f"VERDICT: PROMISING* ({main_acc:.3f} on a subsample that OVERSTATES it) — "
                  f"MUST be confirmed with --screen-games 0 before acting.")
            res["verdict"] = "PROMISING_UNCONFIRMED"
        else:
            print("VERDICT: PROMISING — worth a hand-authored DeckProfile + the full two-stage gate.")
            res["verdict"] = "PROMISING"
    elif sub and main_acc >= MAIN_ACC_MIN - BORDERLINE_MARGIN:
        print(f"VERDICT: BORDERLINE ({main_acc:.3f} vs bar {MAIN_ACC_MIN}; the subsample runs "
              f"OPTIMISTIC, so full data will likely be lower) — re-run with --screen-games 0.")
        res["verdict"] = "BORDERLINE"
    else:
        print(f"VERDICT: DISCARD (clonability {main_acc:.3f} < {MAIN_ACC_MIN}"
              f"{'; the subsample overstates, so full data would only be worse' if sub else ''}).")
        res["verdict"] = "DISCARD_CLONE"
    return res


def _load_batch(path: str) -> list[tuple[str, str]]:
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sep = "\t" if "\t" in line else ("|" if "|" in line else None)
        if sep is None:
            print(f"  skipping unparseable batch line (need TAB or |): {line!r}")
            continue
        folder, player = [p.strip() for p in line.split(sep, 1)]
        pairs.append((folder, player))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("folder", nargs="?")
    ap.add_argument("player", nargs="?")
    ap.add_argument("--batch", default=None, help="file with one 'folder<TAB>player' per line")
    ap.add_argument("--full-field", action="store_true", help="disable early stopping")
    ap.add_argument("--no-cache", action="store_true", help="force a field rebuild")
    ap.add_argument("--n", type=int, default=6, help="games per field deck")
    ap.add_argument("--screen-games", dest="screen_games", type=int, default=SCREEN_GAMES,
                    help="games used for the clonability estimate (0 = full set; the "
                         "subsample only understates accuracy, so a pass is safe)")
    args = ap.parse_args()

    if args.batch:
        pairs = _load_batch(args.batch)
    elif args.folder and args.player:
        pairs = [(args.folder, args.player)]
    else:
        print(__doc__)
        return 2

    # loaded ONCE for every candidate in this process (the expensive fixed costs)
    sdk, cards = _cards()
    t0 = time.perf_counter()
    field = extract_field(cards, use_cache=not args.no_cache)
    print(f"field: {len(field)} decks  [{time.perf_counter()-t0:.0f}s"
          f"{'' if args.no_cache else ', cached'}]")

    results = []
    for folder, player in pairs:
        try:
            results.append(screen_one(folder, player, field, sdk, cards,
                                      n=args.n, full=args.full_field,
                                      screen_games=args.screen_games))
        except Exception as exc:  # one bad candidate must not kill a batch
            print(f"  ERROR screening {player!r}: {exc}")
            results.append({"player": player, "folder": folder, "verdict": f"ERROR: {exc}"})

    if len(results) > 1:
        print(f"\n{'='*70}\nBATCH SUMMARY")
        print(f"{'player':<28}{'strength':>10}{'main_acc':>10}  verdict")
        for r in results:
            s = f"{r['strength']:.3f}" if "strength" in r else "-"
            m = f"{r['main_acc']:.3f}" if "main_acc" in r else "-"
            print(f"{r['player'][:27]:<28}{s:>10}{m:>10}  {r.get('verdict','?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
