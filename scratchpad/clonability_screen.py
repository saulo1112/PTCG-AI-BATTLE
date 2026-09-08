"""Clonability screen for candidate teacher players (M8-lesson checklist).

Scores each downloaded submission_id folder under replays/ against the
checklist in docs/handoff.md §7 — fixed-deck consistency, evolution-line
complexity, decision-context concentration (MAIN/TO_HAND), a greedy-accuracy
headroom proxy, gust-trainer usage rate, and elo level/stability (pulled fresh
from Kaggle's ListEpisodes, since replay JSONs carry no elo/timestamp data).

Read-only / analysis-only: no training, no deck-profile building, no shipping.

Run:  uv run python scratchpad/clonability_screen.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import extract_top_decks as et  # noqa: E402 - scratchpad sibling, see above
import kaggle_api  # noqa: E402 - tools/ sibling, path inserted above

from ptcg_ai.cards.database import CardDatabase  # noqa: E402
from ptcg_ai.config import load_config  # noqa: E402
from ptcg_ai.decision.greedy import GreedyPolicy  # noqa: E402
from ptcg_ai.environment.sdk import load_sdk  # noqa: E402
from ptcg_ai.imitation import kaggle_replay  # noqa: E402
from ptcg_ai.imitation.dataset import split_by_game  # noqa: E402
from ptcg_ai.imitation.train import _classify_contexts  # noqa: E402
from ptcg_ai.observation.parser import ObservationParser  # noqa: E402

REPLAYS_DIR = REPO_ROOT / "replays"
OWN_SUBMISSION_ID = 53922595  # the user's own current v1 submission, not a candidate
COOKIES_PATH = REPO_ROOT / "tools" / "cookies.txt.json"
_SWITCH, _TO_ACTIVE = 3, 4  # SelectContextKind values that pick an opponent's Pokémon


def _replay_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.json") if p.name != "metadata.json")


def discover_name(folder: Path) -> str | None:
    """The player name present in ~every game of this folder is the candidate."""
    counter: Counter[str] = Counter()
    for f in _replay_files(folder):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for a in data.get("info", {}).get("Agents", []):
            name = a.get("Name")
            if name:
                counter[name] += 1
    return counter.most_common(1)[0][0] if counter else None


def majority_deck(folder: Path, name: str) -> tuple[tuple[int, ...] | None, int, int]:
    """Most-used 60-card decklist + (games with that deck, total games found)."""
    decks: Counter[tuple[int, ...]] = Counter()
    total = 0
    for f in _replay_files(folder):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        idx = et.player_index(data, name)
        if idx is None:
            continue
        total += 1
        deck = et.extract_deck(data, idx)
        if deck is not None:
            decks[tuple(sorted(deck))] += 1
    if not decks:
        return None, 0, total
    top_deck, top_count = decks.most_common(1)[0]
    return top_deck, top_count, total


def deck_complexity(deck_ids: tuple[int, ...], cards: CardDatabase) -> dict:
    stage2, stage1, basics = [], [], []
    for cid in sorted(set(deck_ids)):
        info = cards.get_card(cid)
        if info is None or info.cardType.name != "POKEMON":
            continue
        if info.stage2:
            stage2.append(info.name)
        elif info.stage1:
            stage1.append(info.name)
        elif info.basic:
            basics.append(info.name)
    return {"stage2": stage2, "stage1": stage1, "basics": basics}


def gust_usage(rows: list, parser: ObservationParser) -> tuple[int, int]:
    """Fraction of SWITCH/TO_ACTIVE decisions whose chosen option targets the
    opponent's board (the runtime signature of a Boss's-Orders-style gust)."""
    hits, total = 0, 0
    for r in rows:
        if r.context not in (_SWITCH, _TO_ACTIVE):
            continue
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None:
            continue
        total += 1
        yidx = obs.current.yourIndex
        opts = obs.select.option
        if any(
            0 <= a < len(opts) and opts[a].playerIndex is not None and opts[a].playerIndex != yidx
            for a in r.action
        ):
            hits += 1
    return hits, total


def _parse_time(s: str) -> datetime:
    s = s.rstrip("Z")
    if "." in s:
        date_part, frac = s.split(".", 1)
        s = f"{date_part}.{(frac + '000000')[:6]}"
    return datetime.fromisoformat(s)


def elo_history(submission_id: int) -> dict | None:
    try:
        session = kaggle_api.load_session(COOKIES_PATH)
        episodes = kaggle_api.list_episodes(session, submission_id, successful_only=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  (elo history unavailable: {exc})")
        return None
    scores: list[float] = []
    times: list[datetime] = []
    for ep in episodes:
        for agent in ep.get("agents", []):
            if agent.get("submissionId") != submission_id:
                continue
            score = agent.get("updatedScore")
            if score is not None:
                scores.append(score)
            ct = ep.get("createTime")
            if ct:
                times.append(_parse_time(ct))
    if not scores:
        return None
    return {
        "n": len(scores),
        "mean": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
        "span_days": (max(times) - min(times)).total_seconds() / 86400 if times else None,
    }


def analyze_candidate(sid: int, cards: CardDatabase, parser: ObservationParser, greedy: GreedyPolicy) -> dict | None:
    folder = REPLAYS_DIR / str(sid)
    print(f"\n=== submission {sid} ===")
    name = discover_name(folder)
    if name is None:
        print("  could not detect player name, skipping")
        return None
    print(f"  player: {name!r}")

    deck, deck_count, n_games = majority_deck(folder, name)
    if deck is None:
        print("  no decklists extracted, skipping")
        return None
    fixed_ratio = deck_count / n_games if n_games else 0.0
    print(f"  fixed-deck ratio: {deck_count}/{n_games} = {fixed_ratio:.2f}")

    complexity = deck_complexity(deck, cards)
    print(f"  stage-2 evolutions: {complexity['stage2'] or 'none'}")
    print(f"  stage-1 evolutions: {complexity['stage1'] or 'none'}")

    rows = list(kaggle_replay.load_replay_dir(folder, name))
    train_rows, val_rows = split_by_game(rows, val_fraction=0.3, seed=0)
    cls = _classify_contexts(None, train_rows, val_rows, parser, cards, greedy)

    total_train = sum(d["n_train"] for d in cls.values())
    main_hand_train = sum(d["n_train"] for ctx, d in cls.items() if ctx in ("MAIN", "TO_HAND"))
    concentration = main_hand_train / total_train if total_train else 0.0
    total_val = sum(d["g_tot"] for d in cls.values())
    macro_greedy_acc = (
        sum(d["greedy_acc"] * d["g_tot"] for d in cls.values()) / total_val if total_val else 0.0
    )
    print(f"  decisions: {len(rows)}  (MAIN+TO_HAND concentration: {concentration:.1%})")
    print(f"  macro greedy-as-predictor accuracy: {macro_greedy_acc:.3f} (lower = more BC headroom)")

    gust_hits, gust_total = gust_usage(rows, parser)
    gust_rate = gust_hits / gust_total if gust_total else 0.0
    print(f"  opponent-targeting (gust) decisions: {gust_hits}/{gust_total} = {gust_rate:.1%}")

    elo = elo_history(sid)
    if elo:
        print(
            f"  elo: mean={elo['mean']:.1f} range=[{elo['min']:.1f},{elo['max']:.1f}] "
            f"over {elo['span_days']:.1f} days (n={elo['n']})"
        )

    return {
        "submission_id": sid,
        "name": name,
        "fixed_ratio": fixed_ratio,
        "has_stage2": bool(complexity["stage2"]),
        "n_stage1": len(complexity["stage1"]),
        "n_decisions": len(rows),
        "concentration": concentration,
        "macro_greedy_acc": macro_greedy_acc,
        "gust_rate": gust_rate,
        "elo": elo,
    }


def main() -> None:
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()
    greedy = GreedyPolicy(deck=None)

    candidate_ids = sorted(
        int(p.name)
        for p in REPLAYS_DIR.iterdir()
        if p.is_dir() and p.name.isdigit() and int(p.name) != OWN_SUBMISSION_ID
    )
    print(f"candidates found: {candidate_ids}")

    results = []
    for sid in candidate_ids:
        r = analyze_candidate(sid, cards, parser, greedy)
        if r is not None:
            results.append(r)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    header = f"{'submission':>11} {'name':<18} {'fixed%':>7} {'stage2':>7} {'MAIN+HAND%':>11} {'greedy_acc':>11} {'gust%':>7} {'elo_mean':>9}"
    print(header)
    for r in results:
        elo_mean = f"{r['elo']['mean']:.0f}" if r["elo"] else "n/a"
        print(
            f"{r['submission_id']:>11} {r['name']:<18} {r['fixed_ratio']*100:>6.1f}% "
            f"{'YES' if r['has_stage2'] else 'no':>7} {r['concentration']*100:>10.1f}% "
            f"{r['macro_greedy_acc']:>11.3f} {r['gust_rate']*100:>6.1f}% {elo_mean:>9}"
        )

    print("\nRECOMMENDATION")
    print("-" * 100)
    ranked = sorted(
        results,
        key=lambda r: (
            r["has_stage2"],       # False (0) sorts before True (1) -- prefer no stage-2
            -r["fixed_ratio"],
            r["gust_rate"],
            -r["concentration"],
        ),
    )
    if ranked:
        best = ranked[0]
        print(f"Most promising candidate: submission {best['submission_id']} ({best['name']})")
        print(
            f"  - fixed one deck in {best['fixed_ratio']:.1%} of games, "
            f"{'has a Stage-2 line (red flag)' if best['has_stage2'] else 'no Stage-2 evolution (Basic/Stage-1 only)'}"
        )
        print(f"  - {best['concentration']:.1%} of decisions concentrated in MAIN/TO_HAND")
        print(f"  - gust (opponent-targeting) decisions: {best['gust_rate']:.1%}")
        print(f"  - macro greedy-as-predictor accuracy: {best['macro_greedy_acc']:.3f}")
        if best["elo"]:
            print(
                f"  - elo mean {best['elo']['mean']:.1f}, range "
                f"[{best['elo']['min']:.1f}, {best['elo']['max']:.1f}] over {best['elo']['span_days']:.1f} days"
            )
    else:
        print("No candidate could be fully analyzed.")


if __name__ == "__main__":
    main()
