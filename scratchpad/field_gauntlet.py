"""Field gauntlet — the M8.1 promotion gate that fixes M8's evaluation flaw.

M8 shipped v2 because it beat v1 in ONE head-to-head matchup; the real ladder is
a diverse FIELD, and v2 lost ground there (non-transitivity). This harness pits a
candidate against EVERY distinct opponent deck seen across our own ladder replays
(greedy-v5, imitation-v1, imitation-v2 folders), each piloted by greedy, plus the
reference/meta decks. It reports per-deck win rate and — the decisive number — the
PAIRED per-deck delta between two candidates over the identical field, with a
bootstrap CI over decks.

Honest limitation: a greedy-piloted field is weaker than the human ladder; the
value here is DECK DIVERSITY (mill, stall, mirrors, meta) the single-matchup gate
never saw. Cross-check, don't treat as ground truth.

Usage:
  python scratchpad/field_gauntlet.py build          # extract + validate field, print summary
  python scratchpad/field_gauntlet.py run  <n>        # v1, v2.0, v2.1 over the field + paired deltas
"""

from __future__ import annotations

import collections
import dataclasses
import json
import sys
from pathlib import Path

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import Deck, load_deck, validate_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.decision.safety import SafePolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.runner import BattleRunner
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.evaluation.metrics import MatchStats
from ptcg_ai.imitation.policy import ImitationPolicy

#: M37: the field must be the meta we ACTUALLY face. The three ``Logs/Submission …``
#: folders below are from 2026-07-06..10, when we laddered at ~650-700 elo; measured
#: against ``imitation-fetch``'s real opponents (submission 55145833, ~870-890 elo) that
#: field carried 7.5% Grimmsnarl against a true 30.8% of episodes. Judging a candidate on
#: a meta that no longer exists is worse than judging it on fewer decks, so the fresh
#: replay folder is the field and the historical ones are kept only as a comment.
#:
#: CAVEAT the freshness fix does NOT solve: ``_extract_field_uncached`` DEDUPES, and the
#: macro WR is an unweighted mean over distinct decks. Those 30.8% of episodes collapse
#: into 8 of 69 distinct decks (11.6%), so the veto still under-weights popular
#: archetypes relative to the ladder. Read the per-archetype slices, not just the macro.
LADDER_FOLDERS = [
    "replays/55145833",          # imitation-fetch, the agent currently on the ladder
    # superseded (~650-700 elo era, 2026-07-06..10):
    #   "Logs/Submission greedy v5", "Logs/Submission imitation v1",
    #   "Logs/Submission imitation v2",
]
REFERENCE_DECKS = [
    "decks/meta_lucario.csv", "decks/meta_dragapult.csv",
    "decks/greengreenpurple.csv", "decks/lucario800.csv",
]
OUR_NAME = "Saulo Quiñones Góngora"

# Candidates: (label, kind, deck_csv, weights_or_None)
CANDIDATES = [
    ("v1", "imitation", "decks/greengreenpurple.csv", "data/models/bc_650_v1.json"),
    ("v1.2", "imitation", "decks/greengreenpurple.csv", "data/models/bc_650_v2.json"),
    # superseded/rejected (M8/M9) — re-enable only to re-audit; skipped for speed:
    # ("v2.0", "imitation", "decks/lucario800.csv", "data/models/bc_800_v1.json"),
    # ("v2.1", "imitation", "decks/lucario800.csv", "data/models/bc_800_v2.json"),
    # ("v3", "imitation", "decks/kenn2439.csv", "data/models/bc_940_v1.json"),
]

_SDK = None
_CARDS = None


def _cards():
    global _SDK, _CARDS
    if _SDK is None:
        _SDK = load_sdk(load_config(profile="benchmark").paths.sdk_dir)
        _CARDS = CardDatabase.from_sdk(_SDK)
    return _SDK, _CARDS


FIELD_CACHE = Path("data/field_cache.json")


def _field_signature() -> str:
    """Fingerprint of the replay folders + reference decks. Any added/changed replay
    or deck changes the signature, so the cache below self-invalidates."""
    parts = []
    for folder in LADDER_FOLDERS:
        p = Path(folder)
        files = sorted(p.glob("*.json")) if p.is_dir() else []
        newest = max((f.stat().st_mtime for f in files), default=0.0)
        parts.append(f"{folder}:{len(files)}:{newest:.0f}")
    for ref in REFERENCE_DECKS:
        p = Path(ref)
        parts.append(f"{ref}:{p.stat().st_mtime:.0f}" if p.is_file() else f"{ref}:-")
    return "|".join(parts)


def extract_field(cards: CardDatabase, use_cache: bool = True) -> list[tuple[str, tuple[int, ...]]]:
    """Distinct, legal opponent decks across our ladder replays + reference decks.

    Cached to ``data/field_cache.json`` keyed by :func:`_field_signature`: the field is
    IDENTICAL for every candidate, so re-parsing ~355 replay JSONs (plus deck validation)
    on every screen was pure repeated work. Pass ``use_cache=False`` to force a rebuild."""
    if use_cache:
        sig = _field_signature()
        if FIELD_CACHE.is_file():
            try:
                blob = json.loads(FIELD_CACHE.read_text(encoding="utf-8"))
                if blob.get("signature") == sig:
                    return [(name, tuple(d)) for name, d in blob["field"]]
            except Exception:
                pass  # corrupt/stale cache -> rebuild below
    field = _extract_field_uncached(cards)
    if use_cache:
        FIELD_CACHE.parent.mkdir(parents=True, exist_ok=True)
        FIELD_CACHE.write_text(json.dumps(
            {"signature": _field_signature(), "field": [[n, list(d)] for n, d in field]}),
            encoding="utf-8")
    return field


def _extract_field_uncached(cards: CardDatabase) -> list[tuple[str, tuple[int, ...]]]:
    seen: dict[tuple, str] = {}
    for folder in LADDER_FOLDERS:
        for f in sorted(Path(folder).glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            agents = data.get("info", {}).get("Agents", [])
            our = {i for i, a in enumerate(agents) if a.get("Name", "").lower() == OUR_NAME.lower()}
            for seat in range(len(agents)):
                if seat in our:
                    continue  # opponents only
                deck = _first_deck(data, seat)
                if deck is None:
                    continue
                key = tuple(sorted(deck))
                if key not in seen and not validate_deck(Deck(card_ids=deck), cards):
                    seen[key] = f"{Path(folder).name[:12]}/{f.stem}"
    field = [(name, tuple(sorted(k))) for k, name in seen.items()]
    for ref in REFERENCE_DECKS:
        p = Path(ref)
        if p.is_file():
            d = tuple(sorted(load_deck(p).card_ids))
            if d not in {k for _, k in field}:
                field.append((p.stem, d))
    return field


def _first_deck(data: dict, seat: int) -> tuple[int, ...] | None:
    for step in data.get("steps", []):
        if seat < len(step):
            a = step[seat].get("action")
            if isinstance(a, list) and len(a) == 60 and all(isinstance(x, int) for x in a):
                return tuple(a)
    return None


def _mk(kind, deck_ids, seed, weights):
    inner = ImitationPolicy(weights, deck=deck_ids) if kind == "imitation" else GreedyPolicy(deck=deck_ids)
    return SafePolicy(inner, deck=deck_ids, seed=seed)


def run_candidate(label, kind, deck_csv, weights, field, n, sdk, cards):
    """WR of one candidate (its own deck) vs each greedy-piloted field deck."""
    base = load_config(profile="benchmark")
    da = load_deck(Path(deck_csv))
    per_deck: dict[str, float] = {}
    interventions = 0
    for dname, dids in field:
        db = Deck(card_ids=dids)
        config = dataclasses.replace(base, paths=dataclasses.replace(
            base.paths, deck_path=Path(deck_csv), opponent_deck_path=Path(deck_csv)))
        pol_a = _mk(kind, da.as_list(), 1, weights)
        pol_b = SafePolicy(GreedyPolicy(deck=db.as_list()), deck=db.as_list(), seed=2)
        agent_a = PTCGAgent(pol_a, deck=da, cards=cards)
        agent_b = PTCGAgent(pol_b, deck=db, cards=cards)
        env = BattleEnvironment(config, sdk=sdk)
        runner = BattleRunner(env, max_decisions=config.battle.max_decisions)
        stats = MatchStats()
        for game in range(n):
            a_side = game % 2
            pol_a.on_battle_start(); pol_b.on_battle_start()
            if a_side == 0:
                rec = runner.run(agent_a, agent_b, da.as_list(), db.as_list())
            else:
                rec = runner.run(agent_b, agent_a, db.as_list(), da.as_list())
            stats.add(rec.outcome.winner, a_played_as=a_side)
        per_deck[dname] = stats.score_rate
        interventions += pol_a.interventions
    macro = sum(per_deck.values()) / max(len(per_deck), 1)
    micro_w = sum(per_deck.values())  # each deck same n -> macro==micro here
    print(f"  [{label}] macro WR over {len(per_deck)} decks = {macro:.3f}  (interventions={interventions})")
    return per_deck


def _paired_delta(a: dict, b: dict, seed=0):
    """Mean per-deck (a-b) with a bootstrap 90% CI over shared decks."""
    import random
    keys = [k for k in a if k in b]
    diffs = [a[k] - b[k] for k in keys]
    mean = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    boots = []
    for _ in range(2000):
        s = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boots.append(sum(s) / len(s))
    boots.sort()
    lo, hi = boots[int(0.05 * len(boots))], boots[int(0.95 * len(boots))]
    wins = sum(1 for d in diffs if d > 1e-9)
    losses = sum(1 for d in diffs if d < -1e-9)
    return mean, lo, hi, wins, losses, len(keys)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    sdk, cards = _cards()
    field = extract_field(cards)
    print(f"field: {len(field)} distinct legal decks "
          f"({len(field) - len(REFERENCE_DECKS)} from ladder + refs)")
    if mode == "build":
        arche = collections.Counter()
        for _, d in field:
            s = set(d)
            arche["Lucario(678)"] += 678 in s
            arche["Dragapult-ish(has 3+ trainers we lost to)"] += 0
        print(f"  Lucario-archetype decks in field: {arche['Lucario(678)']}")
        return 0

    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    results = {}
    for label, kind, deck_csv, weights in CANDIDATES:
        if weights and not Path(weights).is_file():
            print(f"  [{label}] SKIP: weights {weights} missing")
            continue
        results[label] = run_candidate(label, kind, deck_csv, weights, field, n, sdk, cards)

    print("\n=== paired per-deck deltas (positive = first candidate better) ===")
    if "v2.1" in results and "v1" in results:
        m, lo, hi, w, l, k = _paired_delta(results["v2.1"], results["v1"])
        print(f"v2.1 - v1 : mean {m:+.3f}  90%CI [{lo:+.3f},{hi:+.3f}]  decks better/worse={w}/{l} of {k}")
        print(f"  SHIP-vs-v1? {'YES' if (m >= 0 and lo > -0.02) else 'no'} (bar: mean>=0 AND lo>-0.02)")
    if "v2.1" in results and "v2.0" in results:
        m, lo, hi, w, l, k = _paired_delta(results["v2.1"], results["v2.0"])
        print(f"v2.1 - v2.0: mean {m:+.3f}  90%CI [{lo:+.3f},{hi:+.3f}]  decks better/worse={w}/{l} of {k}")
    if "v3" in results and "v1" in results:
        m, lo, hi, w, l, k = _paired_delta(results["v3"], results["v1"])
        print(f"v3   - v1 : mean {m:+.3f}  90%CI [{lo:+.3f},{hi:+.3f}]  decks better/worse={w}/{l} of {k}")
        print(f"  SHIP-vs-v1? {'YES' if (m >= 0 and lo > -0.02) else 'no'} (bar: mean>=0 AND lo>-0.02)")
    if "v1.2" in results and "v1" in results:
        m, lo, hi, w, l, k = _paired_delta(results["v1.2"], results["v1"])
        print(f"v1.2 - v1 : mean {m:+.3f}  90%CI [{lo:+.3f},{hi:+.3f}]  decks better/worse={w}/{l} of {k}")
        print(f"  SHIP-vs-v1? {'YES' if (m >= 0 and lo > -0.02) else 'no'} (bar: mean>=0 AND lo>-0.02)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
