"""M48 Fase 2.3/2.4 -- search with the shipped champion + a critic proven calibrated
on REAL LADDER PLAY, not self-play.

CONTEXT. M18 closed determinized search because its leaf V was fit on TR-650 self-play
and mis-calibrated outside that distribution (won the mirror/Bellibolt, lost Lucario/
Cinderace). M48 F2.1 checked whether the SAME failure applies here: `v_alakazam.json`
(the RL critic M47 measured at AUC 0.6526 on SELF-PLAY states) scores **0.7550 AUC on
imitation-final's own 135 real ladder replays** -- because it was fit on Yushin's
2330-game corpus, which already spans the real archetype field, not a 3-opponent
self-play mix. Trying to broaden it with our own replay data made it slightly WORSE
(-0.0078): the premise "needs a broader critic" was wrong, the existing one already
qualifies. So this uses `v_alakazam.json` UNCHANGED as the search leaf V.

Subcommands:
  probe   10 games, wall-clock only -- is search affordable at all before any arena runs
  sweep   greedy_bias screen on the mirror, n=60 (cheap, not a ship decision)
  gate_a  the M43 bar: mirror n=600 + control, IC 95% entirely above 0.500
  gate_b  the M18 killer: n=600 vs Grimmsnarl AND vs Mega Lucario (held out of every
          self-play mix this project has ever run) -- M18 won the mirror by +0.100 and
          still lost the pooled field by -0.012. Passing gate_a alone proves nothing.

Run:
  PYTHONPATH=src python -u scratchpad/m48_search_arena.py probe
  PYTHONPATH=src python -u scratchpad/m48_search_arena.py sweep
  PYTHONPATH=src python -u scratchpad/m48_search_arena.py gate_a --gb 0.05 --n 600
  PYTHONPATH=src python -u scratchpad/m48_search_arena.py gate_b --gb 0.05 --n 600
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.safety import SafePolicy
from ptcg_ai.decision.search import SearchConfig
from ptcg_ai.decision.search_bc import ImitationSearchPolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.runner import BattleRunner
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.evaluation.metrics import MatchStats
from ptcg_ai.imitation.policy import ImitationPolicy

DECK = "decks/yushinito.csv"
CHAMPION = "data/models/bc_alakazam_final.json"
CRITIC = "data/models/v_alakazam.json"          # base-7, AUC 0.7550 on real ladder held-out
GRIMMSNARL = ("decks/luca.csv", "data/models/bc_luca_full.json")
LUCARIO = ("decks/lucario800.csv", "data/models/bc_800_v2.json")   # NOT in any self-play mix

_SDK = None
_CARDS = None


def _sdk_cards():
    global _SDK, _CARDS
    if _SDK is None:
        cfg = load_config(profile="benchmark")
        _SDK = load_sdk(cfg.paths.sdk_dir)
        _CARDS = CardDatabase.from_sdk(_SDK)
    return _SDK, _CARDS


#: Measured too expensive at the SearchConfig defaults (determinizations=4,
#: max_candidates=16): 74.3 s/game, comfortably under Kaggle's 600s/episode budget for
#: DEPLOYMENT but 8-15x our own ~0.5-1s/game imitation arenas, making a full n=600 gate
#: protocol (sweep + gate_a + gate_b, ~10 arena runs) a 12+ hour affair per arm -- not
#: affordable before the Friday deadline. Cut per the plan's own fallback.
CHEAP_SEARCH_KW = {"determinizations": 2, "max_candidates": 8, "max_candidate_sets": 4}


def _search_policy(gb, seed, search_cfg_kw=None):
    scfg = SearchConfig(greedy_bias=gb, **(search_cfg_kw if search_cfg_kw is not None
                                           else CHEAP_SEARCH_KW))
    sdk, cards = _sdk_cards()
    dids = load_deck(Path(DECK)).as_list()
    inner = ImitationSearchPolicy(CHAMPION, CRITIC, deck=dids, cards=cards, api=sdk.api,
                                  search_cfg=scfg, rng_seed=seed, bc_rollout=True)
    return SafePolicy(inner, deck=dids, seed=1)


def _arena_slice(label, is_search, gb, opp_deck, opp_weights, n, shard, n_shards):
    """One shard of games. Worker entry point for `_arena`'s multiprocessing pool.

    Rebuilds everything from scratch (no shared state across a process spawn), mirroring
    `rl_selfplay._collect_slice`'s pattern -- Windows spawns by re-import, so nothing built
    in the parent process (SDK, cards, policies) survives into a worker.
    """
    sdk, cards = _sdk_cards()
    base_cfg = load_config(profile="benchmark")
    config = dataclasses.replace(base_cfg, paths=dataclasses.replace(
        base_cfg.paths, deck_path=Path(DECK), opponent_deck_path=Path(opp_deck)))
    da = load_deck(Path(DECK)); dids = da.as_list()
    db = load_deck(Path(opp_deck)); odids = db.as_list()
    env = BattleEnvironment(config, sdk=sdk)
    runner = BattleRunner(env, max_decisions=config.battle.max_decisions)
    pol_a = _search_policy(gb, seed=1) if is_search else \
        SafePolicy(ImitationPolicy(CHAMPION, deck=dids), deck=dids, seed=1)
    pol_b = SafePolicy(ImitationPolicy(opp_weights, deck=odids), deck=odids, seed=2)
    wins = losses = draws = 0
    played = 0
    for game in range(n):
        if (game - shard) % n_shards != 0:
            continue
        played += 1
        a_side = game % 2
        pol_a.on_battle_start(); pol_b.on_battle_start()
        if a_side == 0:
            rec = runner.run(PTCGAgent(pol_a, deck=da, cards=cards),
                             PTCGAgent(pol_b, deck=db, cards=cards), dids, odids)
        else:
            rec = runner.run(PTCGAgent(pol_b, deck=db, cards=cards),
                             PTCGAgent(pol_a, deck=da, cards=cards), odids, dids)
        w = rec.outcome.winner
        if w is None:
            draws += 1
        elif w == a_side:
            wins += 1
        else:
            losses += 1
    return wins, losses, draws, played


def _arena(label, is_search, gb, opp_deck, opp_weights, n, workers=1):
    """One arena, sides swapped, optionally sharded across worker processes.

    `is_search`/`gb` (not a policy factory) so a worker can rebuild the policy itself --
    a `SearchPolicy` holds a native SDK handle, which is NOT picklable across a spawned
    process boundary.
    """
    t0 = time.perf_counter()
    if workers <= 1:
        wins, losses, draws, played = _arena_slice(label, is_search, gb, opp_deck,
                                                    opp_weights, n, 0, 1)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_arena_slice, label, is_search, gb, opp_deck, opp_weights,
                                n, i, workers) for i in range(workers)]
            parts = [f.result() for f in futs]
        wins = sum(p[0] for p in parts); losses = sum(p[1] for p in parts)
        draws = sum(p[2] for p in parts); played = sum(p[3] for p in parts)
        if played != n:
            raise RuntimeError(f"sharding lost games: played {played} of {n} requested")
    wall = time.perf_counter() - t0
    stats = MatchStats(wins=wins, losses=losses, draws=draws)
    lo, hi = stats.wilson_interval()
    print(f"\n=== {label} (n={n}, workers={workers}, swapped) ===")
    print(f"A: {stats.summary()}   clears0.5? {'YES' if lo > 0.5 else 'no'}")
    print(f"  wall={wall:.0f}s ({wall / n:.2f}s/game equivalent)")
    return stats.score_rate, lo, hi


def cmd_probe(n=10, gb=0.05, workers=1):
    """Wall-clock only: is search affordable against the 600s/episode budget?"""
    print(f"[probe] {n} games, greedy_bias={gb}, vs mirror (imitation-final)")
    _arena(f"PROBE search(gb={gb}) vs champion mirror", True, gb, DECK, CHAMPION, n,
          workers=workers)
    print(f"\nIf wall/game is not comfortably under Kaggle's ~600s/episode budget, cut "
          f"SearchConfig.determinizations/max_candidates further before any n=600 run.")


def cmd_sweep(n=60, workers=1):
    """Cheap screen, NOT a ship decision -- M18's lesson: the greedy_bias that wins the
    mirror can be the one that amplifies bad overrides out of distribution. Gate B picks
    the final value, not this."""
    print(f"[sweep] greedy_bias screen on the mirror, n={n}")
    for gb in (0.02, 0.05, 0.10):
        _arena(f"search(gb={gb}) vs champion mirror", True, gb, DECK, CHAMPION, n,
              workers=workers)


def cmd_gate_a(n=600, gb=0.05, workers=1):
    print(f"[gate_a] mirror n={n}, greedy_bias={gb}")
    s_score, s_lo, s_hi = _arena(f"CANDIDATE search(gb={gb}) vs CHAMPION (mirror)",
                                 True, gb, DECK, CHAMPION, n, workers=workers)
    c_score, c_lo, c_hi = _arena("CHAMPION vs CHAMPION control",
                                 False, gb, DECK, CHAMPION, n, workers=workers)
    print("\n" + "=" * 70)
    print(f"candidate vs champion (mirror)   {s_score:.3f}  [{s_lo:.3f}, {s_hi:.3f}]")
    print(f"champion vs champion (control)   {c_score:.3f}  [{c_lo:.3f}, {c_hi:.3f}]")
    if s_lo > 0.500:
        print(f"GATE A PASSES: CI entirely above 0.500.")
    else:
        print(f"GATE A FAILS: CI touches or crosses 0.500.")
    print("Check the control is ~0.48-0.52 before trusting the candidate row.")


def cmd_gate_b(n=600, gb=0.05, workers=1):
    """The M18 killer. A pass on gate_a alone is not a ship decision -- test generalization
    to Grimmsnarl (25% of the real field, the project's known wall) AND Mega Lucario
    (11.5% of the field, held out of EVERY self-play mix this project has ever run)."""
    print(f"[gate_b] field generalization n={n}, greedy_bias={gb}")
    rows = []
    for label, (odeck, ow) in (("grimmsnarl", GRIMMSNARL), ("lucario [HELD OUT]", LUCARIO)):
        s_score, s_lo, s_hi = _arena(f"CANDIDATE search(gb={gb}) vs {label}",
                                     True, gb, odeck, ow, n, workers=workers)
        c_score, c_lo, c_hi = _arena(f"CHAMPION vs {label}",
                                     False, gb, odeck, ow, n, workers=workers)
        rows.append((label, s_score, c_score, s_score - c_score))

    print("\n" + "=" * 70)
    print(f"{'opponent':<20}{'search':>9}{'champion':>10}{'delta':>9}")
    for label, s, c, d in rows:
        print(f"{label:<20}{s:>9.3f}{c:>10.3f}{d:>+9.3f}")
    pooled = sum(d for _, _, _, d in rows) / len(rows)
    print(f"\npooled delta: {pooled:+.3f}")
    if pooled >= 0.05:
        print("GATE B PASSES (>= +0.05 pooled).")
    else:
        print("GATE B FAILS. M18's exact failure mode: a mirror/single-matchup win that")
        print("does not generalize. Do NOT ship on gate_a alone.")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, n_default in (("probe", 10), ("sweep", 60), ("gate_a", 600), ("gate_b", 600)):
        p = sub.add_parser(name)
        p.add_argument("--n", type=int, default=n_default)
        p.add_argument("--gb", type=float, default=0.05)
        p.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    if args.cmd == "probe":
        cmd_probe(args.n, args.gb, args.workers)
    elif args.cmd == "sweep":
        cmd_sweep(args.n, args.workers)
    elif args.cmd == "gate_a":
        cmd_gate_a(args.n, args.gb, args.workers)
    elif args.cmd == "gate_b":
        cmd_gate_b(args.n, args.gb, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
