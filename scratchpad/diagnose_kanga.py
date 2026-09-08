"""M26 diagnosis of Kanga's real ladder play + clone-vs-teacher behavioral gap.

Kanga (clone of 懒惰的金枪鱼 #32/1052, profile KANGASKHAN_1052) has been stable at
760-770 on the ladder for >1 day. This is the M10-Phase-0 diagnosis (never run on
Kanga) plus a clone-vs-teacher per-turn comparison the user asked for.

Adapts scratchpad/diagnose_v1.py (which hard-codes TR_650 and one log dir) to:
  * two live replay folders (54911514 = kanga-new, 54893233 = the prior upload),
  * the KANGASKHAN_1052 profile / bc_kangaskhan_1052.json weights,
  * the teacher's own dataset (data/imitation/懒惰的金枪鱼_screen.jsonl.gz, 461 games)
    for a lethal-audit instrument sanity AND the per-turn behavioral comparison.

Reuses: kaggle_replay.iter_player_decisions / player_seats, dataset.read_decision_dataset,
ImitationPolicy (parity + interventions), the profile's damage_fn (lethal), extract_top_decks.

Everything is READ-ONLY. Run:
  uv run --group dev python scratchpad/diagnose_kanga.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.imitation.kaggle_replay import ReplayDecision, iter_player_decisions, player_seats
from ptcg_ai.imitation.policy import ImitationPolicy
from ptcg_ai.observation.models import OptionKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

LIVE_FOLDERS = [Path("replays/54911514"), Path("replays/54893233")]
TEACHER_DS = Path("data/imitation/懒惰的金枪鱼_screen.jsonl.gz")
OUR = "Saulo Quiñones Góngora"
DECK_CSV = Path("decks/懒惰的金枪鱼.csv")
WEIGHTS = "data/models/bc_kangaskhan_1052.json"
PROFILE = get_profile("KANGASKHAN_1052")

# opponent archetype markers (same key cards as diagnose_v1)
ARCHETYPE_MARKERS = {
    678: "Mega Lucario", 675: "Lunatone/Solrock", 269: "Bellibolt",
    190: "Archaludon", 743: "Alakazam", 400: "Team Rocket", 666: "Cinderace",
    24: "TR Kangaskhan", 431: "TR Mewtwo",
}

import extract_top_decks as et  # noqa: E402


def _files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.json") if p.name != "metadata.json")


# ---- Fase 0: record / loss reasons / archetypes (live replays only) ---------

def loss_reason(data: dict, our_seat: int) -> tuple[str, int]:
    steps = data.get("steps", [])
    winner_seat = 1 - our_seat
    for step in reversed(steps):
        if winner_seat >= len(step):
            continue
        cur = (step[winner_seat].get("observation") or {}).get("current")
        if not cur or "players" not in cur:
            continue
        yidx = cur.get("yourIndex", winner_seat)
        us = cur["players"][1 - yidx] if len(cur["players"]) > 1 else None
        win = cur["players"][yidx]
        if us is None:
            break
        our_prizes = len(us.get("prize", []))
        our_deck = us.get("deckCount", 99)
        our_active = sum(1 for a in (us.get("active") or []) if a)
        our_bench = len(us.get("bench", []))
        if our_deck <= 0:
            return "deck-out", our_prizes
        if our_active == 0 and our_bench == 0:
            return "bench-out", our_prizes
        return "prize-race", our_prizes
    return "unknown", 6


def record(folder: Path) -> dict:
    wins = losses = selfmatch = 0
    loss_reasons: collections.Counter = collections.Counter()
    prize_margin: collections.Counter = collections.Counter()
    loss_arch: collections.Counter = collections.Counter()
    opp_rating_on_loss: list[float] = []
    opp_rating_on_win: list[float] = []
    for f in _files(folder):
        data = json.loads(f.read_text(encoding="utf-8"))
        seats = player_seats(data, OUR)
        if len(seats) != 1:
            selfmatch += 1
            continue
        our = seats[0]
        rew = (data.get("rewards") or [0, 0])[our]
        # opponent live rating if present in info.Agents
        agents = data.get("info", {}).get("Agents", [])
        opp_rating = None
        if len(agents) == 2:
            opp = agents[1 - our]
            for k in ("Rating", "rating", "score", "Score"):
                if isinstance(opp.get(k), (int, float)):
                    opp_rating = float(opp[k]); break
        if rew > 0:
            wins += 1
            if opp_rating is not None:
                opp_rating_on_win.append(opp_rating)
            continue
        losses += 1
        if opp_rating is not None:
            opp_rating_on_loss.append(opp_rating)
        reason, our_prizes = loss_reason(data, our)
        loss_reasons[reason] += 1
        prize_margin[our_prizes] += 1
        deck = et.extract_deck(data, 1 - our)
        if deck:
            labels = [lab for cid, lab in ARCHETYPE_MARKERS.items() if cid in deck]
            loss_arch[labels[0] if labels else "other"] += 1
    return {
        "wins": wins, "losses": losses, "selfmatch": selfmatch,
        "loss_reasons": loss_reasons, "prize_margin": prize_margin,
        "loss_arch": loss_arch,
        "opp_rating_on_loss": opp_rating_on_loss, "opp_rating_on_win": opp_rating_on_win,
    }


# ---- decision-stream helpers (work for live replays AND teacher dataset) -----

def live_decisions():
    for folder in LIVE_FOLDERS:
        for f in _files(folder):
            yield from iter_player_decisions(f, OUR)


def teacher_decisions():
    yield from read_decision_dataset(TEACHER_DS)


def lethal_audit(decisions, parser, cards) -> dict:
    """Turn-level lethal miss with the profile's damage_fn (M10/M8.1 metric)."""
    turns: dict[tuple[str, int], list[bool]] = collections.defaultdict(lambda: [False, False])
    turn_won: dict[tuple[str, int], bool] = {}
    for dec in decisions:
        if SelectContextKind(dec.context) is not SelectContextKind.MAIN:
            continue
        obs = parser.parse(dec.raw_observation)
        if obs.select is None or obs.current is None:
            continue
        gs = GameState.build(obs, cards)
        if gs.opp_active is None or gs.opp_active.hp <= 0:
            continue
        key = (dec.game_id, obs.current.turn)
        turn_won[key] = dec.won
        lethal_idx = [
            i for i, opt in enumerate(obs.select.option)
            if opt.type is OptionKind.ATTACK
            and PROFILE.damage_fn(opt.attackId, gs, cards) >= gs.opp_active.hp
        ]
        if lethal_idx:
            turns[key][0] = True
            if any(a in lethal_idx for a in dec.action):
                turns[key][1] = True
    avail = [k for k, (a, _) in turns.items() if a]
    taken = [k for k in avail if turns[k][1]]
    missed = [k for k in avail if not turns[k][1]]
    losing_miss = [k for k in missed if not turn_won.get(k, True)]
    return {
        "turns": len(turns), "avail": len(avail), "taken": len(taken),
        "missed": len(missed), "losing_miss_turns": len(losing_miss),
        "losing_miss_games": len({g for g, _ in losing_miss}),
    }


def parity_and_routing(parser, cards) -> dict:
    """Re-score live MAIN-family decisions with the shipped bundle: does the live
    action match the bundle's pick? Also census which contexts route to the learned
    scorer vs greedy, split by won/lost game."""
    deck = [int(x) for x in DECK_CSV.read_text(encoding="utf-8").split()]
    pol = ImitationPolicy(WEIGHTS, deck=deck)
    learned = set(pol._weights.keys())
    parity_ok = parity_tot = 0
    ctx_routed: collections.Counter = collections.Counter()  # learned vs greedy
    ctx_by_outcome = {"win": collections.Counter(), "loss": collections.Counter()}
    for dec in live_decisions():
        ctxname = SelectContextKind(dec.context).name
        routed = "learned" if ctxname in learned else "greedy"
        ctx_routed[routed] += 1
        ctx_by_outcome["win" if dec.won else "loss"][ctxname] += 1
        if ctxname not in learned:
            continue
        obs = parser.parse(dec.raw_observation)
        if obs.select is None or obs.current is None or len(obs.select.option) < 2:
            continue
        ctx = DecisionContext(raw=dec.raw_observation, observation=obs, cards=cards)
        try:
            pick = set(pol.choose(ctx))
        except Exception:
            continue
        parity_tot += 1
        if pick == set(dec.action):
            parity_ok += 1
    return {
        "parity_ok": parity_ok, "parity_tot": parity_tot,
        "bc_used": pol._bc_used, "bc_failures": pol._bc_failures,
        "ctx_routed": ctx_routed, "ctx_by_outcome": ctx_by_outcome,
    }


def behavioral(decisions, parser, cards) -> dict:
    """Per-turn aggregates (the M20-correct unit) for a clone-vs-teacher comparison."""
    games: set[str] = set()
    attacks_per_game: collections.Counter = collections.Counter()
    first_attack_turn: dict[str, int] = {}
    main_turns: set[tuple[str, int]] = set()
    attack_turns: set[tuple[str, int]] = set()
    bench_max: dict[str, int] = {}          # deepest bench reached per game
    insurance_turns: set[tuple[str, int]] = set()  # MAIN turns with a charged bencher
    for dec in decisions:
        games.add(dec.game_id)
        ctxkind = SelectContextKind(dec.context)
        if ctxkind is not SelectContextKind.MAIN:
            continue
        obs = parser.parse(dec.raw_observation)
        if obs.select is None or obs.current is None:
            continue
        turn = obs.current.turn
        main_turns.add((dec.game_id, turn))
        gs = GameState.build(obs, cards)
        bench_max[dec.game_id] = max(bench_max.get(dec.game_id, 0), gs.my_bench_count)
        if gs.has_bench_insurance:
            insurance_turns.add((dec.game_id, turn))
        chose_attack = any(
            obs.select.option[a].type is OptionKind.ATTACK
            for a in dec.action if 0 <= a < len(obs.select.option)
        )
        if chose_attack:
            attacks_per_game[dec.game_id] += 1
            attack_turns.add((dec.game_id, turn))
            if dec.game_id not in first_attack_turn:
                first_attack_turn[dec.game_id] = turn
    ng = max(len(games), 1)
    fa = sorted(first_attack_turn.values())
    return {
        "games": len(games),
        "attacks_per_game": sum(attacks_per_game.values()) / ng,
        "first_attack_turn_median": (fa[len(fa) // 2] if fa else None),
        "games_that_ever_attacked": len(first_attack_turn),
        "main_turns_per_game": len(main_turns) / ng,
        "attack_turn_frac": len(attack_turns) / max(len(main_turns), 1),
        "bench_max_avg": sum(bench_max.values()) / ng,
        "bench_insurance_turn_frac": len(insurance_turns) / max(len(main_turns), 1),
    }


def _pct(a, b):
    return f"{a}/{b} = {a/max(b,1):.1%}"


def main() -> int:
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()

    print("=" * 72)
    print("FASE 0 — record per live folder")
    tot_w = tot_l = 0
    for folder in LIVE_FOLDERS:
        r = record(folder)
        tot_w += r["wins"]; tot_l += r["losses"]
        print(f"\n[{folder}]  {r['wins']}W-{r['losses']}L  (+{r['selfmatch']} self)")
        orl, orw = r["opp_rating_on_loss"], r["opp_rating_on_win"]
        if orl or orw:
            aw = sum(orw) / len(orw) if orw else float("nan")
            al = sum(orl) / len(orl) if orl else float("nan")
            print(f"  opp live rating: avg on WIN {aw:.0f} (n={len(orw)}) | on LOSS {al:.0f} (n={len(orl)})")
        print(f"  loss reasons: {dict(r['loss_reasons'])}")
        print(f"  prize margin (our prizes still needed at loss): {dict(sorted(r['prize_margin'].items()))}")
        print(f"  loss archetypes: {dict(r['loss_arch'])}")
    print(f"\nPOOLED live: {tot_w}W-{tot_l}L  (WR {tot_w/max(tot_w+tot_l,1):.1%})")

    print("\n" + "=" * 72)
    print("FASE 1 — bug-class leak audit")
    print("\n[1.1] lethal audit — KANGA LIVE (turn-level, KANGASKHAN_1052.damage_fn)")
    la = lethal_audit(live_decisions(), parser, cards)
    print(f"  turns w/ KO available: {la['avail']} of {la['turns']} MAIN turns")
    if la["avail"]:
        print(f"  KO taken:  {_pct(la['taken'], la['avail'])}")
        print(f"  KO missed: {_pct(la['missed'], la['avail'])}   "
              f"(losing-game misses: {la['losing_miss_turns']} in {la['losing_miss_games']} games)")

    print("\n[1.1-sanity] lethal audit — TEACHER dataset (instrument check, expect ~perfect)")
    lat = lethal_audit(teacher_decisions(), parser, cards)
    print(f"  turns w/ KO available: {lat['avail']} of {lat['turns']}")
    if lat["avail"]:
        print(f"  KO taken:  {_pct(lat['taken'], lat['avail'])}")
        print(f"  KO missed: {_pct(lat['missed'], lat['avail'])}")

    print("\n[1.2] bundle parity + routing (live)")
    pr = parity_and_routing(parser, cards)
    print(f"  MAIN-family parity (live action == bundle pick): {_pct(pr['parity_ok'], pr['parity_tot'])}")
    print(f"  bc_used={pr['bc_used']}  bc_failures(SafePolicy-style)={pr['bc_failures']}")
    print(f"  routing: {dict(pr['ctx_routed'])}")
    print(f"  contexts in WON games:  {dict(pr['ctx_by_outcome']['win'].most_common(8))}")
    print(f"  contexts in LOST games: {dict(pr['ctx_by_outcome']['loss'].most_common(8))}")

    print("\n" + "=" * 72)
    print("FASE 2 — clone(live) vs teacher(dataset) per-turn behavior")
    bl = behavioral(live_decisions(), parser, cards)
    bt = behavioral(teacher_decisions(), parser, cards)
    print(f"{'axis':<26}{'kanga-live':>14}{'teacher':>14}")
    for k in ("games", "attacks_per_game", "first_attack_turn_median",
              "main_turns_per_game", "attack_turn_frac", "games_that_ever_attacked",
              "bench_max_avg", "bench_insurance_turn_frac"):
        vl, vt = bl[k], bt[k]
        fl = f"{vl:.3f}" if isinstance(vl, float) else str(vl)
        ft = f"{vt:.3f}" if isinstance(vt, float) else str(vt)
        print(f"{k:<26}{fl:>14}{ft:>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
