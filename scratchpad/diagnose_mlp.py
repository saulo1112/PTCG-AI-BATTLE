"""M32 diagnosis of the CHAMPION `imitation-mlp`'s real ladder play (replays/55011997).

`imitation-mlp` = MLP clone of Yushin Ito (#1, Stage-2 Alakazam combo, profile
ALAKAZAM dim 658, weights bc_alakazam_mlp.json). It leads the ladder (889 vs a
concurrent kanga 845.5) but we have never looked at its OWN losses. This is the
M26/M10-Phase-0 diagnosis re-pointed at it, plus two combo-specific phases the
Kanga/nsr versions did not need.

Adapts scratchpad/diagnose_kanga.py to:
  * replays/55011997 (100 downloaded episodes),
  * the ALAKAZAM profile / bc_alakazam_mlp.json (MLP ensemble scorer),
  * the teacher's own dataset (data/imitation/yushinito_screen.jsonl.gz),
  * FASE 0b: per-archetype W-L (M29 showed the field mix, not the agent, moves elo),
  * FASE 3: combo assembly — does the clone actually ASSEMBLE Alakazam, and when?

Everything is READ-ONLY. Run:
  uv run --group dev python scratchpad/diagnose_mlp.py
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.imitation.kaggle_replay import iter_player_decisions, player_seats
from ptcg_ai.imitation.policy import ImitationPolicy
from ptcg_ai.observation.models import OptionKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

# M43: overridable from the environment so this can be re-pointed at a NEW
# submission's replays without editing (and thereby invalidating) the M32 record.
# Defaults are exactly M32's constants, so a bare run reproduces that milestone.
#   PTCG_LIVE=replays/55369569 PTCG_WEIGHTS=data/models/bc_alakazam_ctx.json \
#       python scratchpad/diagnose_mlp.py
LIVE_FOLDERS = [Path(p) for p in os.environ.get("PTCG_LIVE", "replays/55011997").split(",")]
TEACHER_DS = Path("data/imitation/yushinito_screen.jsonl.gz")
OUR = "Saulo Quiñones Góngora"
DECK_CSV = Path("decks/yushinito.csv")
WEIGHTS = os.environ.get("PTCG_WEIGHTS", "data/models/bc_alakazam_mlp.json")
PROFILE = get_profile("ALAKAZAM")

# combo pieces (see deck_profiles ALAKAZAM section)
ABRA, KADABRA, ALAKAZAM = 741, 742, 743
POWERFUL_HAND, CRUEL_ARROW = 1072, 183

# opponent archetype markers. ORDER MATTERS: _archetype returns the FIRST match,
# so a card that also appears as a support piece in another archetype must sit
# AFTER the deck it would otherwise steal (e.g. 104 Froslass is a 2-of in the
# Grimmsnarl list, so 648 must come first).
#
# M35 CORRECTIONS — two bugs found by cross-checking against the community meta
# analysis (74,634 games) and verifying every id against the card DB:
#   * 1191 was labelled "Cynthia's Garchomp". It is **Kieran, a SUPPORTER card**.
#     Every "Cynthia's Garchomp" number in M32/M33/M34 is therefore MISLABELLED —
#     it counted any deck running Kieran. Removed.
#   * The real Garchomp line (380 Cynthia's Gabite / 381 Cynthia's Garchomp ex)
#     had NO marker, so it always fell into "other". It matters: the community
#     matchup matrix has Garchomp beating Grimmsnarl 59% at only 7% share, i.e.
#     the most likely next meta shift.
#   * 400 is specifically **Team Rocket's Tarountula** (+401 Spidops), not a
#     generic "Team Rocket" deck. This retroactively confirms M33: we measured
#     Yushin at 26.2% against it (n=126) and the community measured alakazam at
#     28% vs tarountula (n=1038) — two independent reads of the same matchup,
#     and it is a dedicated Alakazam hunter (72% the other way).
# Deliberately NOT added: the Dudunsparce ids (66/306/997) — 66 is a 2-of in
# Yushin's own Alakazam list, so it would misclassify our own mirror.
ARCHETYPE_MARKERS = {
    648: "Marnie's Grimmsnarl",          # before 104: Froslass is a 2-of here
    381: "Cynthia's Garchomp", 380: "Cynthia's Garchomp",
    400: "TR Tarountula", 401: "TR Tarountula",
    743: "Alakazam", 245: "Alakazam",
    756: "Mega Kangaskhan", 24: "TR Kangaskhan",
    121: "Dragapult ex",
    1031: "Mega Starmie", 361: "Misty's Starmie",
    190: "Archaludon", 170: "Archaludon", 840: "Archaludon",
    678: "Mega Lucario", 269: "Bellibolt",
    345: "Crustle", 533: "Crustle",
    879: "Trevenant", 723: "Mega Abomasnow", 419: "Abomasnow",
    747: "Mega Gardevoir", 431: "TR Mewtwo", 666: "Cinderace",
    104: "Froslass", 675: "Lunatone/Solrock",
}

import extract_top_decks as et  # noqa: E402


def _files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.json") if p.name != "metadata.json")


def _archetype(deck: list[int] | None) -> str:
    if not deck:
        return "unknown"
    labels = [lab for cid, lab in ARCHETYPE_MARKERS.items() if cid in deck]
    return labels[0] if labels else "other"


# ---- Fase 0: record / loss reasons / archetypes (live replays only) ---------

def loss_reason(data: dict, our_seat: int) -> tuple[str, int]:
    """Reconstruct WHY we lost from the winner's last observation of us."""
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
    by_arch: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])  # [W, L]
    loss_reason_by_arch: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    game_len: dict[str, list[int]] = {"win": [], "loss": []}
    for f in _files(folder):
        data = json.loads(f.read_text(encoding="utf-8"))
        seats = player_seats(data, OUR)
        if len(seats) != 1:
            selfmatch += 1
            continue
        our = seats[0]
        rew = (data.get("rewards") or [0, 0])[our]
        arch = _archetype(et.extract_deck(data, 1 - our))
        nsteps = len(data.get("steps", []))
        if rew > 0:
            wins += 1
            by_arch[arch][0] += 1
            game_len["win"].append(nsteps)
            continue
        losses += 1
        by_arch[arch][1] += 1
        game_len["loss"].append(nsteps)
        reason, our_prizes = loss_reason(data, our)
        loss_reasons[reason] += 1
        prize_margin[our_prizes] += 1
        loss_reason_by_arch[arch][reason] += 1
    return {
        "wins": wins, "losses": losses, "selfmatch": selfmatch,
        "loss_reasons": loss_reasons, "prize_margin": prize_margin,
        "by_arch": by_arch, "loss_reason_by_arch": loss_reason_by_arch,
        "game_len": game_len,
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
    parity_by_outcome = {"win": [0, 0], "loss": [0, 0]}  # [ok, tot]
    ctx_routed: collections.Counter = collections.Counter()
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
        bucket = parity_by_outcome["win" if dec.won else "loss"]
        bucket[1] += 1
        if pick == set(dec.action):
            parity_ok += 1
            bucket[0] += 1
    return {
        "parity_ok": parity_ok, "parity_tot": parity_tot,
        "parity_by_outcome": parity_by_outcome,
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
    bench_max: dict[str, int] = {}
    insurance_turns: set[tuple[str, int]] = set()
    hand_sizes: list[int] = []
    for dec in decisions:
        games.add(dec.game_id)
        if SelectContextKind(dec.context) is not SelectContextKind.MAIN:
            continue
        obs = parser.parse(dec.raw_observation)
        if obs.select is None or obs.current is None:
            continue
        turn = obs.current.turn
        main_turns.add((dec.game_id, turn))
        gs = GameState.build(obs, cards)
        hand_sizes.append(gs.hand_size)
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
        "hand_size_avg": sum(hand_sizes) / max(len(hand_sizes), 1),
    }


# ---- Fase 3: combo assembly (Alakazam-specific) ------------------------------

def combo_audit(decisions, parser, cards, label: str) -> dict:
    """Does the pilot actually ASSEMBLE the Stage-2 combo, and when?

    Per game we track the earliest turn each line piece is seen in play, whether
    Powerful Hand was ever used, and the hand_size at each Powerful Hand attack
    (the damage dial: 20 x hand_size).
    """
    games: set[str] = set()
    won: dict[str, bool] = {}
    first_turn: dict[int, dict[str, int]] = {ABRA: {}, KADABRA: {}, ALAKAZAM: {}}
    ph_games: set[str] = set()
    ph_hand: list[int] = []
    ph_turn: dict[str, int] = {}
    for dec in decisions:
        if SelectContextKind(dec.context) is not SelectContextKind.MAIN:
            continue
        obs = parser.parse(dec.raw_observation)
        if obs.select is None or obs.current is None:
            continue
        g, turn = dec.game_id, obs.current.turn
        games.add(g)
        won[g] = dec.won
        me = obs.current.players[obs.current.yourIndex] if obs.current.players else None
        if me is not None:
            in_play = [p for p in me.active if p is not None] + list(me.bench)
            for p in in_play:
                d = first_turn.get(p.id)
                if d is not None and g not in d:
                    d[g] = turn
        gs = GameState.build(obs, cards)
        for a in dec.action:
            if not (0 <= a < len(obs.select.option)):
                continue
            opt = obs.select.option[a]
            if opt.type is OptionKind.ATTACK and opt.attackId == POWERFUL_HAND:
                ph_games.add(g)
                ph_hand.append(gs.hand_size)
                ph_turn.setdefault(g, turn)
    def _split(d: dict[str, int]) -> dict:
        w = [v for g, v in d.items() if won.get(g)]
        l = [v for g, v in d.items() if not won.get(g)]
        med = lambda xs: (sorted(xs)[len(xs) // 2] if xs else None)  # noqa: E731
        return {"games": len(d), "median": med(list(d.values())),
                "median_win": med(w), "median_loss": med(l)}
    ng = max(len(games), 1)
    nw = max(sum(1 for g in games if won.get(g)), 1)
    nl = max(sum(1 for g in games if not won.get(g)), 1)
    return {
        "label": label, "games": len(games),
        "wins": sum(1 for g in games if won.get(g)),
        "abra": _split(first_turn[ABRA]),
        "kadabra": _split(first_turn[KADABRA]),
        "alakazam": _split(first_turn[ALAKAZAM]),
        "alakazam_frac": len(first_turn[ALAKAZAM]) / ng,
        "alakazam_frac_win": sum(1 for g in first_turn[ALAKAZAM] if won.get(g)) / nw,
        "alakazam_frac_loss": sum(1 for g in first_turn[ALAKAZAM] if not won.get(g)) / nl,
        "ph_frac": len(ph_games) / ng,
        "ph_frac_win": sum(1 for g in ph_games if won.get(g)) / nw,
        "ph_frac_loss": sum(1 for g in ph_games if not won.get(g)) / nl,
        "ph_attacks": len(ph_hand),
        "ph_hand_avg": sum(ph_hand) / max(len(ph_hand), 1),
        "ph_first_turn": _split(ph_turn),
    }


def _pct(a, b):
    return f"{a}/{b} = {a/max(b,1):.1%}"


def main() -> int:
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()

    print("=" * 78)
    print("FASE 0 — record on the live ladder (imitation-mlp, champion)")
    tot_w = tot_l = 0
    for folder in LIVE_FOLDERS:
        r = record(folder)
        tot_w += r["wins"]; tot_l += r["losses"]
        print(f"\n[{folder}]  {r['wins']}W-{r['losses']}L  (+{r['selfmatch']} self-match, excluded)")
        print(f"  loss reasons: {dict(r['loss_reasons'])}")
        print(f"  prize margin (our prizes still needed at loss): {dict(sorted(r['prize_margin'].items()))}")
        gl = r["game_len"]
        for k in ("win", "loss"):
            xs = gl[k]
            if xs:
                print(f"  game length ({k}): avg {sum(xs)/len(xs):.0f} steps (n={len(xs)})")
        print("\n  FASE 0b — per-archetype record (M29: the field mix drives elo)")
        print(f"  {'archetype':<20}{'W':>4}{'L':>4}{'WR':>8}   loss reasons")
        rows = sorted(r["by_arch"].items(), key=lambda kv: -(kv[1][0] + kv[1][1]))
        for arch, (w, l) in rows:
            wr = w / max(w + l, 1)
            lr = dict(r["loss_reason_by_arch"].get(arch, {}))
            print(f"  {arch:<20}{w:>4}{l:>4}{wr:>7.0%}   {lr if lr else ''}")
    print(f"\nPOOLED live: {tot_w}W-{tot_l}L  (WR {tot_w/max(tot_w+tot_l,1):.1%})")

    print("\n" + "=" * 78)
    print("FASE 1 — bug-class leak audit")
    print("\n[1.1] lethal audit — MLP LIVE (turn-level, ALAKAZAM.damage_fn)")
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
    for k in ("win", "loss"):
        ok, tot = pr["parity_by_outcome"][k]
        print(f"    in {k+'s:':<6} {_pct(ok, tot)}")
    print(f"  bc_used={pr['bc_used']}  bc_failures(SafePolicy-style)={pr['bc_failures']}")
    print(f"  routing: {dict(pr['ctx_routed'])}")
    print(f"  contexts in WON games:  {dict(pr['ctx_by_outcome']['win'].most_common(8))}")
    print(f"  contexts in LOST games: {dict(pr['ctx_by_outcome']['loss'].most_common(8))}")

    print("\n" + "=" * 78)
    print("FASE 2 — clone(live) vs teacher(dataset) per-turn behavior")
    bl = behavioral(live_decisions(), parser, cards)
    bt = behavioral(teacher_decisions(), parser, cards)
    print(f"{'axis':<28}{'mlp-live':>14}{'teacher':>14}")
    for k in ("games", "attacks_per_game", "first_attack_turn_median",
              "main_turns_per_game", "attack_turn_frac", "games_that_ever_attacked",
              "bench_max_avg", "bench_insurance_turn_frac", "hand_size_avg"):
        vl, vt = bl[k], bt[k]
        fl = f"{vl:.3f}" if isinstance(vl, float) else str(vl)
        ft = f"{vt:.3f}" if isinstance(vt, float) else str(vt)
        print(f"{k:<28}{fl:>14}{ft:>14}")

    print("\n" + "=" * 78)
    print("FASE 3 — combo assembly (does the clone actually build Alakazam?)")
    cl = combo_audit(live_decisions(), parser, cards, "mlp-live")
    ct = combo_audit(teacher_decisions(), parser, cards, "teacher")
    print(f"{'axis':<30}{'mlp-live':>14}{'teacher':>14}")
    rows = [
        ("games", cl["games"], ct["games"]),
        ("wins", cl["wins"], ct["wins"]),
        ("Alakazam reached (frac games)", cl["alakazam_frac"], ct["alakazam_frac"]),
        ("  ...in WON games", cl["alakazam_frac_win"], ct["alakazam_frac_win"]),
        ("  ...in LOST games", cl["alakazam_frac_loss"], ct["alakazam_frac_loss"]),
        ("Alakazam median turn", cl["alakazam"]["median"], ct["alakazam"]["median"]),
        ("  ...median turn (win)", cl["alakazam"]["median_win"], ct["alakazam"]["median_win"]),
        ("  ...median turn (loss)", cl["alakazam"]["median_loss"], ct["alakazam"]["median_loss"]),
        ("Kadabra median turn", cl["kadabra"]["median"], ct["kadabra"]["median"]),
        ("Abra median turn", cl["abra"]["median"], ct["abra"]["median"]),
        ("Powerful Hand used (frac games)", cl["ph_frac"], ct["ph_frac"]),
        ("  ...in WON games", cl["ph_frac_win"], ct["ph_frac_win"]),
        ("  ...in LOST games", cl["ph_frac_loss"], ct["ph_frac_loss"]),
        ("PH attacks total", cl["ph_attacks"], ct["ph_attacks"]),
        ("PH hand_size avg (=dmg/20)", cl["ph_hand_avg"], ct["ph_hand_avg"]),
        ("PH first-use median turn", cl["ph_first_turn"]["median"], ct["ph_first_turn"]["median"]),
    ]
    for name, vl, vt in rows:
        fl = f"{vl:.3f}" if isinstance(vl, float) else str(vl)
        ft = f"{vt:.3f}" if isinstance(vt, float) else str(vt)
        print(f"{name:<30}{fl:>14}{ft:>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
