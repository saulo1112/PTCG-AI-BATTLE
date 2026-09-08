"""M33 — deep behavioral analysis of the TEACHER (Yushin Ito, submission 54773249).

Two failed clone-improvement attempts (M32 ALAKAZAM_MATCHUP, ALAKAZAM_SPECIALIST)
were both built on the assumption that the teacher reacts to Munkidori by firing
Powerful Hand earlier. That assumption was falsified: his PH pick-rate is 18.0%
with Munkidori energized vs 18.1% without. This script stops assuming and
characterises what he ACTUALLY does, so the next lever is chosen from evidence.

Sections (mirroring docs/m33_yushin_deep_dive.md):
  1. Field / record sanity check (must reproduce known numbers)
  2. What actually triggers Powerful Hand (conditional pick-rates)
  3. The hand-size economy   <- the central axis: PH damage = 20 x hand_size, so
     every card played COSTS 20 damage. Board development and weapon-loading are
     in direct conflict, and nothing in the project has analysed that tradeoff.
  4. Why Yushin LOSES (never analysed anywhere): loss reasons by archetype,
     deck-out risk, and what separates his wins from his losses in one matchup.

Everything is READ-ONLY. Run:
  uv run --group dev python scratchpad/analyze_yushin.py
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.kaggle_replay import (
    ReplayDecision, _ACTIVE, _DECK_LEN, _strip_observation,
)
from ptcg_ai.observation.models import EnergyKind, OptionKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

import diagnose_mlp as dm
import extract_top_decks as et

TEACHER_FOLDER = Path("replays/54773249")
CLONE_FOLDER = Path("replays/55011997")

POWERFUL_HAND = dm.POWERFUL_HAND       # 1072, 20 x hand_size
CRUEL_ARROW = 183                      # flat 100
ALAKAZAM_ID, KADABRA_ID, ABRA_ID = 743, 742, 741
DUDUNSPARCE_ID, DUNSPARCE_ID, FEZANDIPITI_ID = 66, 305, 140
GRIMM_MARKER, TR_MARKER = 648, 400


# --------------------------------------------------------------------------
# shared primitives (lifted from the 5 copy-pasted versions across scratchpad/)
# --------------------------------------------------------------------------

def decisions_for_seat(path: Path, seat: int):
    """Every ACTIVE-with-select decision made by ``seat`` in one replay.

    Same body as ``kaggle_replay.iter_player_decisions`` but keyed by SEAT rather
    than player name -- needed whenever the seat was found by deck identity.
    Previously duplicated verbatim in 5 scratchpad scripts.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    game_id = path.stem
    steps = data.get("steps", [])
    rewards = data.get("rewards") or []
    won = seat < len(rewards) and (rewards[seat] or 0) > 0
    for i in range(len(steps) - 1):
        row, nxt = steps[i], steps[i + 1]
        if seat >= len(row) or seat >= len(nxt):
            continue
        cell = row[seat]
        if not cell or cell.get("status") != _ACTIVE:
            continue
        obs = cell.get("observation") or {}
        select = obs.get("select")
        options = select.get("option") if select else None
        if not options:
            continue
        action = nxt[seat].get("action")
        if not isinstance(action, list) or len(action) == _DECK_LEN:
            continue
        yield ReplayDecision(
            game_id=game_id, seat=seat, step_index=i,
            raw_observation=_strip_observation(obs),
            action=[int(a) for a in action], won=won,
            context=int(select.get("context", -1)),
            select_type=int(select.get("type", -1)),
            min_count=int(select.get("minCount", 1)),
            max_count=int(select.get("maxCount", 1)),
            n_options=len(options),
        )


def teacher_games(folder: Path = TEACHER_FOLDER) -> list[tuple[Path, int, str, bool]]:
    """(path, teacher_seat, opponent_archetype, won) for every non-mirror game.

    Seat found by DECK IDENTITY (his exact 60-card list), not by player name --
    previously inlined in 3 places and fused into teacher_record() in a 4th.
    """
    yushin = tuple(sorted(int(x) for x in dm.DECK_CSV.read_text(encoding="utf-8").split()))
    out = []
    for fp in sorted(glob.glob(str(folder / "*.json"))):
        if "metadata" in fp:
            continue
        f = Path(fp)
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        decks = [et.extract_deck(data, s) for s in (0, 1)]
        seats = [s for s in (0, 1) if decks[s] and tuple(sorted(decks[s])) == yushin]
        if len(seats) != 1:            # 0 = unreadable, 2 = true mirror
            continue
        seat = seats[0]
        rew = (data.get("rewards") or [0, 0])[seat]
        out.append((f, seat, dm._archetype(decks[1 - seat]), rew > 0))
    return out


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} = {a / max(b, 1):.1%}"


def _fmt_stats(vals: list[float]) -> str:
    if not vals:
        return "n=0"
    return (f"n={len(vals)} mean={statistics.mean(vals):.2f} "
            f"median={statistics.median(vals):.1f}")


# --------------------------------------------------------------------------
# per-decision feature extraction
# --------------------------------------------------------------------------

class DecisionView:
    """The board facts every section below conditions on, computed once."""

    __slots__ = ("turn", "hand_size", "opp_hp", "ph_legal", "ph_idx", "chose_ph",
                 "ph_damage", "lethal", "overkill", "other_attack_legal",
                 "chose_type", "won", "game_id", "arch", "opp_hand",
                 "my_bench", "deck_count", "alakazam_in_play")

    def __init__(self, dec: ReplayDecision, obs, gs: GameState, arch: str):
        self.game_id = dec.game_id
        self.won = dec.won
        self.arch = arch
        self.turn = obs.current.turn
        self.hand_size = gs.hand_size
        self.opp_hand = gs.opponent.handCount if gs.opponent else 0
        self.my_bench = gs.my_bench_count
        self.deck_count = gs.my_deck_count
        opp_active = gs.opp_active
        self.opp_hp = opp_active.hp if opp_active is not None else 0
        me = gs.me
        in_play = ([p for p in me.active if p is not None] + list(me.bench)) if me else []
        self.alakazam_in_play = any(p.id == ALAKAZAM_ID for p in in_play)

        self.ph_idx = [
            i for i, o in enumerate(obs.select.option)
            if o.type is OptionKind.ATTACK and o.attackId == POWERFUL_HAND
        ]
        self.ph_legal = bool(self.ph_idx)
        self.other_attack_legal = any(
            o.type is OptionKind.ATTACK and o.attackId != POWERFUL_HAND
            for o in obs.select.option
        )
        self.ph_damage = 20 * self.hand_size
        self.lethal = self.opp_hp > 0 and self.ph_damage >= self.opp_hp
        self.overkill = self.ph_damage - self.opp_hp
        self.chose_ph = any(a in self.ph_idx for a in dec.action)
        chosen = [obs.select.option[a] for a in dec.action if 0 <= a < len(obs.select.option)]
        self.chose_type = chosen[0].type.name if chosen else "NONE"


def iter_views(games, parser, cards, *, main_only=True, limit_games=None):
    for n, (f, seat, arch, _won) in enumerate(games):
        if limit_games and n >= limit_games:
            break
        for dec in decisions_for_seat(f, seat):
            if main_only and SelectContextKind(dec.context) is not SelectContextKind.MAIN:
                continue
            obs = parser.parse(dec.raw_observation)
            if obs.select is None or obs.current is None:
                continue
            gs = GameState.build(obs, cards)
            yield DecisionView(dec, obs, gs, arch)


# --------------------------------------------------------------------------
# Section 1 — record sanity check
# --------------------------------------------------------------------------

def section_record(games) -> None:
    print("=" * 78)
    print("SECTION 1 — record & field (sanity check vs known M32 numbers)")
    print("=" * 78)
    by_arch: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for _f, _s, arch, won in games:
        by_arch[arch][0 if won else 1] += 1
    w = sum(v[0] for v in by_arch.values())
    l = sum(v[1] for v in by_arch.values())
    print(f"non-mirror games: {w + l}   record {w}-{l} = {w / max(w + l, 1):.1%}")
    print(f"  (M32 reference on 1284 eps: 612-451 = 57.6%; more eps now, so expect a shift)")
    print(f"\n  {'archetype':<24}{'W':>5}{'L':>5}{'n':>6}{'WR':>8}{'field share':>13}")
    for arch, (aw, al) in sorted(by_arch.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        n = aw + al
        print(f"  {arch:<24}{aw:>5}{al:>5}{n:>6}{aw / max(n, 1):>7.1%}{n / max(w + l, 1):>12.1%}")


# --------------------------------------------------------------------------
# Section 2 — what actually triggers Powerful Hand
# --------------------------------------------------------------------------

def _rate_table(title: str, buckets: dict[str, list[int]]) -> None:
    """buckets[label] = [n_chose_ph, n_legal]"""
    print(f"\n  {title}")
    print(f"    {'bucket':<28}{'chose PH':>10}{'PH legal':>10}{'pick-rate':>12}")
    for label, (chose, legal) in buckets.items():
        if legal == 0:
            continue
        print(f"    {label:<28}{chose:>10}{legal:>10}{chose / legal:>11.1%}")


def section_ph_trigger(views: list[DecisionView]) -> None:
    print("\n" + "=" * 78)
    print("SECTION 2 — what actually triggers Powerful Hand")
    print("=" * 78)
    legal = [v for v in views if v.ph_legal]
    chose = [v for v in legal if v.chose_ph]
    print(f"MAIN decisions where PH was legal: {len(legal)}")
    print(f"  per-DECISION pick rate: {_pct(len(chose), len(legal))}")
    print("  !! M20's lesson: a per-DECISION rate is DILUTED -- a turn contains many")
    print("     setup decisions (play/evolve/attach) and at most ONE attack, so this")
    print("     number understates his aggression. The per-TURN rate below is the")
    print("     correct unit. (The 18.0/18.1% Munkidori comparison shares this")
    print("     dilution, but it was a RELATIVE comparison so it still holds.)")

    # ---- per-TURN aggregation (the M20-correct unit) --------------------
    turn_legal: dict[tuple[str, int], bool] = {}
    turn_fired: dict[tuple[str, int], bool] = {}
    turn_best: dict[tuple[str, int], DecisionView] = {}
    for v in views:
        key = (v.game_id, v.turn)
        if v.ph_legal:
            turn_legal[key] = True
            # keep the view with the largest hand seen while PH was legal that turn
            cur = turn_best.get(key)
            if cur is None or v.hand_size > cur.hand_size:
                turn_best[key] = v
        if v.chose_ph:
            turn_fired[key] = True
    n_turns_legal = len(turn_legal)
    n_turns_fired = sum(1 for k in turn_legal if turn_fired.get(k))
    print(f"\n  TURNS where PH was legal at some point: {n_turns_legal}")
    print(f"  turns where he FIRED it: {_pct(n_turns_fired, n_turns_legal)}   <- the real rate")

    def bucketise(keyfn, order=None):
        b: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for v in legal:
            k = keyfn(v)
            b[k][1] += 1
            if v.chose_ph:
                b[k][0] += 1
        if order:
            return {k: b[k] for k in order if k in b}
        return dict(sorted(b.items()))

    def bucketise_turn(keyfn, order=None):
        """Same conditioning, but one sample per TURN (uses the best-hand view)."""
        b: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for key, v in turn_best.items():
            k = keyfn(v)
            b[k][1] += 1
            if turn_fired.get(key):
                b[k][0] += 1
        if order:
            return {k: b[k] for k in order if k in b}
        return dict(sorted(b.items()))

    _rate_table("PER-TURN by LETHALITY (correct unit)",
                bucketise_turn(lambda v: "LETHAL" if v.lethal else "not lethal",
                               order=["LETHAL", "not lethal"]))

    _rate_table("by LETHALITY (is 20*hand >= opp active HP?)",
                bucketise(lambda v: "LETHAL" if v.lethal else "not lethal",
                          order=["LETHAL", "not lethal"]))

    def hand_bucket(v):
        h = v.hand_size
        for lo, hi in ((0, 4), (5, 7), (8, 10), (11, 13), (14, 16)):
            if lo <= h <= hi:
                return f"hand {lo}-{hi}"
        return "hand 17+"
    _rate_table("by HAND SIZE (= the damage dial, 20 per card)", bucketise(hand_bucket))

    def overkill_bucket(v):
        if not v.lethal:
            return "not lethal"
        o = v.overkill
        if o <= 20:
            return "lethal, margin <=20"
        if o <= 60:
            return "lethal, margin 21-60"
        if o <= 120:
            return "lethal, margin 61-120"
        return "lethal, margin >120"
    _rate_table("by OVERKILL MARGIN (how much damage is wasted)", bucketise(overkill_bucket))

    def turn_bucket(v):
        t = v.turn
        for lo, hi in ((1, 3), (4, 5), (6, 8), (9, 12)):
            if lo <= t <= hi:
                return f"turn {lo}-{hi}"
        return "turn 13+"
    _rate_table("by TURN", bucketise(turn_bucket))

    _rate_table("by ALTERNATIVE ATTACK AVAILABLE",
                bucketise(lambda v: "other attack legal" if v.other_attack_legal else "PH is only attack"))

    _rate_table("by MATCHUP (top archetypes)", {
        k: v for k, v in bucketise(lambda v: v.arch).items() if v[1] >= 300
    })

    print("\n  what he does INSTEAD on the ~82% (chosen option type when PH was legal but declined):")
    other = collections.Counter(v.chose_type for v in legal if not v.chose_ph)
    tot = sum(other.values())
    for t, n in other.most_common(10):
        print(f"    {t:<28}{n:>8}{n / max(tot, 1):>9.1%}")

    print("\n  LETHAL x hand-size cross-tab (does he hold back even when lethal?):")
    cross: dict[tuple, list[int]] = collections.defaultdict(lambda: [0, 0])
    for v in legal:
        if not v.lethal:
            continue
        cross[hand_bucket(v)][1] += 1
        if v.chose_ph:
            cross[hand_bucket(v)][0] += 1
    for k in sorted(cross):
        c, n = cross[k]
        print(f"    LETHAL & {k:<20}{c:>8}/{n:<8}{c / max(n, 1):>8.1%}")


# --------------------------------------------------------------------------
# Section 3 — the hand-size economy
# --------------------------------------------------------------------------

def section_hand_economy(teacher_views: list[DecisionView],
                         clone_views: list[DecisionView]) -> None:
    print("\n" + "=" * 78)
    print("SECTION 3 — the hand-size economy (PH damage = 20 x hand_size)")
    print("=" * 78)

    def traj(views, label):
        per_turn: dict[int, list[int]] = collections.defaultdict(list)
        # one sample per (game, turn): the hand size at the FIRST decision of that turn
        seen: set[tuple[str, int]] = set()
        for v in views:
            key = (v.game_id, v.turn)
            if key in seen:
                continue
            seen.add(key)
            per_turn[v.turn].append(v.hand_size)
        print(f"\n  hand size at start of each turn — {label}")
        print(f"    {'turn':<8}{'n':>7}{'mean hand':>12}{'implied PH dmg':>16}")
        for t in sorted(per_turn):
            if t > 12 or len(per_turn[t]) < 10:
                continue
            m = statistics.mean(per_turn[t])
            print(f"    {t:<8}{len(per_turn[t]):>7}{m:>12.2f}{20 * m:>16.0f}")
        return per_turn

    t_traj = traj(teacher_views, "TEACHER")
    c_traj = traj(clone_views, "CLONE")

    print("\n  side-by-side mean hand size (teacher vs clone):")
    print(f"    {'turn':<8}{'teacher':>10}{'clone':>10}{'delta':>10}")
    for t in sorted(set(t_traj) & set(c_traj)):
        if t > 12 or len(t_traj[t]) < 10 or len(c_traj[t]) < 10:
            continue
        tm, cm = statistics.mean(t_traj[t]), statistics.mean(c_traj[t])
        print(f"    {t:<8}{tm:>10.2f}{cm:>10.2f}{cm - tm:>+10.2f}")

    # actual PH damage dealt at fire time
    for views, label in ((teacher_views, "TEACHER"), (clone_views, "CLONE")):
        fired = [v for v in views if v.chose_ph]
        dmg = [v.ph_damage for v in fired]
        hands = [v.hand_size for v in fired]
        print(f"\n  {label}: Powerful Hand actually fired {len(fired)} times")
        print(f"    hand size at fire : {_fmt_stats([float(h) for h in hands])}")
        print(f"    damage dealt      : {_fmt_stats([float(d) for d in dmg])}")
        if dmg:
            kills = sum(1 for v in fired if v.lethal)
            print(f"    of which lethal   : {_pct(kills, len(fired))}")

    # decisions per turn = a proxy for how many cards get played
    for views, label in ((teacher_views, "TEACHER"), (clone_views, "CLONE")):
        per_game_turn: dict[tuple[str, int], int] = collections.Counter()
        for v in views:
            per_game_turn[(v.game_id, v.turn)] += 1
        vals = [float(x) for x in per_game_turn.values()]
        print(f"\n  {label}: MAIN decisions per turn (proxy for cards played): {_fmt_stats(vals)}")

    # ---- THE KEY MEASUREMENT: hand GROWTH within a single turn ----------
    # The smoke test showed mean hand at START of turn 4 is ~7, yet he fires PH
    # on turn 4 with a hand of ~14 -- so the hand roughly DOUBLES inside one turn
    # via the draw engines, and THAT growth (not the turn number) is the skill.
    print("\n  " + "-" * 70)
    print("  HAND GROWTH WITHIN A TURN (start of turn -> size when PH is fired)")
    print("  " + "-" * 70)
    for views, label in ((teacher_views, "TEACHER"), (clone_views, "CLONE")):
        start: dict[tuple[str, int], int] = {}
        fire: dict[tuple[str, int], int] = {}
        for v in views:
            key = (v.game_id, v.turn)
            if key not in start:
                start[key] = v.hand_size
            if v.chose_ph:
                fire[key] = v.hand_size
        growth = [float(fire[k] - start[k]) for k in fire if k in start]
        at_fire = [float(x) for x in fire.values()]
        starts = [float(start[k]) for k in fire if k in start]
        print(f"\n  {label} (turns where PH was fired, n={len(growth)}):")
        print(f"    hand at START of that turn : {_fmt_stats(starts)}")
        print(f"    hand WHEN FIRED            : {_fmt_stats(at_fire)}")
        print(f"    growth within the turn     : {_fmt_stats(growth)}")
        if growth:
            print(f"    => extra damage from growth: {20 * statistics.mean(growth):.0f}")


# --------------------------------------------------------------------------
# Section 4 — why Yushin loses
# --------------------------------------------------------------------------

def section_losses(games, parser, cards) -> None:
    print("\n" + "=" * 78)
    print("SECTION 4 — why Yushin LOSES (never analysed before)")
    print("=" * 78)
    reasons_by_arch: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    all_reasons: collections.Counter = collections.Counter()
    prize_margin: collections.Counter = collections.Counter()
    losses = 0
    for f, seat, arch, won in games:
        if won:
            continue
        losses += 1
        data = json.loads(f.read_text(encoding="utf-8"))
        reason, prizes_left = dm.loss_reason(data, seat)
        reasons_by_arch[arch][reason] += 1
        all_reasons[reason] += 1
        prize_margin[prizes_left] += 1

    print(f"total losses analysed: {losses}")
    print(f"\n  loss reasons overall: {dict(all_reasons)}")
    for r, n in all_reasons.most_common():
        print(f"    {r:<16}{n:>6}{n / max(losses, 1):>9.1%}")
    print(f"\n  prizes still needed when he lost (6 = never took one):")
    for p in sorted(prize_margin):
        print(f"    {p} prizes left: {prize_margin[p]:>5}")

    print(f"\n  loss reasons BY ARCHETYPE (n>=10):")
    print(f"    {'archetype':<24}{'losses':>8}   breakdown")
    for arch, ctr in sorted(reasons_by_arch.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(ctr.values())
        if n < 10:
            continue
        print(f"    {arch:<24}{n:>8}   {dict(ctr)}")


def section_win_vs_loss(views: list[DecisionView], arch: str) -> None:
    print("\n" + "-" * 78)
    print(f"SECTION 4b — what separates his WINS from his LOSSES vs {arch}")
    print("-" * 78)
    sel = [v for v in views if v.arch == arch]
    if not sel:
        print("  no games")
        return
    for won in (True, False):
        sub = [v for v in sel if v.won == won]
        games = {v.game_id for v in sub}
        fired = [v for v in sub if v.chose_ph]
        first_ph: dict[str, int] = {}
        for v in sorted(fired, key=lambda v: v.turn):
            first_ph.setdefault(v.game_id, v.turn)
        max_turn: dict[str, int] = {}
        for v in sub:
            max_turn[v.game_id] = max(max_turn.get(v.game_id, 0), v.turn)
        label = "WINS" if won else "LOSSES"
        print(f"\n  {label}: {len(games)} games, {len(sub)} MAIN decisions")
        print(f"    PH fired            : {len(fired)} times "
              f"({len(fired) / max(len(games), 1):.2f} per game)")
        print(f"    first PH turn       : {_fmt_stats([float(t) for t in first_ph.values()])}")
        print(f"    games that fired PH : {_pct(len(first_ph), len(games))}")
        print(f"    hand size at fire   : {_fmt_stats([float(v.hand_size) for v in fired])}")
        print(f"    PH damage at fire   : {_fmt_stats([float(v.ph_damage) for v in fired])}")
        print(f"    game length (turns) : {_fmt_stats([float(t) for t in max_turn.values()])}")
        print(f"    mean hand (all dec) : {statistics.mean([v.hand_size for v in sub]):.2f}")
        print(f"    mean deck left      : {statistics.mean([v.deck_count for v in sub]):.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-games", type=int, default=None,
                     help="cap teacher games parsed (dev speed); default = all")
    args = ap.parse_args()

    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()

    print("loading teacher games (seat by deck identity)...", flush=True)
    games = teacher_games()
    if args.limit_games:
        games = games[: args.limit_games]
    print(f"  {len(games)} non-mirror teacher games\n", flush=True)

    section_record(games)

    print("\nparsing teacher MAIN decisions...", flush=True)
    tviews = list(iter_views(games, parser, cards))
    print(f"  {len(tviews)} teacher MAIN decisions", flush=True)

    section_ph_trigger(tviews)

    # clone side, for the economy comparison
    print("\nparsing clone MAIN decisions...", flush=True)
    from ptcg_ai.imitation.kaggle_replay import player_seats
    clone_games = []
    for fp in sorted(glob.glob(str(CLONE_FOLDER / "*.json"))):
        if "metadata" in fp:
            continue
        f = Path(fp)
        d = json.loads(f.read_text(encoding="utf-8"))
        seats = player_seats(d, dm.OUR)
        if len(seats) != 1:
            continue
        seat = seats[0]
        deck = et.extract_deck(d, 1 - seat)
        rew = (d.get("rewards") or [0, 0])[seat]
        clone_games.append((f, seat, dm._archetype(deck), rew > 0))
    cviews = list(iter_views(clone_games, parser, cards))
    print(f"  {len(cviews)} clone MAIN decisions ({len(clone_games)} games)", flush=True)

    section_hand_economy(tviews, cviews)
    section_losses(games, parser, cards)
    section_win_vs_loss(tviews, "Marnie's Grimmsnarl")
    section_win_vs_loss(tviews, "Team Rocket")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
