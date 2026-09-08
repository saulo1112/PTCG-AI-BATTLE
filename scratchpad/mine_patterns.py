"""M20 Fase A — mine DISCRETE, explainable decision patterns from strong players' replays.

The M20 premise (docs/m20_findings.md): cloning a strong player's decisions WHOLE fails
(M8/M9/M14 — the better the player, the worse the clone), and importing rules from external
strategy guides fails (M17 — most claims were about a different game / absent cards). What
DID survive M17 was a rule grounded in an exact mechanical fact of THIS engine. So: instead
of cloning behavior or importing prose, extract *discrete conditional patterns* from the
behavior of players we already have on disk, and audit each candidate against hard facts.

Method (deck-independent by construction):

  decision  ->  (situation cell, set of AVAILABLE OptionKinds, CHOSEN OptionKind)

`OptionKind` (observation/models.py, already used by the BC featurizer) is the deck-agnostic
action taxonomy: PLAY / ATTACH / EVOLVE / ABILITY / ATTACK / RETREAT / END / ...  The
situation cell is built only from quantities `GameState` already derives, so any pattern is
expressible as an if-then over named state, never over raw option indices.

The statistic is a CONDITIONAL rate, not a raw frequency: for a kind K we count only the
decisions where K was actually AVAILABLE, so "strong players retreat more" can never be an
artifact of their deck simply offering more retreats. We then contrast:

  STRONG pool  (>=850 elo, several distinct archetypes)   vs   the 650 teacher v1 imitates.

That difference is exactly the piloting gap imitation-v1 inherits from its teacher.

Anti-spurious protocol (pre-registered, mirrors rl_selfplay.py's kill criteria):
  * winner bias   — every game is mined regardless of outcome; `won` is descriptive only.
  * deck confound — a cell must replicate across >=3 strong players of DIFFERENT decks.
  * multiplicity  — discovery pool vs a held-out player split BY PLAYER (not by game).
  * K-D           — no cell clears (support>=40, >=3 players, |gap|>=25pts, holdout agrees
                    within 15pts) => the line is CLOSED and documented as a negative.

Structured-damage caveat (the M17/T3 finding): `has_lethal` uses GameState.attack_damage,
which is blind to prose-scaling attacks (Rocket Rush, Voltaic Chain). Cells whose reading
depends on that are flagged `~prose` and must be re-derived with exact math in Fase B.

Usage:
  uv run --group dev python scratchpad/mine_patterns.py            # full report
  ... --max-games 120        # games per player (cost control)
  ... --holdout "jiatu.l"    # which strong player is held out of discovery
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config.loader import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset
from ptcg_ai.imitation.kaggle_replay import iter_player_decisions
from ptcg_ai.observation.models import OptionKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

# --- the population, with ladder scores read from the Kaggle episode API -------------
# (tools/find_candidates.py --ours 54555926 54556007, 2026-07-21)

@dataclass(frozen=True)
class Source:
    label: str
    src: str           # a replay DIRECTORY, or a prebuilt data/imitation/*.jsonl.gz
    player: str
    score: float
    pool: str          # "strong" | "teacher"


SOURCES: tuple[Source, ...] = (
    # --- strong pool: everything meaningfully above the 650 teacher -----------------
    Source("tuna_1223", "replays/54708568", "懒惰的金枪鱼", 1222.8, "strong"),
    Source("jiatu_1085", "replays/54611538", "jiatu.l", 1084.6, "strong"),
    Source("vibechu_top", "Logs/Higher ranking logs/Vibechu", "vibechu", 1000.0, "strong"),
    Source("majkel_top", "Logs/Higher ranking logs/Majkel", "Majkel1337", 1000.0, "strong"),
    Source("kenn_940", "data/imitation/kenn2439.jsonl.gz", "kenN2439", 940.0, "strong"),
    Source("ricecake_897", "replays/54757913", "sam_the_rice_cake", 896.9, "strong"),
    Source("nikita_800", "Logs/Higher ranking logs/800 elo", "[RU] Nikita Kuznetsov", 800.0, "strong"),
    Source("shu_776", "replays/54556193", "shu", 775.9, "strong"),
    Source("kazama_729", "replays/54767753", "Kazama Yusuke", 729.4, "strong"),
    # --- the baseline: the exact bot imitation-v1 clones ----------------------------
    Source("teacher_650", "Logs/Higher ranking logs/650 elo", "greengreenpurple", 650.0, "teacher"),
    # --- deck-confound control: OTHER weak players, different decks -----------------
    # Without these, "strong pool vs one 650 bot" cannot separate skill from that one
    # deck's idiosyncrasies. A pattern is only credible if the strong pool departs from
    # the teacher AND from these, i.e. the behaviour tracks elo, not decklist.
    Source("takehiro_706", "replays/54569163", "PP.TAKEHIRO_KAWADA", 705.8, "weak"),
    Source("rocket_646", "replays/54730288", "Team Rocket", 646.5, "weak"),
    Source("vvhan_586", "replays/54534563", "vvhan", 585.7, "weak"),
)

#: vibechu / Majkel1337 are M6's #1/#2 ladder players — their exact live score is not in our
#: episode history (we never faced them), so 1000.0 is a placeholder ORDERING value, never
#: used as a number. Every other score is read from the Kaggle episode API.


def _ascii(s: str) -> str:
    """Windows consoles here are cp1252 (M19 lesson) — never print raw CJK player names."""
    return s.encode("ascii", "replace").decode("ascii")

MAIN = SelectContextKind.MAIN

# Kinds worth contrasting (the ones a MAIN-phase pilot actually chooses between).
_KINDS = (
    OptionKind.ATTACK, OptionKind.RETREAT, OptionKind.PLAY, OptionKind.ATTACH,
    OptionKind.EVOLVE, OptionKind.ABILITY, OptionKind.END,
)


@dataclass(frozen=True)
class Situation:
    """One PLAYER TURN reduced to deck-independent, auditable coordinates.

    The unit is deliberately the turn, not the decision. MAIN is a multi-decision
    phase (a turn is ~4-5 selects: play, attach, ability, ... then ATTACK or END), so
    a per-decision rate dilutes every strategic choice with the setup clicks that
    precede it in the same turn — measured, and it is what made the first version of
    this harness read the known-perfect lethal discipline as 0.10-0.40.

    Conditioning state is read at the turn's FIRST decision (the board as the player
    inherits it); availability and choice are unioned over the whole turn ("was this
    ever offered" / "was it ever taken").
    """

    dies_if_pass: bool       # opponent's best payable attack (+1 attach) KOs my Active
    has_lethal: bool         # some offered attack KOs their Active (structured damage)
    atk_avail: bool          # ATTACK was offered at some point this turn
    bench_ready: bool        # >=1 benched Pokemon (nearly) able to attack
    prize: str               # "behind" | "even" | "ahead"  (prize race, my POV)
    opp_ex: bool             # their Active is worth >=2 prizes
    early: bool              # turn <= 3
    available: frozenset     # OptionKinds offered anywhere in the turn
    chosen: frozenset        # OptionKinds actually taken during the turn
    n_decisions: int
    play_offered: int        # decisions this turn that offered PLAY
    play_taken: int          # ... on which a card was actually played
    won: bool


def _cards() -> CardDatabase:
    sdk = load_sdk(load_config(profile="benchmark").paths.sdk_dir)
    return CardDatabase.from_sdk(sdk)


def _lethal_available(gs: GameState, obs, cards: CardDatabase) -> bool:
    """Does an OFFERED attack option KO the opponent's Active right now?

    Structured damage only (GameState.attack_damage) — see the module docstring's
    prose caveat. Deliberately conservative: it under-counts lethal for swarm decks.
    """
    my_active, opp_active = gs.my_active, gs.opp_active
    if my_active is None or opp_active is None or obs.select is None:
        return False
    for opt in obs.select.option:
        if opt.type is not OptionKind.ATTACK or opt.attackId is None:
            continue
        atk = cards.get_attack(opt.attackId)
        if atk is None:
            continue
        if gs.attack_damage(atk, my_active, opp_active) >= opp_active.hp:
            return True
    return False


def _iter_decisions(src: Source, max_games: int):
    """Yield ReplayDecision rows from a replay directory OR a prebuilt dataset.

    Both paths reuse the audited loaders (kaggle_replay / dataset) — no new parsing.
    ``max_games`` caps DISTINCT games, so cost is bounded the same way for both.
    """
    path = Path(src.src)
    if path.is_file():  # prebuilt data/imitation/*.jsonl.gz
        seen: set[str] = set()
        for dec in read_decision_dataset(path):
            if dec.game_id not in seen:
                if len(seen) >= max_games:
                    continue
                seen.add(dec.game_id)
            yield dec
        return
    seen_games = 0
    for f in sorted(path.glob("*.json")):
        if f.name == "metadata.json":
            continue
        if seen_games >= max_games:
            break
        try:
            decisions = list(iter_player_decisions(f, src.player))
        except Exception:
            continue
        if not decisions:
            continue
        seen_games += 1
        yield from decisions


def iter_situations(src: Source, cards: CardDatabase, parser: ObservationParser,
                    max_games: int) -> "list[Situation]":
    """Reduce one player's MAIN decisions to one Situation row per player-turn."""
    # (game_id, seat, turn) -> accumulator; turn numbers alternate by seat, so the
    # triple identifies one player-turn uniquely.
    turns: dict[tuple, dict] = {}
    order: list[tuple] = []
    for dec in _iter_decisions(src, max_games):
        if dec.context != int(MAIN) or not dec.action or not dec.legal:
            continue
        try:
            obs = parser.parse(dec.raw_observation)
        except Exception:
            continue
        select = obs.select
        if select is None or not select.option:
            continue
        idx = dec.action[0]
        if not (0 <= idx < len(select.option)):
            continue
        gs = GameState.build(obs, cards)
        my_active, opp_active = gs.my_active, gs.opp_active
        if my_active is None or opp_active is None:
            continue  # no board yet: nothing to condition on
        turn_no = obs.current.turn if obs.current is not None else 0
        key = (dec.game_id, dec.seat, turn_no)
        acc = turns.get(key)
        if acc is None:
            diff = gs.opp_prizes_left - gs.my_prizes_left
            acc = {
                "dies_if_pass": gs.max_threat(opp_active, my_active, extra_energy=1) >= my_active.hp,
                "bench_ready": gs.reserve_attackers(gs.me) >= 1,
                "prize": "ahead" if diff > 0 else ("behind" if diff < 0 else "even"),
                "opp_ex": gs.opp_active_prize_value >= 2,
                "early": turn_no <= 3,
                "has_lethal": False, "available": set(), "chosen": set(),
                "n": 0, "play_offered": 0, "play_taken": 0, "won": dec.won,
            }
            turns[key] = acc
            order.append(key)
        acc["n"] += 1
        kinds = {o.type for o in select.option}
        acc["available"].update(kinds)
        picked = select.option[idx].type
        acc["chosen"].add(picked)
        if OptionKind.PLAY in kinds:
            acc["play_offered"] += 1
            if picked is OptionKind.PLAY:
                acc["play_taken"] += 1
        # lethal is unioned over the turn: the player may only reach it after attaching.
        if not acc["has_lethal"] and _lethal_available(gs, obs, cards):
            acc["has_lethal"] = True
    return [
        Situation(
            dies_if_pass=a["dies_if_pass"], has_lethal=a["has_lethal"],
            atk_avail=OptionKind.ATTACK in a["available"],
            bench_ready=a["bench_ready"], prize=a["prize"], opp_ex=a["opp_ex"],
            early=a["early"], available=frozenset(a["available"]),
            chosen=frozenset(a["chosen"]), n_decisions=a["n"],
            play_offered=a["play_offered"], play_taken=a["play_taken"], won=a["won"],
        )
        for a in (turns[k] for k in order)
    ]


# --- cells --------------------------------------------------------------------------

def cell_key(s: Situation, axes: tuple[str, ...]) -> tuple:
    return tuple(getattr(s, a) for a in axes)


def conditional_rates(rows: "list[Situation]", axes: tuple[str, ...]):
    """(cell, kind) -> [n_available, n_chosen], counting ONLY decisions offering the kind."""
    acc: dict[tuple, dict[OptionKind, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0, 0]))
    for s in rows:
        key = cell_key(s, axes)
        for k in _KINDS:
            if k in s.available:
                acc[key][k][0] += 1
                if k in s.chosen:
                    acc[key][k][1] += 1
    return acc


def _rate(pair: list[int]) -> float:
    return pair[1] / pair[0] if pair[0] else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--max-games", type=int, default=150)
    ap.add_argument("--holdout", default="jiatu.l", help="strong player excluded from discovery")
    ap.add_argument("--min-support", type=int, default=40)
    ap.add_argument("--min-players", type=int, default=3)
    ap.add_argument("--min-gap", type=float, default=0.25)
    ap.add_argument("--out", default="data/rl/m20_mine.json")
    args = ap.parse_args()

    cards = _cards()
    parser = ObservationParser()
    print(f"cards={len(cards)}  max_games/player={args.max_games}\n")

    per_player: dict[str, list[Situation]] = {}
    meta: list[dict] = []
    for src in SOURCES:
        t0 = time.perf_counter()
        rows = iter_situations(src, cards, parser, args.max_games)
        per_player[src.label] = rows
        meta.append({"label": src.label, "player": src.player, "score": src.score,
                     "pool": src.pool, "rows": len(rows)})
        print(f"{src.label:<14} {_ascii(src.player)[:20]:<22} score={src.score:>7.1f} "
              f"pool={src.pool:<8} MAIN rows={len(rows):>6}  [{time.perf_counter()-t0:.0f}s]")
        if not rows:
            print(f"  !! NO decisions for {_ascii(src.player)!r} in {src.src} - check name/path")

    strong_labels = [s.label for s in SOURCES if s.pool == "strong" and per_player[s.label]]
    hold_labels = [s.label for s in SOURCES if s.player == args.holdout]
    disc_labels = [l for l in strong_labels if l not in hold_labels]
    teach_labels = [s.label for s in SOURCES if s.pool == "teacher"]
    weak_labels = [s.label for s in SOURCES if s.pool == "weak" and per_player[s.label]]

    discovery = [r for l in disc_labels for r in per_player[l]]
    holdout = [r for l in hold_labels for r in per_player[l]]
    teacher = [r for l in teach_labels for r in per_player[l]]
    weak = [r for l in weak_labels for r in per_player[l]]
    print(f"\ndiscovery={len(discovery)} rows ({len(disc_labels)} players)   "
          f"holdout={len(holdout)}   teacher650={len(teacher)}   "
          f"weak_control={len(weak)} ({len(weak_labels)} players)")

    # --- SANITY: the harness must rediscover lethal discipline ---------------------
    print(f"\n{'='*78}\nSANITY — lethal discipline (must be high for strong players)")
    print(f"{'player':<16}{'lethal&ATTACK avail':>22}{'ATTACK rate':>14}")
    for src in SOURCES:
        rows = [r for r in per_player[src.label]
                if r.has_lethal and OptionKind.ATTACK in r.available]
        rate = (sum(1 for r in rows if OptionKind.ATTACK in r.chosen) / len(rows)
                if rows else float("nan"))
        print(f"{src.label:<16}{len(rows):>22}{rate:>14.3f}")

    # --- the divergence table ------------------------------------------------------
    # `atk_avail` is an AXIS, not just a covariate: without it the END rate ("ended the
    # turn without attacking") silently mixes "chose not to attack" with "could not
    # attack at all", which is a property of the deck's energy curve, not of piloting.
    # Turns with an immediate lethal are excluded outright — the sanity block above
    # shows every population converts those at 0.92-1.00, so there is nothing to learn
    # there; the open question is what strong players do when NO KO is on the table.
    axes = ("dies_if_pass", "atk_avail", "bench_ready", "prize")
    discovery = [r for r in discovery if not r.has_lethal]
    holdout = [r for r in holdout if not r.has_lethal]
    teacher = [r for r in teacher if not r.has_lethal]
    weak = [r for r in weak if not r.has_lethal]
    per_player = {k: [r for r in v if not r.has_lethal] for k, v in per_player.items()}
    print(f"\nno-lethal turns only: discovery={len(discovery)} holdout={len(holdout)} "
          f"teacher={len(teacher)} weak={len(weak)}")
    d_acc = conditional_rates(discovery, axes)
    t_acc = conditional_rates(teacher, axes)
    h_acc = conditional_rates(holdout, axes)
    w_acc = conditional_rates(weak, axes)
    per_player_acc = {l: conditional_rates(per_player[l], axes)
                      for l in disc_labels + weak_labels + teach_labels + hold_labels}

    findings = []
    for cell, kinds in d_acc.items():
        for kind, pair in kinds.items():
            n_s = pair[0]
            n_t = t_acc.get(cell, {}).get(kind, [0, 0])[0]
            if n_s < args.min_support or n_t < args.min_support:
                continue
            r_s, r_t = _rate(pair), _rate(t_acc[cell][kind])
            gap = r_s - r_t
            # how many discovery players independently show the same direction
            agree = 0
            for l in disc_labels:
                p = per_player_acc[l].get(cell, {}).get(kind, [0, 0])
                if p[0] >= 10 and (_rate(p) - r_t) * gap > 0:
                    agree += 1
            hp = h_acc.get(cell, {}).get(kind, [0, 0])
            r_h = _rate(hp) if hp[0] >= 10 else float("nan")
            wp = w_acc.get(cell, {}).get(kind, [0, 0])
            r_w = _rate(wp) if wp[0] >= 20 else float("nan")
            findings.append({
                "cell": dict(zip(axes, [str(c) for c in cell])), "kind": kind.name,
                "n_strong": n_s, "rate_strong": round(r_s, 3),
                "n_teacher": n_t, "rate_teacher": round(r_t, 3),
                "gap": round(gap, 3), "players_agree": agree,
                "n_holdout": hp[0], "rate_holdout": None if hp[0] < 10 else round(r_h, 3),
                "n_weak": wp[0], "rate_weak": None if wp[0] < 20 else round(r_w, 3),
                "per_player": {
                    l: [per_player_acc[l].get(cell, {}).get(kind, [0, 0])[0],
                        round(_rate(per_player_acc[l][cell][kind]), 3)
                        if per_player_acc[l].get(cell, {}).get(kind, [0, 0])[0] else None]
                    for l in per_player_acc
                },
            })

    findings.sort(key=lambda f: -abs(f["gap"]))
    print(f"\n{'='*78}\nDIVERGENCE — conditional rate among decisions OFFERING the kind")
    print(f"(cell axes: {axes};  strong = {len(disc_labels)} players, holdout = {args.holdout})\n")
    hdr = (f"{'kind':<9}{'dies':<6}{'canat':<7}{'bench':<7}{'prize':<8}"
           f"{'n_str':>6}{'strong':>8}{'n_650':>7}{'650':>7}{'gap':>8}{'agr':>5}"
           f"{'hold':>8}{'weak':>8}")
    print(hdr); print("-" * len(hdr))
    for f in findings[:30]:
        c = f["cell"]
        hold = "-" if f["rate_holdout"] is None else f"{f['rate_holdout']:.3f}"
        wk = "-" if f["rate_weak"] is None else f"{f['rate_weak']:.3f}"
        print(f"{f['kind']:<9}{c['dies_if_pass'][:5]:<6}{c['atk_avail'][:5]:<7}"
              f"{c['bench_ready'][:5]:<7}{c['prize']:<8}{f['n_strong']:>6}"
              f"{f['rate_strong']:>8.3f}{f['n_teacher']:>7}{f['rate_teacher']:>7.3f}"
              f"{f['gap']:>+8.3f}{f['players_agree']:>5}{hold:>8}{wk:>8}")
        if f["kind"] == "PLAY" and abs(f["gap"]) >= 0.15:
            # "played >=1 card this turn" is coarse — a turn-length artifact can fake a
            # gap. Intensity (cards played per PLAY-offering decision) cannot.
            cellv = tuple(f["cell"][a] for a in axes)
            parts = []
            for tag, rws in (("strong", discovery), ("650", teacher),
                             ("weak", weak), ("holdout", holdout)):
                sel = [r for r in rws
                       if tuple(str(getattr(r, a)) for a in axes) == cellv and r.play_offered]
                if sel:
                    parts.append(f"{tag}={sum(r.play_taken for r in sel)/sum(r.play_offered for r in sel):.3f}")
            print(f"    -> play-intensity: {'  '.join(parts)}")

    # --- K-D ----------------------------------------------------------------------
    survivors = [
        f for f in findings
        if abs(f["gap"]) >= args.min_gap
        and f["players_agree"] >= args.min_players
        and f["rate_holdout"] is not None
        and abs(f["rate_holdout"] - f["rate_strong"]) <= 0.15
        # deck-confound control: the strong pool must also depart from the OTHER weak
        # players (different decks), in the same direction, by at least half the gap.
        and f["rate_weak"] is not None
        and (f["rate_strong"] - f["rate_weak"]) * f["gap"] > 0
        and abs(f["rate_strong"] - f["rate_weak"]) >= abs(f["gap"]) / 2
    ]
    print(f"\n{'='*78}\nK-D gate: support>={args.min_support}, players_agree>={args.min_players}, "
          f"|gap|>={args.min_gap}, holdout within 0.15, weak-pool control agrees")
    if survivors:
        print(f"SURVIVORS: {len(survivors)} - candidate patterns for Fase B\n")
        for f in survivors[:10]:
            print(f"  {f['kind']:<9} {f['cell']}  strong {f['rate_strong']:.3f} vs "
                  f"650 {f['rate_teacher']:.3f} (gap {f['gap']:+.3f}), "
                  f"{f['players_agree']} players agree, holdout {f['rate_holdout']:.3f}, "
                  f"weak-pool {f['rate_weak']:.3f}")
            pp = {k: v for k, v in f["per_player"].items() if v[0] >= 10}
            print("      per-player: " + "  ".join(
                f"{k}={v[1]:.2f}(n{v[0]})" for k, v in pp.items()))
            if f["kind"] == "PLAY":
                # "played at least one card" is coarse; show the INTENSITY too, so a
                # gap can't be an artifact of turn length.
                cellv = tuple(f["cell"][a] for a in axes)
                for tag, rows in (("strong", discovery), ("650", teacher), ("weak", weak)):
                    sel = [r for r in rows
                           if tuple(str(getattr(r, a)) for a in axes) == cellv and r.play_offered]
                    if sel:
                        inten = sum(r.play_taken for r in sel) / sum(r.play_offered for r in sel)
                        print(f"      play-intensity {tag}: {inten:.3f} "
                              f"(cards played per PLAY-offering decision, n_turns={len(sel)})")
    else:
        print("NO SURVIVORS -> K-D fires: no discrete, cross-player-replicated behavioural "
              "divergence at this resolution. Document as a negative and STOP.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"meta": meta, "axes": list(axes), "holdout": args.holdout,
         "findings": findings, "survivors": survivors}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
