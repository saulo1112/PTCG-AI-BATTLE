"""M33 follow-up — if he fires Powerful Hand 99.6% of the turns it is LEGAL,
then the whole game is decided by WHEN IT BECOMES LEGAL. This measures the
assembly bottleneck: what has to be true for PH to be legal, and how fast the
teacher gets there vs the clone.

PH costs {P} and belongs to Alakazam, so it needs BOTH:
  (a) an Alakazam in the ACTIVE spot (not benched), and
  (b) a {P}-providing energy attached to it,
out of a deck with only 7 energy cards in 60.

Reports, per side, the first turn each precondition is met and the first turn PH
is actually legal -- isolating which precondition is the real bottleneck.

READ-ONLY. Run:
  uv run --group dev python scratchpad/analyze_ph_availability.py
"""

from __future__ import annotations

import collections
import glob
import json
import statistics
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.kaggle_replay import player_seats
from ptcg_ai.observation.models import EnergyKind, OptionKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

import analyze_yushin as ay
import diagnose_mlp as dm
import extract_top_decks as et

ALAKAZAM_ID = 743
POWERFUL_HAND = dm.POWERFUL_HAND
GRIMM_MARKER = 648


def _p_energy(pkmn) -> int:
    return sum(1 for e in pkmn.energies if e is EnergyKind.PSYCHIC)


def measure(games, parser, cards, label, arch_filter=None):
    first_zam_bench: dict[str, int] = {}   # Alakazam anywhere in play
    first_zam_active: dict[str, int] = {}  # Alakazam in the ACTIVE spot
    first_zam_charged: dict[str, int] = {} # active Alakazam WITH {P}
    first_ph_legal: dict[str, int] = {}
    first_ph_fired: dict[str, int] = {}
    won: dict[str, bool] = {}

    for f, seat, arch, w in games:
        if arch_filter and arch != arch_filter:
            continue
        for dec in ay.decisions_for_seat(f, seat):
            if SelectContextKind(dec.context) is not SelectContextKind.MAIN:
                continue
            obs = parser.parse(dec.raw_observation)
            if obs.select is None or obs.current is None:
                continue
            gs = GameState.build(obs, cards)
            g, t = dec.game_id, obs.current.turn
            won[g] = dec.won
            me = gs.me
            if me is None:
                continue
            in_play = [p for p in me.active if p is not None] + list(me.bench)
            if any(p.id == ALAKAZAM_ID for p in in_play):
                first_zam_bench.setdefault(g, t)
            act = gs.my_active
            if act is not None and act.id == ALAKAZAM_ID:
                first_zam_active.setdefault(g, t)
                if _p_energy(act) >= 1:
                    first_zam_charged.setdefault(g, t)
            if any(o.type is OptionKind.ATTACK and o.attackId == POWERFUL_HAND
                   for o in obs.select.option):
                first_ph_legal.setdefault(g, t)
            if any(0 <= a < len(obs.select.option)
                   and obs.select.option[a].type is OptionKind.ATTACK
                   and obs.select.option[a].attackId == POWERFUL_HAND
                   for a in dec.action):
                first_ph_fired.setdefault(g, t)

    n = len(won)
    print(f"\n  {label}  ({n} games{' vs ' + arch_filter if arch_filter else ''})")
    print(f"    {'milestone':<34}{'games':>8}{'mean turn':>12}{'median':>9}")
    for name, d in (("Alakazam anywhere in play", first_zam_bench),
                    ("Alakazam in ACTIVE spot", first_zam_active),
                    ("active Alakazam WITH {P}", first_zam_charged),
                    ("Powerful Hand LEGAL", first_ph_legal),
                    ("Powerful Hand FIRED", first_ph_fired)):
        vals = [float(v) for v in d.values()]
        if not vals:
            print(f"    {name:<34}{0:>8}{'-':>12}{'-':>9}")
            continue
        print(f"    {name:<34}{len(vals):>8}{statistics.mean(vals):>12.2f}"
              f"{statistics.median(vals):>9.1f}")
    # gap between legal and fired = hesitation (should be ~0 given 99.6%)
    both = [k for k in first_ph_legal if k in first_ph_fired]
    if both:
        lag = [float(first_ph_fired[k] - first_ph_legal[k]) for k in both]
        print(f"    lag LEGAL -> FIRED: mean {statistics.mean(lag):+.2f} turns "
              f"(0 = fires the moment it can)")
    return {"legal": first_ph_legal, "charged": first_zam_charged,
            "active": first_zam_active, "won": won}


def main() -> int:
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()

    print("=" * 78)
    print("PH AVAILABILITY — the real bottleneck, given he fires 99.6% of legal turns")
    print("=" * 78)

    tgames = ay.teacher_games()
    print(f"teacher: {len(tgames)} non-mirror games", flush=True)

    cgames = []
    for fp in sorted(glob.glob("replays/55011997/*.json")):
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
        cgames.append((f, seat, dm._archetype(deck), rew > 0))
    print(f"clone: {len(cgames)} games", flush=True)

    measure(tgames, parser, cards, "TEACHER — all matchups")
    measure(tgames, parser, cards, "TEACHER", arch_filter="Marnie's Grimmsnarl")
    measure(cgames, parser, cards, "CLONE — all matchups")
    measure(cgames, parser, cards, "CLONE", arch_filter="Marnie's Grimmsnarl")

    # teacher wins vs losses, on the assembly milestones
    print("\n" + "-" * 78)
    print("  TEACHER vs Grimmsnarl: assembly speed in WINS vs LOSSES")
    print("-" * 78)
    gw = [g for g in tgames if g[2] == "Marnie's Grimmsnarl" and g[3]]
    gl = [g for g in tgames if g[2] == "Marnie's Grimmsnarl" and not g[3]]
    measure(gw, parser, cards, "TEACHER Grimmsnarl WINS")
    measure(gl, parser, cards, "TEACHER Grimmsnarl LOSSES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
