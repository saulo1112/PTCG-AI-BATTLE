"""M32 Strategy B follow-up, root cause check: does YUSHIN HIMSELF avoid
Powerful Hand when Munkidori is energized, in his own real Grimmsnarl games?

diagnose_switch_misses.py found the specialist CONSISTENTLY chose something
other than Powerful Hand even when it was legal, many times in a row -- not
random noise, a learned preference. If the teacher's own labels show the same
avoidance, the specialist is faithfully imitating a real (if perhaps net-
suboptimal) teacher tendency, and no amount of better training data/warm-start
can produce a MORE aggressive policy than the teacher's own labels support --
that would require a different objective, not a better-trained BC model.

READ-ONLY. Run:
  uv run --group dev python scratchpad/diagnose_teacher_munkidori_avoidance.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.deck_profiles import EnergyKind, get_profile
from ptcg_ai.imitation.kaggle_replay import (
    ReplayDecision, _ACTIVE, _DECK_LEN, _strip_observation,
)
from ptcg_ai.observation.models import OptionKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

import diagnose_mlp as dm
import extract_top_decks as et

TEACHER_FOLDER = Path("replays/54773249")
MUNKIDORI_ID = 112
GRIMM_MARKER = 648
POWERFUL_HAND = dm.POWERFUL_HAND
DARK = EnergyKind.DARKNESS


def _decisions_for_seat(path: Path, seat: int):
    data = json.loads(path.read_text(encoding="utf-8"))
    game_id = path.stem
    steps = data.get("steps", [])
    rewards = data.get("rewards") or []
    won = seat < len(rewards) and (rewards[seat] or 0) > 0
    for i in range(len(steps) - 1):
        row = steps[i]; nxt = steps[i + 1]
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


def main() -> int:
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()

    yushin_deck = tuple(sorted(int(x) for x in dm.DECK_CSV.read_text(encoding="utf-8").split()))
    grimm_seats = {}
    for fp in sorted(glob.glob(str(TEACHER_FOLDER / "*.json"))):
        if "metadata" in fp:
            continue
        f = Path(fp)
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        decks = [et.extract_deck(d, s) for s in (0, 1)]
        seats = [s for s in (0, 1) if decks[s] and tuple(sorted(decks[s])) == yushin_deck]
        if len(seats) != 1:
            continue
        our = seats[0]
        if decks[1 - our] and GRIMM_MARKER in decks[1 - our]:
            grimm_seats[f] = our
    print(f"teacher's Grimmsnarl games: {len(grimm_seats)}")

    # counts: [energized][chose_ph] -> n, among decisions where PH was LEGAL
    counts = {True: [0, 0], False: [0, 0]}  # key=energized, value=[not_chosen, chosen]
    n_decisions_checked = 0

    for f, seat in grimm_seats.items():
        for dec in _decisions_for_seat(f, seat):
            if SelectContextKind(dec.context) is not SelectContextKind.MAIN:
                continue
            obs = parser.parse(dec.raw_observation)
            if obs.select is None or obs.current is None:
                continue
            ph_idx = [
                i for i, o in enumerate(obs.select.option)
                if o.type is OptionKind.ATTACK and o.attackId == POWERFUL_HAND
            ]
            if not ph_idx:
                continue
            n_decisions_checked += 1
            gs = GameState.build(obs, cards)
            opp = gs.opponent
            energized = False
            if opp is not None:
                for p in list(opp.active) + list(opp.bench):
                    if p is not None and p.id == MUNKIDORI_ID and any(e is DARK for e in p.energies):
                        energized = True
                        break
            chose_ph = any(a in ph_idx for a in dec.action)
            counts[energized][1 if chose_ph else 0] += 1

    print(f"\nMAIN decisions where Powerful Hand was LEGAL: {n_decisions_checked}")
    print(f"{'munkidori_energized':<22}{'chose PH':>10}{'chose other':>14}{'PH pick-rate':>14}")
    for energized in (False, True):
        not_chosen, chosen = counts[energized]
        total = not_chosen + chosen
        rate = chosen / total if total else float("nan")
        print(f"{str(energized):<22}{chosen:>10}{not_chosen:>14}{rate:>13.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
