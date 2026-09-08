"""M32 Strategy B follow-up: WHY did the switch never fire Powerful Hand in the
2 Grimmsnarl games where the shipped V1 policy did? Replays those specific
games turn-by-turn through the REAL ImitationPolicy (both V1 and switch),
reporting every MAIN decision: whether Powerful Hand was a legal option, which
scorer/route fired, what it chose instead, and the score margin -- to tell
apart (a) genuine undertraining ("got lost", no coherent preference), (b) a
deliberate different line that still won/lost, or (c) a detector edge case.

READ-ONLY. Run:
  uv run --group dev python scratchpad/diagnose_switch_misses.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.kaggle_replay import (
    ReplayDecision, _ACTIVE, _DECK_LEN, _strip_observation, player_seats,
)
from ptcg_ai.imitation.policy import ImitationPolicy
from ptcg_ai.observation.models import OptionKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser

import diagnose_mlp as dm
import extract_top_decks as et

SHIPPED_V1 = Path("data/models/bc_alakazam_mlp.json")
SPECIALIST_RAW = Path("scratchpad/specialist_raw.json")
MUNKIDORI_ID = 112
GRIMM_MARKER = 648
POWERFUL_HAND = dm.POWERFUL_HAND


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


def _find_games(replays_dir: str, marker: int) -> list[tuple[Path, int]]:
    out = []
    for fp in sorted(glob.glob(f"{replays_dir}/*.json")):
        if "metadata" in fp:
            continue
        f = Path(fp)
        d = json.loads(f.read_text(encoding="utf-8"))
        seats = player_seats(d, dm.OUR)
        if len(seats) != 1:
            continue
        our = seats[0]
        deck = et.extract_deck(d, 1 - our)
        if deck and marker in deck:
            out.append((f, our))
    return out


def ph_turn_first(decisions, parser, cards, pol) -> int | None:
    for dec in decisions:
        if SelectContextKind(dec.context) is not SelectContextKind.MAIN:
            continue
        obs = parser.parse(dec.raw_observation)
        if obs.select is None or obs.current is None or len(obs.select.option) < 2:
            continue
        ctx = DecisionContext(raw=dec.raw_observation, observation=obs, cards=cards)
        try:
            pick = pol.choose(ctx)
        except Exception:
            continue
        if not pick:
            continue
        opt = obs.select.option[pick[0]]
        if opt.type is OptionKind.ATTACK and opt.attackId == POWERFUL_HAND:
            return obs.current.turn
    return None


def main() -> int:
    v1_payload = json.loads(SHIPPED_V1.read_text(encoding="utf-8"))
    specialist_members = json.loads(SPECIALIST_RAW.read_text(encoding="utf-8"))["members"]
    switch_payload = dict(v1_payload)
    switch_payload["contexts"] = dict(v1_payload["contexts"])
    switch_payload["contexts"]["MAIN"] = {
        "kind": "mlp_switch",
        "specialist_profile": "ALAKAZAM_SPECIALIST",
        "detector_ids": [MUNKIDORI_ID, GRIMM_MARKER],
        "general": v1_payload["contexts"]["MAIN"],
        "specialist": {"kind": "mlp_ensemble", "members": specialist_members},
    }

    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()

    games = _find_games("replays/55011997", GRIMM_MARKER)
    print(f"scanning {len(games)} Grimmsnarl games for V1-fired / switch-missed cases...\n")

    missed = []
    for f, seat in games:
        decisions = list(_decisions_for_seat(f, seat))
        pol_v1 = ImitationPolicy(v1_payload)
        pol_switch = ImitationPolicy(switch_payload)
        t_v1 = ph_turn_first(decisions, parser, cards, pol_v1)
        t_switch = ph_turn_first(decisions, parser, cards, pol_switch)
        status = "MATCH" if (t_v1 is None) == (t_switch is None) else "MISS" if t_v1 and not t_switch else "-"
        print(f"{f.stem}: V1={t_v1}  switch={t_switch}  {status}")
        if t_v1 is not None and t_switch is None:
            missed.append((f, seat))

    print(f"\n{'=' * 70}\n{len(missed)} games where V1 fired but the switch never did\n{'=' * 70}")

    for f, seat in missed:
        print(f"\n{'#' * 70}\nGAME {f.stem}\n{'#' * 70}")
        decisions = list(_decisions_for_seat(f, seat))
        pol_switch = ImitationPolicy(switch_payload)
        won = None
        for dec in decisions:
            if SelectContextKind(dec.context) is not SelectContextKind.MAIN:
                continue
            obs = parser.parse(dec.raw_observation)
            if obs.select is None or obs.current is None or len(obs.select.option) < 2:
                continue
            won = dec.won
            ph_legal = any(
                o.type is OptionKind.ATTACK and o.attackId == POWERFUL_HAND for o in obs.select.option
            )
            ctx = DecisionContext(raw=dec.raw_observation, observation=obs, cards=cards)
            try:
                pick = pol_switch.choose(ctx)
            except Exception as exc:
                print(f"  turn {obs.current.turn}: EXCEPTION {exc!r}")
                continue
            chosen_opt = obs.select.option[pick[0]] if pick else None
            chosen_desc = f"{chosen_opt.type.name}" + (
                f" attackId={chosen_opt.attackId}" if chosen_opt and chosen_opt.type is OptionKind.ATTACK else ""
            ) if chosen_opt else "NONE"
            print(f"  turn {obs.current.turn:>2}: PH_legal={ph_legal!s:<5} chose={chosen_desc:<30} "
                  f"note={pol_switch.last_note}")
        print(f"  final: won={won}  route_used_total={pol_switch._route_used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
