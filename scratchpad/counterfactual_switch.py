"""M32 Strategy B, decisive check — replay the real 27 clone-vs-Grimmsnarl games
(and the 9-game Mega Lucario control) through the ACTUAL shipped ImitationPolicy
with the "mlp_switch" scorer wired in (general = shipped V1 ALAKAZAM ensemble,
UNTOUCHED; specialist = the Grimmsnarl-filtered ensemble from
scratchpad/specialist_raw.json), and compare Powerful-Hand first-use timing
against the plain V1 policy on the same games.

Unlike Attempt #1's and the earlier standalone re-scoring script, this drives
the REAL ImitationPolicy.choose() code path (routing, ranking, exception
handling) end to end -- not a hand-rolled numpy re-implementation -- so it is
the most faithful pre-ladder check available.

Usage:
  uv run --group dev python scratchpad/counterfactual_switch.py
"""
from __future__ import annotations

import glob
import json
import statistics
from collections import Counter
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.deck_profiles import get_profile
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
LUCARIO_MARKER = 678
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


def ph_turns_by_policy(decisions, parser, cards, pol: ImitationPolicy) -> dict[str, int]:
    """First turn each game fires Powerful Hand, replaying through the REAL
    ImitationPolicy.choose() -- not a re-implemented scorer."""
    ph_turn: dict[str, int] = {}
    for dec in decisions:
        if dec.game_id in ph_turn:
            continue
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
            ph_turn[dec.game_id] = obs.current.turn
    return ph_turn


def _report(name: str, turns: dict[str, int], n_games: int) -> None:
    vals = sorted(turns.values())
    print(f"\n{name}: fired in {len(vals)}/{n_games} games")
    print(f"  raw turns: {vals}")
    if vals:
        print(f"  median: {statistics.median(vals)}  mean: {statistics.mean(vals):.2f}")
        print(f"  histogram: {dict(sorted(Counter(vals).items()))}")


def _run_slice(label: str, marker: int, replays_dir: str, parser, cards, pol_v1, pol_switch) -> None:
    games = _find_games(replays_dir, marker)
    print(f"\n{'#' * 70}\n{label}  ({len(games)} real games, marker={marker})\n{'#' * 70}")
    decisions = []
    for f, seat in games:
        decisions.extend(_decisions_for_seat(f, seat))

    turns_v1 = ph_turns_by_policy(decisions, parser, cards, pol_v1)
    turns_switch = ph_turns_by_policy(decisions, parser, cards, pol_switch)

    _report("SHIPPED V1 (re-played, sanity check)", turns_v1, len(games))
    _report("SWITCH (real ImitationPolicy, mlp_switch)", turns_switch, len(games))

    common = sorted(set(turns_v1) & set(turns_switch))
    if common:
        deltas = [turns_switch[g] - turns_v1[g] for g in common]
        print(f"\nPAIRED (both fired, n={len(common)}): switch - V1 turn delta")
        print(f"  mean delta: {statistics.mean(deltas):+.2f}  (negative = switch fires EARLIER)")
        print(f"  per-game: {dict(zip(common, deltas))}")


def main() -> int:
    v1_payload = json.loads(SHIPPED_V1.read_text(encoding="utf-8"))
    specialist_members = json.loads(SPECIALIST_RAW.read_text(encoding="utf-8"))["members"]

    switch_payload = dict(v1_payload)
    switch_payload["contexts"] = dict(v1_payload["contexts"])
    switch_payload["contexts"]["MAIN"] = {
        "kind": "mlp_switch",
        "specialist_profile": "ALAKAZAM_SPECIALIST",
        "detector_ids": [MUNKIDORI_ID, GRIMM_MARKER],
        "general": v1_payload["contexts"]["MAIN"],  # byte-identical to the shipped model
        "specialist": {"kind": "mlp_ensemble", "members": specialist_members},
    }

    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()

    pol_v1 = ImitationPolicy(v1_payload)
    pol_switch = ImitationPolicy(switch_payload)

    _run_slice("TARGETED SLICE (Marnie's Grimmsnarl)", GRIMM_MARKER, "replays/55011997",
               parser, cards, pol_v1, pol_switch)
    _run_slice("NEGATIVE CONTROL (Mega Lucario, must not regress)", LUCARIO_MARKER, "replays/55011997",
               parser, cards, pol_v1, pol_switch)

    print(f"\n{'=' * 70}")
    print(f"switch routed to the SPECIALIST on {pol_switch._route_used} MAIN decisions total "
          f"(across both slices above)")
    print("PASS RULE: switch fires earlier on Grimmsnarl AND does not regress on Lucario.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
