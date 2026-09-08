"""M32 residual-fix, decisive check — does a candidate MLP actually fire Powerful
Hand earlier against Marnie's Grimmsnarl, on the real 27 live games -- WITHOUT
regressing on a matchup the clone already wins (Mega Lucario, 9 real games)?

Generalized (M32 plan §0) from the Attempt #1 version: CLI-parametrized instead of
hardcoded, plus a mandatory negative-control pass. Attempt #1's offline proxy showed
a small positive lean on the targeted slice that did NOT survive this exact
re-score (fired LATER, not earlier). Do not trust any candidate on the proxy alone
-- this script is the gate.

Sanity check baked in: the SHIPPED V1 model, re-scored through this harness, must
reproduce bootstrap_ph_turn.py's raw numbers (Grimmsnarl median turn 6, n=24;
Lucario median turn 4, n=9) -- if it doesn't, the harness itself is broken.

Usage:
  uv run --group dev python scratchpad/counterfactual_ph_timing.py \\
      --candidate scratchpad/matchup_raw.json --candidate-profile ALAKAZAM_MATCHUP

  # against a different baseline / replay pool / archetype marker:
  uv run --group dev python scratchpad/counterfactual_ph_timing.py \\
      --candidate build/specialist_raw.json --candidate-profile ALAKAZAM_SPECIALIST \\
      --baseline data/models/bc_alakazam_mlp.json --baseline-profile ALAKAZAM \\
      --marker 648 --control-marker 678 --replays-dir replays/55011997
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.imitation.kaggle_replay import ReplayDecision, _ACTIVE, _DECK_LEN, _strip_observation, player_seats
from ptcg_ai.imitation import features as F
from ptcg_ai.observation.models import OptionKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

import diagnose_mlp as dm
import extract_top_decks as et

DEFAULT_REPLAYS_DIR = "replays/55011997"
DEFAULT_BASELINE = "data/models/bc_alakazam_mlp.json"
DEFAULT_GRIMM_MARKER = 648
DEFAULT_LUCARIO_MARKER = 678


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


def mlp_ensemble_scores(X: np.ndarray, members: list[dict]) -> np.ndarray:
    """sum over members of relu(X @ W1 + b1) @ w2 + b2 -- matches colab_train_mlp.Split.scores."""
    total = np.zeros(X.shape[0], dtype=np.float64)
    for m in members:
        W1 = np.asarray(m["W1"], dtype=np.float64)
        b1 = np.asarray(m["b1"], dtype=np.float64)
        w2 = np.asarray(m["w2"], dtype=np.float64)
        b2 = float(m["b2"])
        h = np.maximum(X @ W1 + b1, 0.0)
        total += h @ w2 + b2
    return total


def ph_turns_by_model(decisions, parser, cards, profile, members) -> dict[str, int]:
    """First turn each game's MAIN decisions would fire Powerful Hand, per the given
    ensemble -- a counterfactual re-score, not the recorded action."""
    powerful_hand = dm.POWERFUL_HAND
    ph_turn: dict[str, int] = {}
    for dec in decisions:
        if dec.game_id in ph_turn:
            continue
        if SelectContextKind(dec.context) is not SelectContextKind.MAIN:
            continue
        obs = parser.parse(dec.raw_observation)
        if obs.select is None or obs.current is None or len(obs.select.option) < 2:
            continue
        gs = GameState.build(obs, cards)
        X = np.asarray(F.featurize_decision(profile, obs.select, obs, gs, cards), dtype=np.float64)
        scores = mlp_ensemble_scores(X, members)
        pick = int(np.argmax(scores))
        opt = obs.select.option[pick]
        if opt.type is OptionKind.ATTACK and opt.attackId == powerful_hand:
            ph_turn[dec.game_id] = obs.current.turn
    return ph_turn


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


def _report(name: str, turns: dict[str, int], n_games: int) -> None:
    vals = sorted(turns.values())
    print(f"\n{name}: fired in {len(vals)}/{n_games} games")
    print(f"  raw turns: {vals}")
    if vals:
        print(f"  median: {statistics.median(vals)}  mean: {statistics.mean(vals):.2f}")
        print(f"  histogram: {dict(sorted(Counter(vals).items()))}")


def _load_members(payload_path: str) -> list[dict]:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    # accept either a raw {"members": [...]} dump (--emit-raw) or a full bundle payload
    if "members" in payload:
        return payload["members"]
    return payload["contexts"]["MAIN"]["members"]


def _run_slice(label: str, replays_dir: str, marker: int, parser, cards,
                baseline_profile, baseline_members, candidate_profile, candidate_members) -> None:
    games = _find_games(replays_dir, marker)
    print(f"\n{'#' * 70}\n{label}  ({len(games)} real games, marker={marker})\n{'#' * 70}")
    decisions = []
    for f, seat in games:
        decisions.extend(_decisions_for_seat(f, seat))

    turns_base = ph_turns_by_model(decisions, parser, cards, baseline_profile, baseline_members)
    turns_cand = ph_turns_by_model(decisions, parser, cards, candidate_profile, candidate_members)

    _report("BASELINE (re-scored)", turns_base, len(games))
    _report("CANDIDATE (re-scored)", turns_cand, len(games))

    common = sorted(set(turns_base) & set(turns_cand))
    if common:
        deltas = [turns_cand[g] - turns_base[g] for g in common]
        print(f"\nPAIRED (both fired, n={len(common)}): candidate - baseline turn delta")
        print(f"  mean delta: {statistics.mean(deltas):+.2f}  "
              f"(negative = candidate fires EARLIER)")
        print(f"  per-game: {dict(zip(common, deltas))}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True, help="path to candidate weights "
                     "(--emit-raw dump {members:[...]} or a full bundle payload)")
    ap.add_argument("--candidate-profile", default="ALAKAZAM_MATCHUP")
    ap.add_argument("--baseline", default=DEFAULT_BASELINE, help="shipped/reference weights")
    ap.add_argument("--baseline-profile", default="ALAKAZAM")
    ap.add_argument("--replays-dir", default=DEFAULT_REPLAYS_DIR)
    ap.add_argument("--marker", type=int, default=DEFAULT_GRIMM_MARKER,
                     help="targeted-slice opponent card id (Marnie's Grimmsnarl ex = 648)")
    ap.add_argument("--control-marker", type=int, default=DEFAULT_LUCARIO_MARKER,
                     help="negative-control opponent card id, a matchup the clone already "
                          "wins (Mega Lucario = 678) -- the candidate must NOT fire later here")
    ap.add_argument("--skip-control", action="store_true", help="skip the control-marker pass")
    args = ap.parse_args()

    candidate_members = _load_members(args.candidate)
    baseline_members = _load_members(args.baseline)

    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()

    baseline_profile = get_profile(args.baseline_profile)
    candidate_profile = get_profile(args.candidate_profile)

    _run_slice("TARGETED SLICE", args.replays_dir, args.marker, parser, cards,
               baseline_profile, baseline_members, candidate_profile, candidate_members)

    if not args.skip_control:
        _run_slice("NEGATIVE CONTROL (must not regress)", args.replays_dir, args.control_marker,
                    parser, cards, baseline_profile, baseline_members,
                    candidate_profile, candidate_members)
        print(f"\n{'=' * 70}\nPASS RULE: candidate fires earlier (lower median/mean turn) on the "
              f"TARGETED SLICE AND does not fire meaningfully later on the CONTROL.\n{'=' * 70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
